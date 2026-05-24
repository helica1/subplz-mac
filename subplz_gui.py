"""Minimal PySide6 GUI for SubPlz.

Run from the project venv:
    .venv/bin/python subplz_gui.py

Single pair mode: pick one audio file + one epub/txt.
Folder mode: pick a directory; the GUI auto-detects pairs — subdirectories
first (each containing one audio + one text), falling back to filename-stem
pairs in a flat directory if no subdirs match.

The actual sync work runs in a worker thread via subprocess on the project's
`subplz` CLI, so the UI stays responsive. Progress is estimated from elapsed
time vs the per-backend realtime ratio (76× on MLX, 5.6× on CPU), with phase
transitions parsed from log output.
"""

from __future__ import annotations

import os
import re
import select
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QFont
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


PROJECT_ROOT = Path(__file__).resolve().parent
SUBPLZ_BIN = PROJECT_ROOT / ".venv" / "bin" / "subplz"

AUDIO_EXTS = {".mp3", ".m4a", ".m4b", ".mp4", ".aac", ".flac", ".ogg", ".wav", ".opus", ".mkv", ".webm"}
TEXT_EXTS = {".epub", ".txt", ".srt", ".vtt", ".ass"}

# Realtime ratios measured on this user's M-series Max with turbo + JA.
# Used to estimate per-book progress when we have no granular signal.
REALTIME_RATIO_MLX = 76.0
REALTIME_RATIO_CPU = 5.6


@dataclass
class Job:
    """One audiobook to sync.

    Either `workdir` is set (subdir mode — subplz uses -d) or `audio`/`text`
    are explicit (flat-folder mode — subplz uses --audio/--text/--output-dir).
    """

    label: str
    audio: Path
    text: Path
    workdir: Optional[Path] = None  # if set, use `-d workdir`; else use explicit files


def find_one_audio(folder: Path) -> Optional[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    return files[0] if files else None


def find_one_text(folder: Path) -> Optional[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in TEXT_EXTS]
    # Prefer epub > txt > others
    files.sort(key=lambda p: (
        0 if p.suffix.lower() == ".epub" else 1 if p.suffix.lower() == ".txt" else 2,
        p.name,
    ))
    return files[0] if files else None


def detect_pairs(folder: Path) -> list[Job]:
    """Subdirs-first auto-detection, falling back to flat-stem pairing."""
    pairs: list[Job] = []

    # Strategy 1: each subdir has one audio + one text
    for sub in sorted(folder.iterdir()):
        if not sub.is_dir() or sub.name.startswith("."):
            continue
        a = find_one_audio(sub)
        t = find_one_text(sub)
        if a and t:
            pairs.append(Job(label=sub.name, audio=a, text=t, workdir=sub))

    if pairs:
        return pairs

    # Strategy 2: flat folder, pair by stem
    audios = {p.stem: p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS}
    texts = {p.stem: p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in TEXT_EXTS}
    common = sorted(set(audios) & set(texts))
    for stem in common:
        pairs.append(Job(label=stem, audio=audios[stem], text=texts[stem]))

    return pairs


def get_audio_duration(path: Path) -> float:
    """Return audio duration in seconds via ffprobe, or 0 on failure."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            text=True,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PATH": f"/opt/homebrew/bin:{os.environ.get('PATH', '')}"},
        )
        return float(out.strip())
    except Exception:
        return 0.0


class SyncWorker(QObject):
    """Runs one or more subplz sync jobs on a background thread.

    Communication is one-way via Qt signals so the UI thread can stay
    responsive. We also keep a `_should_stop` flag the main thread can flip
    to cancel an in-flight subprocess.
    """

    status = Signal(str)               # current short status line
    book_progress = Signal(int)        # 0–100 for current book
    batch_progress = Signal(int, int)  # (completed, total)
    log = Signal(str)                  # raw log line
    book_done = Signal(str, bool)      # label, success
    all_done = Signal()
    failed = Signal(str)               # fatal error

    def __init__(self, jobs: list[Job], opts: dict):
        super().__init__()
        self.jobs = jobs
        self.opts = opts
        self._should_stop = False
        self._current_proc: Optional[subprocess.Popen] = None

    def stop(self):
        self._should_stop = True
        if self._current_proc and self._current_proc.poll() is None:
            try:
                self._current_proc.terminate()
            except Exception:
                pass

    def _build_cmd(self, job: Job) -> list[str]:
        cmd = [
            str(SUBPLZ_BIN),
            "sync",
            "--lang",
            self.opts["lang"],
            "--model",
            self.opts["model"],
            "--respect-grouping",
            "--overwrite",
            "--rerun",
        ]
        if self.opts.get("mlx"):
            cmd.append("--mlx")
        if job.workdir is not None:
            cmd += ["-d", str(job.workdir)]
        else:
            cmd += [
                "--audio",
                str(job.audio),
                "--text",
                str(job.text),
                "--output-dir",
                str(job.audio.parent),
            ]
        return cmd

    def run(self):
        total = len(self.jobs)
        if total == 0:
            self.failed.emit("No audiobook/epub pairs to process.")
            return

        for idx, job in enumerate(self.jobs):
            if self._should_stop:
                break
            self.batch_progress.emit(idx, total)
            self.book_progress.emit(0)
            self.status.emit(f"📚 {job.label}: preparing…")
            try:
                self._run_one(job)
                self.book_done.emit(job.label, True)
            except Exception as e:
                self.log.emit(f"❗ Error on '{job.label}': {e}")
                self.book_done.emit(job.label, False)
            self.batch_progress.emit(idx + 1, total)

        self.all_done.emit()

    def _run_one(self, job: Job):
        # Estimate transcription duration based on backend.
        audio_dur = get_audio_duration(job.audio)
        ratio = REALTIME_RATIO_MLX if self.opts.get("mlx") else REALTIME_RATIO_CPU
        expected_transcribe = max(audio_dur / ratio, 5.0) if audio_dur > 0 else 60.0

        cmd = self._build_cmd(job)
        self.log.emit(f"$ {' '.join(cmd)}")
        env = {**os.environ, "PATH": f"/opt/homebrew/bin:{os.environ.get('PATH', '')}"}

        self._current_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        transcribe_start: Optional[float] = None
        phase = "starting"

        # select-based read so we can emit periodic time-based progress even
        # when subprocess is silent during long MLX transcription.
        stdout = self._current_proc.stdout
        assert stdout is not None
        fd = stdout.fileno()

        while True:
            if self._should_stop:
                self._current_proc.terminate()
                break

            ready, _, _ = select.select([fd], [], [], 0.5)
            if ready:
                line = stdout.readline()
                if not line:
                    break
                line = line.rstrip("\n")
                # Strip ANSI escapes from the loguru-formatted lines
                line_clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
                self.log.emit(line_clean)

                # Phase detection from known log markers
                if "MLX-Whisper backend" in line_clean or "Transcribing with" in line_clean or "Attempting transcription" in line_clean:
                    if transcribe_start is None:
                        transcribe_start = time.time()
                        phase = "transcribing"
                        self.status.emit(f"📚 {job.label}: transcribing audio…")
                elif "Transcribing took:" in line_clean:
                    phase = "aligning"
                    self.book_progress.emit(92)
                    self.status.emit(f"📚 {job.label}: aligning text to audio…")
                elif "Greedy alignment:" in line_clean and "matched" not in line_clean:
                    self.book_progress.emit(95)
                elif "Successfully wrote" in line_clean:
                    self.book_progress.emit(100)
                    self.status.emit(f"📚 {job.label}: done.")
            elif phase == "transcribing" and transcribe_start is not None:
                elapsed = time.time() - transcribe_start
                pct = min(int(elapsed / expected_transcribe * 90), 90)
                self.book_progress.emit(pct)

        self._current_proc.wait()
        rc = self._current_proc.returncode
        self._current_proc = None
        if rc != 0 and not self._should_stop:
            raise RuntimeError(f"subplz exited with status {rc}")


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SubPlz")
        self.resize(720, 580)
        # Accept drag-and-drop anywhere on the window. We classify what was
        # dropped (folder / audio / text / mixed) and route to the right
        # mode + field, so the user doesn't have to think about which slot
        # to drop on.
        self.setAcceptDrops(True)

        self.thread: Optional[QThread] = None
        self.worker: Optional[SyncWorker] = None
        # Output folders produced by the most recent run, kept for the
        # "Open folder" button. We collect them as books complete and pick
        # the most relevant one to reveal in Finder.
        self._last_output_dirs: list[Path] = []

        # Mode selector
        self.mode_single = QRadioButton("Single pair (one audio + one epub)")
        self.mode_folder = QRadioButton("Folder (auto-detect pairs)")
        self.mode_single.setChecked(True)
        mode_group = QButtonGroup(self)
        mode_group.addButton(self.mode_single)
        mode_group.addButton(self.mode_folder)
        self.mode_single.toggled.connect(self._on_mode_changed)

        mode_box = QGroupBox("Mode")
        mb = QHBoxLayout()
        mb.addWidget(self.mode_single)
        mb.addWidget(self.mode_folder)
        mb.addStretch()
        mode_box.setLayout(mb)

        # Single-pair pickers
        self.audio_edit = QLineEdit()
        self.audio_edit.setPlaceholderText("Path to audio file (.mp3, .m4b, …)")
        audio_btn = QPushButton("Browse…")
        audio_btn.clicked.connect(self._pick_audio)
        self.text_edit = QLineEdit()
        self.text_edit.setPlaceholderText("Path to epub/txt file")
        text_btn = QPushButton("Browse…")
        text_btn.clicked.connect(self._pick_text)

        # Folder picker
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Folder containing audiobook subfolders or a flat list of pairs")
        folder_btn = QPushButton("Browse…")
        folder_btn.clicked.connect(self._pick_folder)
        self.folder_edit.textChanged.connect(self._refresh_folder_preview)
        self.folder_preview = QLabel("")
        self.folder_preview.setStyleSheet("color: #666; font-size: 11px;")

        self.single_box = QGroupBox("Files")
        sb = QVBoxLayout()
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Audio:"))
        row1.addWidget(self.audio_edit, 1)
        row1.addWidget(audio_btn)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Text:"))
        row2.addWidget(self.text_edit, 1)
        row2.addWidget(text_btn)
        sb.addLayout(row1)
        sb.addLayout(row2)
        self.single_box.setLayout(sb)

        self.folder_box = QGroupBox("Folder")
        fb = QVBoxLayout()
        row3 = QHBoxLayout()
        row3.addWidget(self.folder_edit, 1)
        row3.addWidget(folder_btn)
        fb.addLayout(row3)
        fb.addWidget(self.folder_preview)
        self.folder_box.setLayout(fb)
        self.folder_box.setVisible(False)

        # Settings
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "turbo",
            "large-v3",
            "large-v3-turbo",
            "small",
            "tiny",
        ])
        self.model_combo.setCurrentText("turbo")
        self.lang_edit = QLineEdit("ja")
        self.lang_edit.setMaximumWidth(60)
        self.mlx_check = QCheckBox("Use MLX (Apple Metal + Neural Engine)")
        self.mlx_check.setChecked(True)

        settings = QGroupBox("Settings")
        sg = QHBoxLayout()
        sg.addWidget(QLabel("Model:"))
        sg.addWidget(self.model_combo)
        sg.addSpacing(20)
        sg.addWidget(QLabel("Language:"))
        sg.addWidget(self.lang_edit)
        sg.addSpacing(20)
        sg.addWidget(self.mlx_check)
        sg.addStretch()
        settings.setLayout(sg)

        # Progress + status
        self.status_label = QLabel("Ready.")
        self.status_label.setFont(QFont("", weight=QFont.DemiBold))
        self.book_bar = QProgressBar()
        self.book_bar.setRange(0, 100)
        self.book_bar.setValue(0)
        self.book_bar.setFormat("Current book — %p%")
        self.batch_label = QLabel("Batch: 0 / 0 books")
        self.batch_bar = QProgressBar()
        self.batch_bar.setRange(0, 100)
        self.batch_bar.setValue(0)
        self.batch_bar.setFormat("Batch — %p%")

        # Buttons
        self.run_btn = QPushButton("▶  Start")
        self.run_btn.clicked.connect(self._on_run)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.cancel_btn.setEnabled(False)
        self.open_folder_btn = QPushButton("📂  Open output folder")
        self.open_folder_btn.clicked.connect(self._on_open_folder)
        self.open_folder_btn.setEnabled(False)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(self.open_folder_btn)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.cancel_btn)

        # Log
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Menlo, monospace", 11))
        self.log_view.setMinimumHeight(160)

        root = QVBoxLayout()
        root.addWidget(mode_box)
        root.addWidget(self.single_box)
        root.addWidget(self.folder_box)
        root.addWidget(settings)
        root.addWidget(self.status_label)
        root.addWidget(self.book_bar)
        root.addWidget(self.batch_label)
        root.addWidget(self.batch_bar)
        root.addLayout(btn_row)
        root.addWidget(self.log_view, 1)
        self.setLayout(root)

    # ----- Drag and drop -----

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls() and any(u.isLocalFile() for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        paths = [Path(u.toLocalFile()) for u in event.mimeData().urls() if u.isLocalFile()]
        if not paths:
            return

        folders = [p for p in paths if p.is_dir()]
        audio_files = [p for p in paths if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
        text_files = [p for p in paths if p.is_file() and p.suffix.lower() in TEXT_EXTS]

        # Folder dropped: switch to folder mode and load it.
        if folders:
            self.mode_folder.setChecked(True)
            self.folder_edit.setText(str(folders[0]))
            if len(paths) > 1:
                self._append_log(f"⚠️  Multiple items dropped; using folder '{folders[0].name}' and ignoring the rest.")
            event.acceptProposedAction()
            return

        # One audio + one text file: switch to single mode and load both.
        if len(audio_files) == 1 and len(text_files) == 1:
            self.mode_single.setChecked(True)
            self.audio_edit.setText(str(audio_files[0]))
            self.text_edit.setText(str(text_files[0]))
            event.acceptProposedAction()
            return

        # Only audio or only text: fill the matching field, switch to single mode.
        if len(audio_files) >= 1 and not text_files:
            self.mode_single.setChecked(True)
            self.audio_edit.setText(str(audio_files[0]))
            if len(audio_files) > 1:
                self._append_log(f"⚠️  Multiple audio files dropped; using '{audio_files[0].name}'.")
            event.acceptProposedAction()
            return

        if len(text_files) >= 1 and not audio_files:
            self.mode_single.setChecked(True)
            self.text_edit.setText(str(text_files[0]))
            if len(text_files) > 1:
                self._append_log(f"⚠️  Multiple text files dropped; using '{text_files[0].name}'.")
            event.acceptProposedAction()
            return

        # Mixed bag (e.g. 2 audio + 1 text) — fall back to filling what we can.
        if audio_files:
            self.audio_edit.setText(str(audio_files[0]))
        if text_files:
            self.text_edit.setText(str(text_files[0]))
        if audio_files or text_files:
            self.mode_single.setChecked(True)
            self._append_log(
                f"⚠️  Mixed drop ({len(audio_files)} audio + {len(text_files)} text). "
                f"Used the first of each. Drop a folder instead to batch."
            )
            event.acceptProposedAction()

    # ----- UI callbacks -----

    def _on_mode_changed(self):
        is_single = self.mode_single.isChecked()
        self.single_box.setVisible(is_single)
        self.folder_box.setVisible(not is_single)

    def _pick_audio(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select audio file", "",
            "Audio (*.mp3 *.m4a *.m4b *.mp4 *.aac *.flac *.ogg *.wav *.opus *.mkv);;All files (*)"
        )
        if p:
            self.audio_edit.setText(p)

    def _pick_text(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select epub or text file", "",
            "Text (*.epub *.txt *.srt *.vtt *.ass);;All files (*)"
        )
        if p:
            self.text_edit.setText(p)

    def _pick_folder(self):
        p = QFileDialog.getExistingDirectory(self, "Select folder")
        if p:
            self.folder_edit.setText(p)

    def _refresh_folder_preview(self):
        folder = self.folder_edit.text().strip()
        if not folder:
            self.folder_preview.setText("")
            return
        try:
            pairs = detect_pairs(Path(folder))
        except Exception as e:
            self.folder_preview.setText(f"⚠️  Couldn't scan folder: {e}")
            return
        if not pairs:
            self.folder_preview.setText("⚠️  No audio+text pairs detected.")
            return
        first = ", ".join(p.label for p in pairs[:3])
        more = f" (+{len(pairs)-3} more)" if len(pairs) > 3 else ""
        self.folder_preview.setText(f"✅  Found {len(pairs)} pair(s): {first}{more}")

    def _on_run(self):
        if not SUBPLZ_BIN.exists():
            QMessageBox.critical(
                self, "subplz not found",
                f"Expected the subplz CLI at:\n  {SUBPLZ_BIN}\n\nRun `.venv/bin/pip install -e .` from the project root first."
            )
            return

        jobs = self._collect_jobs()
        if not jobs:
            QMessageBox.warning(self, "Nothing to do", "No audiobook/text pairs were found.")
            return

        opts = {
            "model": self.model_combo.currentText(),
            "lang": self.lang_edit.text().strip() or "ja",
            "mlx": self.mlx_check.isChecked(),
        }

        self.log_view.clear()
        self.batch_label.setText(f"Batch: 0 / {len(jobs)} books")
        self.book_bar.setValue(0)
        self.batch_bar.setValue(0)
        self.status_label.setText(f"Starting — {len(jobs)} book(s) queued…")

        # Record output folders for the "Open folder" button. For -d mode it's
        # the workdir; for explicit --audio mode it's the audio's parent.
        self._last_output_dirs = [
            job.workdir if job.workdir is not None else job.audio.parent
            for job in jobs
        ]
        self.open_folder_btn.setEnabled(False)

        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        self.thread = QThread()
        self.worker = SyncWorker(jobs, opts)
        self.worker.moveToThread(self.thread)

        self.worker.status.connect(self.status_label.setText)
        self.worker.book_progress.connect(self.book_bar.setValue)
        self.worker.batch_progress.connect(self._on_batch_progress)
        self.worker.log.connect(self._append_log)
        self.worker.book_done.connect(self._on_book_done)
        self.worker.all_done.connect(self._on_all_done)
        self.worker.failed.connect(self._on_failed)

        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def _on_cancel(self):
        if self.worker:
            self.worker.stop()
            self.status_label.setText("Cancelling…")

    def _collect_jobs(self) -> list[Job]:
        if self.mode_single.isChecked():
            a = self.audio_edit.text().strip()
            t = self.text_edit.text().strip()
            if not a or not t:
                return []
            ap, tp = Path(a), Path(t)
            if not ap.exists() or not tp.exists():
                return []
            # If both files share the same parent directory and that directory
            # has nothing else competing for attention, use -d mode; otherwise
            # use explicit --audio / --text.
            if ap.parent == tp.parent and len(list(ap.parent.iterdir())) <= 6:
                return [Job(label=ap.stem, audio=ap, text=tp, workdir=ap.parent)]
            return [Job(label=ap.stem, audio=ap, text=tp)]
        else:
            folder = self.folder_edit.text().strip()
            if not folder or not Path(folder).exists():
                return []
            return detect_pairs(Path(folder))

    def _append_log(self, line: str):
        # Trim to keep the widget responsive on long runs
        if self.log_view.document().blockCount() > 2000:
            cursor = self.log_view.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            for _ in range(500):
                cursor.select(cursor.SelectionType.LineUnderCursor)
                cursor.removeSelectedText()
                cursor.deleteChar()
        self.log_view.append(line)

    def _on_batch_progress(self, done: int, total: int):
        self.batch_label.setText(f"Batch: {done} / {total} books")
        pct = int(done / total * 100) if total else 0
        self.batch_bar.setValue(pct)

    def _on_book_done(self, label: str, success: bool):
        marker = "✅" if success else "❌"
        self._append_log(f"{marker} Finished: {label}")

    def _on_all_done(self):
        self.status_label.setText("✨ All done.")
        self.book_bar.setValue(100)
        self.batch_bar.setValue(100)
        # Enable "Open folder" if we have at least one output dir on disk.
        if any(p.exists() for p in self._last_output_dirs):
            self.open_folder_btn.setEnabled(True)
        self._teardown_thread()

    def _on_open_folder(self):
        """Open the output folder(s) in Finder.

        Multi-book batch: open each unique parent directory once. We use
        `open -R <file>` when an SRT exists so Finder reveals the actual
        output file; otherwise `open <dir>` for the directory itself.
        """
        if not self._last_output_dirs:
            return
        opened: set[str] = set()
        for d in self._last_output_dirs:
            if not d.exists():
                continue
            # Look for an SRT inside this dir to reveal a concrete file.
            srts = sorted(d.glob("*.srt"), key=lambda p: p.stat().st_mtime, reverse=True)
            target = srts[0] if srts else d
            key = str(target)
            if key in opened:
                continue
            opened.add(key)
            try:
                if target.is_file():
                    subprocess.run(["open", "-R", str(target)], check=False)
                else:
                    subprocess.run(["open", str(target)], check=False)
            except Exception as e:
                self._append_log(f"⚠️  Couldn't open '{target}': {e}")

    def _on_failed(self, msg: str):
        self.status_label.setText(f"❌ {msg}")
        QMessageBox.critical(self, "Error", msg)
        self._teardown_thread()

    def _teardown_thread(self):
        if self.thread:
            self.thread.quit()
            self.thread.wait(2000)
            self.thread = None
            self.worker = None
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
