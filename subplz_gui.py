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
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QFont
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


PROJECT_ROOT = Path(__file__).resolve().parent
SUBPLZ_BIN = PROJECT_ROOT / ".venv" / "bin" / "subplz"
# In-flight TTS run registry. Single-job: written on Generate, cleared on
# success. On launch, MainWindow checks for it and offers to resume.
ACTIVE_JOB_PATH = Path.home() / ".subplz" / "active_job.json"

AUDIO_EXTS = {".mp3", ".m4a", ".m4b", ".mp4", ".aac", ".flac", ".ogg", ".wav", ".opus", ".mkv", ".webm"}
TEXT_EXTS = {".epub", ".txt", ".srt", ".vtt", ".ass"}
# SRS mode is audio+SRT only — epub doesn't have timestamps to slice on.
SRS_TEXT_EXTS = {".srt", ".vtt", ".ass"}

# Realtime ratios measured on this user's M-series Max with turbo + JA.
# Used to estimate per-book progress when we have no granular signal.
REALTIME_RATIO_MLX = 76.0
REALTIME_RATIO_CPU = 5.6


@dataclass
class Job:
    """One audiobook to process.

    For sync: either `workdir` is set (subdir mode — subplz uses -d) or
    `audio`/`text` are explicit (flat-folder mode — subplz uses
    --audio/--text/--output-dir).

    For srs: only `audio` + `text` are used (-d isn't a thing for srs);
    workdir is ignored.
    """

    label: str
    audio: Path
    text: Path
    workdir: Optional[Path] = None  # if set, use `-d workdir`; else use explicit files


def find_one_audio(folder: Path) -> Optional[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    return files[0] if files else None


def find_one_text(folder: Path, exts: set[str] = TEXT_EXTS) -> Optional[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    # Prefer epub > txt > others (irrelevant for srs which only accepts srt/vtt/ass)
    files.sort(key=lambda p: (
        0 if p.suffix.lower() == ".epub" else 1 if p.suffix.lower() == ".txt" else 2,
        p.name,
    ))
    return files[0] if files else None


def detect_pairs(folder: Path, text_exts: set[str] = TEXT_EXTS) -> list[Job]:
    """Subdirs-first auto-detection, falling back to flat-stem pairing.

    `text_exts` controls which sidecar files count as the "text" half.
    Defaults to TEXT_EXTS (sync mode); pass SRS_TEXT_EXTS for srs mode.
    """
    pairs: list[Job] = []

    # Strategy 1: each subdir has one audio + one text
    for sub in sorted(folder.iterdir()):
        if not sub.is_dir() or sub.name.startswith("."):
            continue
        a = find_one_audio(sub)
        t = find_one_text(sub, text_exts)
        if a and t:
            pairs.append(Job(label=sub.name, audio=a, text=t, workdir=sub))

    if pairs:
        return pairs

    # Strategy 2: flat folder, pair by stem
    audios = {p.stem: p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS}
    texts = {p.stem: p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in text_exts}
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

    def _build_srs_cmd(self, job: Job) -> list[str]:
        out_dir = self.opts.get("srs_output_dir") or str(job.audio.parent)
        cmd = [
            str(SUBPLZ_BIN),
            "srs",
            "--audio",
            str(job.audio),
            "--text",
            str(job.text),
            "--output-dir",
            out_dir,
            "--pad-ms",
            str(self.opts.get("srs_pad_ms", 200)),
            "--fade-ms",
            str(self.opts.get("srs_fade_ms", 10)),
        ]
        if not self.opts.get("srs_vad_check", True):
            cmd.append("--no-vad-check")
        if not self.opts.get("srs_cover", True):
            cmd.append("--no-cover")
        cover_img = self.opts.get("srs_cover_image")
        if cover_img:
            cmd += ["--cover-image", cover_img]
        # "auto" is the CLI default; only emit when the user changed it, to
        # keep the command line readable in the log.
        br = self.opts.get("srs_bitrate", "auto")
        if br and br != "auto":
            cmd += ["--bitrate", br]
        ch = self.opts.get("srs_channels", "auto")
        if ch and ch != "auto":
            cmd += ["--channels", ch]
        # Per-job deck name; falls back to the audio stem inside the CLI.
        cmd += ["--deck-name", job.label]
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
        """Run all subprocess phases for a single job.

        Dispatches by `action`:
        - sync: one `subplz sync` invocation, progress 0-100
        - srs: one `subplz srs` invocation, progress 0-100
        - both: chained sync (0-50) then srs (50-100), with the SRT produced
          by sync auto-fed to srs as its `--text` input
        """
        action = self.opts.get("action") or "sync"
        if action == "both":
            self._run_phase(job, "sync", self._build_sync_cmd(job), 0, 50)
            if self._should_stop:
                return
            srt = self._locate_synced_srt(job)
            if srt is None:
                raise RuntimeError(
                    f"Sync finished but no SRT was found next to {job.audio.name}; "
                    f"cannot continue to deck building."
                )
            self.log.emit(f"➡  Sync produced: {srt}")
            srs_job = Job(label=job.label, audio=job.audio, text=srt, workdir=None)
            self._run_phase(srs_job, "srs", self._build_srs_cmd(srs_job), 50, 100)
        elif action == "srs":
            self._run_phase(job, "srs", self._build_srs_cmd(job), 0, 100)
        else:
            self._run_phase(job, "sync", self._build_sync_cmd(job), 0, 100)

    def _build_sync_cmd(self, job: Job) -> list[str]:
        """Direct sync builder — used by both single-action sync and the
        sync half of "both" mode. Mirrors the original _build_cmd's sync arm."""
        cmd = [
            str(SUBPLZ_BIN),
            "sync",
            "--lang", self.opts["lang"],
            "--model", self.opts["model"],
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
                "--audio", str(job.audio),
                "--text", str(job.text),
                "--output-dir", str(job.audio.parent),
            ]
        return cmd

    def _locate_synced_srt(self, job: Job) -> Optional[Path]:
        """Find the SRT that sync just wrote. sync places it next to the
        audio (or inside the workdir for `-d` mode) as `<stem>.srt` or
        `<stem>.<lang-ext>.srt`. We glob and pick the newest non-internal
        candidate so name variations don't trip us up."""
        parent = job.workdir if job.workdir is not None else job.audio.parent
        if not parent or not parent.exists():
            return None
        candidates = list(parent.glob(f"{job.audio.stem}*.srt"))
        if not candidates:
            candidates = list(parent.glob("*.srt"))
        # Skip internal files sync writes alongside the real output
        candidates = [
            p for p in candidates
            if not any(tag in p.name.lower() for tag in (".original.", ".subfail.", ".broken."))
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _run_phase(self, job: Job, kind: str, cmd: list[str], pct_start: int, pct_end: int):
        """Run one subprocess (sync or srs) and stream-parse progress lines.

        Per-phase progress (0-100) is rescaled into [pct_start, pct_end] so
        chained phases stack cleanly on a single book progress bar.
        """
        def emit(frac: float):
            """frac is 0-1 within phase; scaled to overall pct_start..pct_end."""
            pct = pct_start + max(0.0, min(frac, 1.0)) * (pct_end - pct_start)
            self.book_progress.emit(int(pct))

        # Estimate transcription duration based on backend (sync mode only).
        audio_dur = get_audio_duration(job.audio)
        ratio = REALTIME_RATIO_MLX if self.opts.get("mlx") else REALTIME_RATIO_CPU
        expected_transcribe = max(audio_dur / ratio, 5.0) if audio_dur > 0 else 60.0

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

        is_srs = kind == "srs"
        transcribe_start: Optional[float] = None
        phase = "starting"
        srs_progress_re = re.compile(r"(cutting|vad-check):\s*\d+%\|[^|]*\|\s*(\d+)/(\d+)")

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
                line_clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
                self.log.emit(line_clean)

                if is_srs:
                    if "Parsed" in line_clean and "cues from" in line_clean:
                        self.status.emit(f"📚 {job.label}: parsed cues")
                    elif "Slicing" in line_clean and "clips" in line_clean:
                        phase = "cutting"
                        self.status.emit(f"📚 {job.label}: cutting audio clips…")
                    elif "VAD edge check" in line_clean:
                        phase = "vad"
                        self.status.emit(f"📚 {job.label}: checking clip boundaries…")
                    elif "Wrote " in line_clean and ".apkg" in line_clean:
                        emit(1.0)
                        self.status.emit(f"📚 {job.label}: deck written.")
                    else:
                        m = srs_progress_re.search(line_clean)
                        if m:
                            kind_tqdm, cur, total = m.group(1), int(m.group(2)), int(m.group(3))
                            if total > 0:
                                frac_in_phase = cur / total
                                # Cutting: phase 0-45%, vad-check: phase 45-95%
                                if kind_tqdm == "cutting":
                                    emit(frac_in_phase * 0.45)
                                else:
                                    emit(0.45 + frac_in_phase * 0.50)
                else:
                    if "MLX-Whisper backend" in line_clean or "Transcribing with" in line_clean or "Attempting transcription" in line_clean:
                        if transcribe_start is None:
                            transcribe_start = time.time()
                            phase = "transcribing"
                            self.status.emit(f"📚 {job.label}: transcribing audio…")
                    elif "Transcribing took:" in line_clean:
                        phase = "aligning"
                        emit(0.92)
                        self.status.emit(f"📚 {job.label}: aligning text to audio…")
                    elif "Greedy alignment:" in line_clean and "matched" not in line_clean:
                        emit(0.95)
                    elif "Successfully wrote" in line_clean:
                        emit(1.0)
                        self.status.emit(f"📚 {job.label}: sub written.")
            elif not is_srs and phase == "transcribing" and transcribe_start is not None:
                elapsed = time.time() - transcribe_start
                frac = min(elapsed / expected_transcribe * 0.90, 0.90)
                emit(frac)

        self._current_proc.wait()
        rc = self._current_proc.returncode
        self._current_proc = None
        if rc != 0 and not self._should_stop:
            raise RuntimeError(f"subplz {kind} exited with status {rc}")


class SyncTab(QWidget):
    """The original SubPlz UI — audio+text→SRT and Anki deck building.

    Wrapped as a tab inside MainWindow alongside the TTS tab. Drag-and-drop
    works on this widget directly when it's the active tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Accept drag-and-drop anywhere on this tab. We classify what was
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

        # Action selector (what to do with the audio+text pair)
        self.action_sync = QRadioButton("Sync subtitles (audio + epub → SRT)")
        self.action_srs = QRadioButton("Build Anki deck (audio + SRT → .apkg)")
        self.action_both = QRadioButton("Sync + Build deck (audio + epub → SRT + .apkg)")
        self.action_sync.setChecked(True)
        action_group = QButtonGroup(self)
        action_group.addButton(self.action_sync)
        action_group.addButton(self.action_srs)
        action_group.addButton(self.action_both)
        # Either toggled() fires twice on radio change (one off, one on); guard
        # against duplicate work by connecting only the becoming-true edge.
        self.action_sync.toggled.connect(lambda c: c and self._on_action_changed())
        self.action_srs.toggled.connect(lambda c: c and self._on_action_changed())
        self.action_both.toggled.connect(lambda c: c and self._on_action_changed())

        action_box = QGroupBox("Action")
        ab = QHBoxLayout()
        ab.addWidget(self.action_sync)
        ab.addWidget(self.action_srs)
        ab.addWidget(self.action_both)
        ab.addStretch()
        action_box.setLayout(ab)

        # Mode selector
        self.mode_single = QRadioButton("Single pair")
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

        self.settings_sync = QGroupBox("Sync settings")
        sg = QHBoxLayout()
        sg.addWidget(QLabel("Model:"))
        sg.addWidget(self.model_combo)
        sg.addSpacing(20)
        sg.addWidget(QLabel("Language:"))
        sg.addWidget(self.lang_edit)
        sg.addSpacing(20)
        sg.addWidget(self.mlx_check)
        sg.addStretch()
        self.settings_sync.setLayout(sg)

        # SRS settings (visible only when action = Build Anki deck)
        self.pad_ms_edit = QLineEdit("200")
        self.pad_ms_edit.setMaximumWidth(60)
        self.fade_ms_edit = QLineEdit("10")
        self.fade_ms_edit.setMaximumWidth(60)
        self.vad_check_box = QCheckBox("VAD edge check")
        self.vad_check_box.setChecked(True)
        self.cover_box = QCheckBox("Embed cover art")
        self.cover_box.setChecked(True)
        self.cover_image_edit = QLineEdit()
        self.cover_image_edit.setPlaceholderText("Optional: path to a cover image (overrides auto-detection)")
        cover_image_btn = QPushButton("Browse…")
        cover_image_btn.clicked.connect(self._pick_cover_image)
        # "auto" means match source bitrate/channels; user can override.
        self.bitrate_edit = QLineEdit("auto")
        self.bitrate_edit.setMaximumWidth(70)
        self.channels_combo = QComboBox()
        self.channels_combo.addItems(["auto", "mono", "stereo"])
        self.srs_output_edit = QLineEdit()
        self.srs_output_edit.setPlaceholderText("Output folder (default: next to audio file)")
        srs_output_btn = QPushButton("Browse…")
        srs_output_btn.clicked.connect(self._pick_srs_output)

        self.settings_srs = QGroupBox("Anki deck settings")
        srs_layout = QVBoxLayout()
        srs_row1 = QHBoxLayout()
        srs_row1.addWidget(QLabel("Pad (ms):"))
        srs_row1.addWidget(self.pad_ms_edit)
        srs_row1.addSpacing(15)
        srs_row1.addWidget(QLabel("Fade (ms):"))
        srs_row1.addWidget(self.fade_ms_edit)
        srs_row1.addSpacing(15)
        srs_row1.addWidget(self.vad_check_box)
        srs_row1.addSpacing(15)
        srs_row1.addWidget(self.cover_box)
        srs_row1.addStretch()
        srs_row2 = QHBoxLayout()
        srs_row2.addWidget(QLabel("Bitrate:"))
        srs_row2.addWidget(self.bitrate_edit)
        srs_row2.addSpacing(15)
        srs_row2.addWidget(QLabel("Channels:"))
        srs_row2.addWidget(self.channels_combo)
        srs_row2.addStretch()
        srs_row3 = QHBoxLayout()
        srs_row3.addWidget(QLabel("Cover image:"))
        srs_row3.addWidget(self.cover_image_edit, 1)
        srs_row3.addWidget(cover_image_btn)
        srs_row4 = QHBoxLayout()
        srs_row4.addWidget(QLabel("Output:"))
        srs_row4.addWidget(self.srs_output_edit, 1)
        srs_row4.addWidget(srs_output_btn)
        srs_layout.addLayout(srs_row1)
        srs_layout.addLayout(srs_row2)
        srs_layout.addLayout(srs_row3)
        srs_layout.addLayout(srs_row4)
        self.settings_srs.setLayout(srs_layout)
        self.settings_srs.setVisible(False)

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
        root.addWidget(action_box)
        root.addWidget(mode_box)
        root.addWidget(self.single_box)
        root.addWidget(self.folder_box)
        root.addWidget(self.settings_sync)
        root.addWidget(self.settings_srs)
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

    def _on_action_changed(self):
        is_srs_only = self.action_srs.isChecked()
        is_both = self.action_both.isChecked()
        # Sync settings visible whenever sync runs (sync-only or both)
        self.settings_sync.setVisible(not is_srs_only)
        # SRS settings visible whenever srs runs (srs-only or both)
        self.settings_srs.setVisible(is_srs_only or is_both)
        # Folder detection rules change with action; refresh preview.
        self._refresh_folder_preview()
        # Hint placeholder text on the text field
        if is_srs_only:
            self.text_edit.setPlaceholderText("Path to SRT/VTT/ASS file (the subtitle source)")
        else:
            self.text_edit.setPlaceholderText("Path to epub/txt file")

    def _pick_srs_output(self):
        p = QFileDialog.getExistingDirectory(self, "Choose output folder for .apkg")
        if p:
            self.srs_output_edit.setText(p)

    def _pick_cover_image(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Choose cover image", "",
            "Images (*.jpg *.jpeg *.png);;All files (*)",
        )
        if p:
            self.cover_image_edit.setText(p)

    def _pick_audio(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select audio file", "",
            "Audio (*.mp3 *.m4a *.m4b *.mp4 *.aac *.flac *.ogg *.wav *.opus *.mkv);;All files (*)"
        )
        if p:
            self.audio_edit.setText(p)

    def _pick_text(self):
        if self.action_srs.isChecked():
            caption = "Select subtitle file"
            filt = "Subtitles (*.srt *.vtt *.ass);;All files (*)"
        else:
            caption = "Select epub or text file"
            filt = "Text (*.epub *.txt *.srt *.vtt *.ass);;All files (*)"
        p, _ = QFileDialog.getOpenFileName(self, caption, "", filt)
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
        # Only the srs-only path expects pre-existing SRTs. Sync and Sync+Build
        # both consume epub/txt as the "text" input.
        exts = SRS_TEXT_EXTS if self.action_srs.isChecked() else TEXT_EXTS
        try:
            pairs = detect_pairs(Path(folder), exts)
        except Exception as e:
            self.folder_preview.setText(f"⚠️  Couldn't scan folder: {e}")
            return
        if not pairs:
            kind = "audio+subtitle" if self.action_srs.isChecked() else "audio+text"
            self.folder_preview.setText(f"⚠️  No {kind} pairs detected.")
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

        if self.action_srs.isChecked():
            action = "srs"
        elif self.action_both.isChecked():
            action = "both"
        else:
            action = "sync"
        opts = {
            "action": action,
            "model": self.model_combo.currentText(),
            "lang": self.lang_edit.text().strip() or "ja",
            "mlx": self.mlx_check.isChecked(),
        }
        if action in ("srs", "both"):
            # Validate pad/fade are integers and non-negative; fall back to defaults
            # if the user typed something weird.
            try:
                opts["srs_pad_ms"] = max(0, int(self.pad_ms_edit.text().strip() or "200"))
            except ValueError:
                opts["srs_pad_ms"] = 200
            try:
                opts["srs_fade_ms"] = max(0, int(self.fade_ms_edit.text().strip() or "10"))
            except ValueError:
                opts["srs_fade_ms"] = 10
            opts["srs_vad_check"] = self.vad_check_box.isChecked()
            opts["srs_cover"] = self.cover_box.isChecked()
            opts["srs_cover_image"] = self.cover_image_edit.text().strip()
            opts["srs_bitrate"] = self.bitrate_edit.text().strip() or "auto"
            opts["srs_channels"] = self.channels_combo.currentText()
            out = self.srs_output_edit.text().strip()
            if out:
                opts["srs_output_dir"] = out

        self.log_view.clear()
        self.batch_label.setText(f"Batch: 0 / {len(jobs)} books")
        self.book_bar.setValue(0)
        self.batch_bar.setValue(0)
        self.status_label.setText(f"Starting — {len(jobs)} book(s) queued…")

        # Record output folders for the "Open folder" button. SRS-explicit dir
        # wins (for srs and both); else sync's -d workdir; else audio's parent.
        srs_out = opts.get("srs_output_dir") if action in ("srs", "both") else None
        if srs_out:
            self._last_output_dirs = [Path(srs_out)] * len(jobs)
        else:
            self._last_output_dirs = [
                job.workdir if (job.workdir is not None and action != "srs") else job.audio.parent
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
            # Sync and Sync+Build both pair on epub/txt; srs-only pairs on SRT.
            exts = SRS_TEXT_EXTS if self.action_srs.isChecked() else TEXT_EXTS
            return detect_pairs(Path(folder), exts)

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
        # Reveal an .apkg if we built one; otherwise an .srt; else the dir.
        built_apkg = self.action_srs.isChecked() or self.action_both.isChecked()
        revealed_glob = "*.apkg" if built_apkg else "*.srt"
        opened: set[str] = set()
        for d in self._last_output_dirs:
            if not d.exists():
                continue
            hits = sorted(d.glob(revealed_glob), key=lambda p: p.stat().st_mtime, reverse=True)
            target = hits[0] if hits else d
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


# ============================================================ TTS tab and helpers

# Mirror of subplz.tts.SBV2_PRESETS for the install dialog. Kept duplicated
# (instead of importing) so the GUI launches without pulling torch/numpy at
# startup time — the actual install runs as a subprocess to `subplz voice`.
SBV2_PRESET_INFO = [
    # (key, tag, blurb shown in the install dialog)
    ("rikka_botan_cool",  "narrator",  "Female, soft/unhurried (おっとり); markets for 朗読 — CC-BY-SA"),
    ("koharune-ami",      "narrator",  "Female (young adult), corpus, 6 styles — amitaro.net terms"),
    ("amitaro",           "narrator",  "Same VA as koharune-ami, livestream-trained — amitaro.net terms"),
    ("lux",               "character", "Original female character (v2.6.1) — CC-BY-4.0"),
    ("rikka_botan_sweet", "character", "Female, sweet/cute register — license unknown"),
    ("fumifumi",          "character", "Female single voice (v2.2-JP-Extra) — license unknown"),
    ("jvnv-f1-jp",        "emotional", "Female, 7 emotion styles (anger/sad/happy/etc.) — CC-BY-SA"),
    ("jvnv-f2-jp",        "emotional", "Second female, 7 emotion styles — CC-BY-SA"),
    ("jvnv-m1-jp",        "emotional", "Adult male, 7 emotion styles — CC-BY-SA"),
    ("jvnv-m2-jp",        "emotional", "Second adult male, 7 emotion styles — CC-BY-SA"),
    ("mofa-girls",        "emotional", "Pack: 4 young females + 1 male, 26 styles each — MIT"),
]


def _list_voices(backend: str) -> list[str]:
    """Read voice names from disk directly (cheap, no subprocess)."""
    root = Path(os.environ.get("SUBPLZ_VOICES_DIR") or (Path.home() / ".subplz" / "voices"))
    bdir = root / backend
    if not bdir.exists():
        return []
    return sorted(p.name for p in bdir.iterdir() if p.is_dir())


def _run_voice_cmd(args: list[str], log_emit) -> bool:
    """Run `subplz voice …`, stream stdout/stderr to log_emit. Returns success."""
    cmd = [str(SUBPLZ_BIN), "voice", *args]
    log_emit(f"$ {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        log_emit(line.rstrip())
    return proc.wait() == 0


class InstallPresetDialog(QDialog):
    """Multi-select dialog to install SBV2 narrator presets."""

    def __init__(self, parent=None, already_installed: Optional[set[str]] = None):
        super().__init__(parent)
        self.setWindowTitle("Install SBV2 voice presets")
        self.resize(560, 420)
        already = already_installed or set()

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Select presets to download (~250 MB each):"))
        self.listw = QListWidget()
        for key, tag, blurb in SBV2_PRESET_INFO:
            item = QListWidgetItem(f"[{tag:9s}] {key} — {blurb}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            if key in already:
                item.setCheckState(Qt.Checked)
                item.setText(item.text() + "   (installed)")
                item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable & ~Qt.ItemIsEnabled)
            else:
                item.setCheckState(Qt.Unchecked)
            item.setData(Qt.UserRole, key)
            self.listw.addItem(item)
        layout.addWidget(self.listw, 1)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)
        self.setLayout(layout)

    def selected(self) -> list[str]:
        out = []
        for i in range(self.listw.count()):
            item = self.listw.item(i)
            if (item.flags() & Qt.ItemIsUserCheckable) and item.checkState() == Qt.Checked:
                out.append(item.data(Qt.UserRole))
        return out


class CloneVoiceDialog(QDialog):
    """Pick an audio file, set start/duration, register as Irodori voice."""

    DEFAULT_DURATION = 20  # Irodori caps reference at 30s; 20 leaves headroom.

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Clone Irodori voice from audio")
        self.resize(540, 200)

        self.audio_edit = QLineEdit()
        self.audio_edit.setPlaceholderText("Pick an audiobook / podcast / clean voice recording")
        self.audio_edit.textChanged.connect(self._sync_default_name)
        audio_btn = QPushButton("Browse…")
        audio_btn.clicked.connect(self._pick_audio)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("voice name (defaults from filename)")

        self.start_spin = QSpinBox()
        self.start_spin.setRange(0, 60 * 60)
        self.start_spin.setValue(60)
        self.start_spin.setSuffix(" s")
        self.dur_spin = QSpinBox()
        self.dur_spin.setRange(5, 30)
        self.dur_spin.setValue(self.DEFAULT_DURATION)
        self.dur_spin.setSuffix(" s")

        form = QFormLayout()
        row = QHBoxLayout()
        row.addWidget(self.audio_edit, 1)
        row.addWidget(audio_btn)
        form.addRow("Audio source:", self._wrap(row))
        form.addRow("Voice name:", self.name_edit)
        form.addRow("Skip first:", self.start_spin)
        form.addRow("Reference length:", self.dur_spin)
        form.addRow(QLabel("Irodori caps reference at 30 s. 20 s is a safe default."))

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._validate_and_accept)
        btns.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(btns)
        self.setLayout(layout)

    @staticmethod
    def _wrap(layout) -> QWidget:
        w = QWidget()
        w.setLayout(layout)
        return w

    def _pick_audio(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Pick reference audio", "",
            "Audio (*.mp3 *.m4a *.m4b *.mp4 *.aac *.flac *.ogg *.wav *.opus);;All files (*)",
        )
        if p:
            self.audio_edit.setText(p)

    def _sync_default_name(self, txt: str):
        if not self.name_edit.text().strip() and txt:
            stem = Path(txt).stem
            # Strip extension-like decorations: ".part1", " -cbr", etc.
            stem = re.sub(r"[\s_-]+cbr$", "", stem, flags=re.I)
            self.name_edit.setText(stem)

    def _validate_and_accept(self):
        if not self.audio_edit.text().strip():
            QMessageBox.warning(self, "Missing audio", "Pick an audio file first.")
            return
        if not self.name_edit.text().strip():
            QMessageBox.warning(self, "Missing name", "Give the voice a name.")
            return
        self.accept()

    def values(self) -> dict:
        return {
            "audio": self.audio_edit.text().strip(),
            "name":  self.name_edit.text().strip(),
            "start": int(self.start_spin.value()),
            "duration": int(self.dur_spin.value()),
        }


class TTSWorker(QObject):
    """Runs `subplz tts …` in a thread, streams progress."""

    log = Signal(str)
    status = Signal(str)
    sentence_progress = Signal(int, int)  # done, total
    done = Signal(bool, str)  # success, message

    def __init__(
        self,
        epub: Path,
        backend: str,
        voice: str,
        out_dir: Path,
        max_chars: int,
        *,
        sentences_file: Optional[Path] = None,
        output_stem: Optional[str] = None,
        num_steps: Optional[int] = None,
        cfg_scale_speaker: Optional[float] = None,
        caption: Optional[str] = None,
        max_retries: int = 3,
    ):
        super().__init__()
        self.epub = epub
        self.backend = backend
        self.voice = voice
        self.out_dir = out_dir
        self.max_chars = max_chars
        self.sentences_file = sentences_file
        self.output_stem = output_stem
        self.num_steps = num_steps
        self.cfg_scale_speaker = cfg_scale_speaker
        self.caption = caption
        self.max_retries = max_retries
        self._cancel = False
        self._proc: Optional[subprocess.Popen] = None

    def stop(self):
        self._cancel = True
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()

    def run(self):
        cmd = [
            str(SUBPLZ_BIN), "tts",
            "--backend", self.backend,
            "--voice", self.voice,
            "--output-dir", str(self.out_dir),
        ]
        # Prefer cached sentences (skips re-parsing the epub in the subprocess).
        if self.sentences_file is not None:
            cmd += ["--sentences-file", str(self.sentences_file)]
            if self.output_stem:
                cmd += ["--output-stem", self.output_stem]
        else:
            cmd += ["--epub", str(self.epub)]
        if self.max_chars > 0:
            cmd += ["--max-chars", str(self.max_chars)]
        # Irodori-only knobs; the CLI silently ignores them for SBV2.
        if self.backend == "irodori":
            if self.num_steps is not None:
                cmd += ["--num-steps", str(self.num_steps)]
            if self.cfg_scale_speaker is not None:
                cmd += ["--cfg-scale-speaker", str(self.cfg_scale_speaker)]
            if self.caption:
                cmd += ["--caption", self.caption]

        # Auto-retry loop. Each retry re-spawns the same subprocess; the
        # pipeline's per-sentence resume logic in subplz/tts.py picks up
        # where the previous attempt left off, so retries are cheap.
        tqdm_re = re.compile(r"\b(\d+)/(\d+)\b")
        last_rc: Optional[int] = None
        for attempt in range(1, self.max_retries + 1):
            if self._cancel:
                break
            if attempt == 1:
                self.status.emit(f"Running {self.backend} on {self.voice}…")
            else:
                self.status.emit(f"Auto-retry {attempt}/{self.max_retries}…")
                self.log.emit(f"⟳ Auto-retry {attempt}/{self.max_retries} (previous exit {last_rc})")
            self.log.emit(f"$ {' '.join(cmd)}")
            try:
                self._proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                for line in self._proc.stdout:
                    if self._cancel:
                        break
                    self.log.emit(line.rstrip())
                    m = tqdm_re.search(line)
                    if m:
                        done = int(m.group(1)); total = int(m.group(2))
                        if total > 0 and done <= total:
                            self.sentence_progress.emit(done, total)
                last_rc = self._proc.wait()
            except Exception as e:
                self.done.emit(False, f"{type(e).__name__}: {e}")
                return
            if self._cancel:
                self.done.emit(False, "Cancelled.")
                return
            if last_rc == 0:
                self.done.emit(True, f"Wrote audiobook to {self.out_dir}")
                return
            # Non-zero exit: retry unless we've exhausted attempts.
            if attempt < self.max_retries:
                time.sleep(2)  # brief backoff before re-spawning
        self.done.emit(False, f"subplz tts failed after {self.max_retries} attempts (last exit {last_rc})")


class _VoiceActionWorker(QObject):
    """One-shot worker for `subplz voice install-preset` / `voice clone`."""

    log = Signal(str)
    done = Signal(bool)

    def __init__(self, args_batches: list[list[str]]):
        super().__init__()
        self.args_batches = args_batches

    def run(self):
        ok_all = True
        for args in self.args_batches:
            ok = _run_voice_cmd(args, self.log.emit)
            if not ok:
                ok_all = False
        self.done.emit(ok_all)


class TTSTab(QWidget):
    """Generate an audiobook (WAV + SRT) from an epub using SBV2 or Irodori."""

    # QSettings keys for persisted Irodori knobs
    K_NUM_STEPS = "tts/irodori/num_steps"
    K_CFG_SPK   = "tts/irodori/cfg_scale_speaker"
    K_CAPTION   = "tts/irodori/caption"
    K_MAX_CHARS = "tts/max_chars"
    K_BACKEND   = "tts/backend"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.thread: Optional[QThread] = None
        self.worker: Optional[object] = None
        self._last_out_dir: Optional[Path] = None
        # Settings: cross-platform store. On Mac this lives in ~/Library/Preferences/.
        self.settings = QSettings("subplz", "gui")
        # Sentence cache keyed by (epub_path, mtime). When user re-uses the
        # same epub across multiple voice runs, we skip the parse entirely.
        self._sentence_cache: dict[tuple[str, float], list[str]] = {}
        self._sentence_temp: Optional[Path] = None
        # Run stats — set in _on_run, read in _on_done
        self._run_start: Optional[float] = None
        self._run_audio_s: Optional[float] = None

        # ----- backend chooser -----
        self.bk_sbv2 = QRadioButton("SBV2  (fast, ~7× realtime)")
        self.bk_irod = QRadioButton("Irodori  (slower, voice-clone from any audio)")
        self.bk_sbv2.setChecked(True)
        bk_grp = QButtonGroup(self)
        bk_grp.addButton(self.bk_sbv2)
        bk_grp.addButton(self.bk_irod)
        self.bk_sbv2.toggled.connect(lambda c: c and self._on_backend_changed())
        self.bk_irod.toggled.connect(lambda c: c and self._on_backend_changed())
        bk_row = QHBoxLayout()
        bk_row.addWidget(self.bk_sbv2)
        bk_row.addWidget(self.bk_irod)
        bk_row.addStretch()
        bk_box = QGroupBox("Backend")
        bk_box.setLayout(bk_row)

        # ----- voice chooser -----
        self.voice_combo = QComboBox()
        self.voice_combo.setMinimumWidth(280)
        self.manage_btn = QPushButton("Install preset…")
        self.manage_btn.clicked.connect(self._on_manage_clicked)
        refresh_btn = QPushButton("⟳")
        refresh_btn.setToolTip("Reload installed voices")
        refresh_btn.setMaximumWidth(32)
        refresh_btn.clicked.connect(self._reload_voices)
        v_row = QHBoxLayout()
        v_row.addWidget(QLabel("Voice:"))
        v_row.addWidget(self.voice_combo, 1)
        v_row.addWidget(refresh_btn)
        v_row.addWidget(self.manage_btn)
        v_box = QGroupBox("Voice")
        v_box.setLayout(v_row)

        # ----- epub picker -----
        self.epub_edit = QLineEdit()
        self.epub_edit.setPlaceholderText("Drop an .epub here, or browse")
        epub_btn = QPushButton("Browse…")
        epub_btn.clicked.connect(self._pick_epub)
        ep_row = QHBoxLayout()
        ep_row.addWidget(QLabel("Epub:"))
        ep_row.addWidget(self.epub_edit, 1)
        ep_row.addWidget(epub_btn)

        # Irodori-only: drop an audio file to quick-clone a voice for this run.
        # Wraps the whole row in a QWidget so it can be hidden as a unit when
        # SBV2 is the active backend.
        self.ref_audio_edit = QLineEdit()
        self.ref_audio_edit.setPlaceholderText(
            "Optional: drop an audio file here to quick-clone a new Irodori voice (overrides Voice above)"
        )
        ref_btn = QPushButton("Browse…")
        ref_btn.clicked.connect(self._pick_ref_audio)
        ref_clear_btn = QPushButton("×")
        ref_clear_btn.setToolTip("Clear — fall back to selected Voice")
        ref_clear_btn.setMaximumWidth(28)
        ref_clear_btn.clicked.connect(lambda: self.ref_audio_edit.clear())
        ref_inner = QHBoxLayout()
        ref_inner.setContentsMargins(0, 0, 0, 0)
        ref_inner.addWidget(QLabel("Ref audio:"))
        ref_inner.addWidget(self.ref_audio_edit, 1)
        ref_inner.addWidget(ref_btn)
        ref_inner.addWidget(ref_clear_btn)
        self.ref_audio_row = QWidget()
        self.ref_audio_row.setLayout(ref_inner)

        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Output folder (default: same folder as the epub)")
        out_btn = QPushButton("Browse…")
        out_btn.clicked.connect(self._pick_out)
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output:"))
        out_row.addWidget(self.out_edit, 1)
        out_row.addWidget(out_btn)

        ep_box = QGroupBox("Source")
        epv = QVBoxLayout()
        epv.addLayout(ep_row)
        epv.addWidget(self.ref_audio_row)
        epv.addLayout(out_row)
        ep_box.setLayout(epv)

        # ----- options -----
        self.chars_spin = QSpinBox()
        self.chars_spin.setRange(0, 5_000_000)
        self.chars_spin.setSingleStep(500)
        self.chars_spin.setValue(1500)
        self.chars_spin.setSuffix(" chars")
        self.chars_spin.setSpecialValueText("entire book")
        opt_row = QHBoxLayout()
        opt_row.addWidget(QLabel("How much to synthesize:"))
        opt_row.addWidget(self.chars_spin)
        opt_row.addWidget(QLabel("(JA chars; whole-sentence boundary, 0 = entire book)"))
        opt_row.addStretch()
        opt_box = QGroupBox("Options")
        opt_box.setLayout(opt_row)

        # ----- Irodori advanced (only visible when Irodori is selected) -----
        # Wider spinboxes so the up/down arrows are real click targets;
        # QFormLayout was crushing both the buttons and the wrapped help text.
        self.irod_steps_spin = QSpinBox()
        self.irod_steps_spin.setRange(8, 80)
        self.irod_steps_spin.setSingleStep(2)
        self.irod_steps_spin.setValue(40)
        self.irod_steps_spin.setMinimumWidth(110)

        self.irod_cfg_spk_spin = QDoubleSpinBox()
        self.irod_cfg_spk_spin.setRange(1.0, 10.0)
        self.irod_cfg_spk_spin.setSingleStep(0.5)
        self.irod_cfg_spk_spin.setDecimals(1)
        self.irod_cfg_spk_spin.setValue(5.0)
        self.irod_cfg_spk_spin.setMinimumWidth(110)

        self.irod_caption_edit = QLineEdit()
        self.irod_caption_edit.setPlaceholderText("e.g. 落ち着いた朗読、内省的な語り口")

        def _help(text: str, lines: int = 2) -> QLabel:
            lbl = QLabel(text)
            lbl.setStyleSheet("color: #888; font-size: 11px;")
            lbl.setWordWrap(True)
            # Qt word-wrapped labels don't compute heightForWidth ahead of
            # layout, so the row collapses to one line of text height. Reserve
            # explicit height for the expected number of wrapped lines.
            lbl.setMinimumHeight(16 * lines + 8)
            lbl.setContentsMargins(0, 0, 0, 6)
            return lbl

        self.irod_steps_help = _help(
            "More steps = richer prosody, less granularity in the voice. "
            "Slower. 24=fast, 40=default, 60+=highest quality.",
            lines=2,
        )
        self.irod_cfg_spk_help = _help(
            "How strictly the voice locks to the cloned reference. Default 5.0. "
            "Lower (3.5–4.0) gives the model more emotional freedom; too low and "
            "the voice starts to drift off the reference.",
            lines=3,
        )
        self.irod_caption_help = _help(
            "Optional style hint (English or Japanese). Steers register/emotion. "
            "Only honored on caption-enabled Irodori checkpoints — leave blank "
            "if unsure (cleanest default behavior).",
            lines=3,
        )

        def _row(label_text: str, widget: QWidget) -> QHBoxLayout:
            h = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setMinimumWidth(220)
            h.addWidget(lbl)
            h.addWidget(widget)
            h.addStretch()
            return h

        irod_layout = QVBoxLayout()
        irod_layout.addLayout(_row("Sampling steps:", self.irod_steps_spin))
        irod_layout.addWidget(self.irod_steps_help)
        irod_layout.addLayout(_row("Speaker lock-in (cfg_scale_speaker):", self.irod_cfg_spk_spin))
        irod_layout.addWidget(self.irod_cfg_spk_help)
        irod_layout.addLayout(_row("Style caption:", self.irod_caption_edit))
        irod_layout.addWidget(self.irod_caption_help)
        self.irod_box = QGroupBox("Irodori — advanced")
        self.irod_box.setLayout(irod_layout)

        # ----- progress + buttons -----
        self.status_label = QLabel("Pick a voice and an epub.")
        self.status_label.setFont(QFont("", weight=QFont.DemiBold))
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("%v / %m sentences")
        # Stats line, populated on completion
        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet("color: #888; font-size: 11px;")

        self.run_btn = QPushButton("▶  Generate audiobook")
        self.run_btn.clicked.connect(self._on_run)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.open_btn = QPushButton("📂  Open output")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._on_open_output)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(self.open_btn)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.cancel_btn)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Menlo, monospace", 11))
        self.log_view.setMinimumHeight(160)

        root = QVBoxLayout()
        root.addWidget(bk_box)
        root.addWidget(v_box)
        root.addWidget(ep_box)
        root.addWidget(opt_box)
        root.addWidget(self.irod_box)
        root.addWidget(self.status_label)
        root.addWidget(self.progress)
        root.addWidget(self.stats_label)
        root.addLayout(btn_row)
        root.addWidget(self.log_view, 1)
        self.setLayout(root)

        # Restore persisted settings BEFORE wiring change handlers so we
        # don't trigger spurious saves during restoration.
        self._restore_settings()
        # Persist on every change. setattr loops keep this terse.
        self.chars_spin.valueChanged.connect(
            lambda v: self.settings.setValue(self.K_MAX_CHARS, int(v)))
        self.irod_steps_spin.valueChanged.connect(
            lambda v: self.settings.setValue(self.K_NUM_STEPS, int(v)))
        self.irod_cfg_spk_spin.valueChanged.connect(
            lambda v: self.settings.setValue(self.K_CFG_SPK, float(v)))
        self.irod_caption_edit.textChanged.connect(
            lambda t: self.settings.setValue(self.K_CAPTION, str(t)))
        self.bk_sbv2.toggled.connect(
            lambda c: c and self.settings.setValue(self.K_BACKEND, "sbv2"))
        self.bk_irod.toggled.connect(
            lambda c: c and self.settings.setValue(self.K_BACKEND, "irodori"))

        self._on_backend_changed()  # populate voices for the default backend
        # After the window has rendered, check for an unfinished run and
        # offer to resume. QTimer.singleShot(0) defers until the event loop
        # is idle so the dialog appears on top of a fully-drawn UI.
        QTimer.singleShot(0, self._maybe_offer_session_resume)

    # ----- session resume registry -----

    def _save_active_job(self, params: dict) -> None:
        try:
            ACTIVE_JOB_PATH.parent.mkdir(parents=True, exist_ok=True)
            ACTIVE_JOB_PATH.write_text(json.dumps(params, indent=2, ensure_ascii=False))
        except OSError:
            pass  # best-effort

    def _clear_active_job(self) -> None:
        try:
            ACTIVE_JOB_PATH.unlink(missing_ok=True)
        except OSError:
            pass

    def _maybe_offer_session_resume(self) -> None:
        """If a previous run was interrupted (registry file exists), prompt
        the user to resume. The actual resume just re-runs Generate with
        the saved params — pipeline's per-sentence cache picks up where it
        left off."""
        if not ACTIVE_JOB_PATH.exists():
            return
        try:
            p = json.loads(ACTIVE_JOB_PATH.read_text())
        except Exception:
            self._clear_active_job()
            return
        # Sanity-check the work_dir actually exists; otherwise the entry is
        # stale (success cleanup ran but registry clear failed, or user
        # deleted the output dir manually).
        out_dir = Path(p.get("out_dir", ""))
        epub_path = Path(p.get("epub", ""))
        stem = f"{epub_path.stem}.{p.get('backend','')}.{p.get('voice','')}"
        work_dir = out_dir / f".{stem}.subplz-work"
        if not work_dir.exists():
            self._clear_active_job()
            return
        n_done = sum(1 for f in work_dir.glob("[0-9]" * 6 + ".wav"))
        n_total = 0
        meta = work_dir / "meta.json"
        if meta.exists():
            try:
                n_total = int(json.loads(meta.read_text()).get("n_sentences", 0))
            except Exception:
                pass
        progress = f"{n_done}/{n_total}" if n_total else f"{n_done}"
        msg = (f"Found an unfinished generation:\n\n"
               f"  Book: {epub_path.name}\n"
               f"  Backend: {p.get('backend')}\n"
               f"  Voice: {p.get('voice')}\n"
               f"  Progress: {progress} sentences\n\n"
               f"Resume now? (No keeps it for later, Discard deletes the partial output.)")
        ans = QMessageBox.question(
            self, "Resume previous run?", msg,
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Discard,
            QMessageBox.Yes,
        )
        if ans == QMessageBox.Yes:
            self._switch_to_tts_tab()
            self._populate_from_job(p)
            QTimer.singleShot(150, self._on_run)
        elif ans == QMessageBox.Discard:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)
            self._clear_active_job()

    def _switch_to_tts_tab(self) -> None:
        """If we're sitting inside a MainWindow with tabs, raise the TTS tab.
        Lookup is by ancestry rather than a hardcoded parent ref so the tab
        still works if reparented elsewhere later."""
        parent = self.parent()
        while parent is not None and not isinstance(parent, QTabWidget):
            parent = parent.parent()
        if isinstance(parent, QTabWidget):
            parent.setCurrentWidget(self)

    def _populate_from_job(self, p: dict) -> None:
        if p.get("backend") == "irodori":
            self.bk_irod.setChecked(True)
        else:
            self.bk_sbv2.setChecked(True)
        self._reload_voices()
        self.epub_edit.setText(p.get("epub", ""))
        self.out_edit.setText(p.get("out_dir", ""))
        try:
            self.chars_spin.setValue(int(p.get("max_chars", 0) or 0))
        except (TypeError, ValueError):
            pass
        if "num_steps" in p:
            try: self.irod_steps_spin.setValue(int(p["num_steps"]))
            except (TypeError, ValueError): pass
        if "cfg_scale_speaker" in p:
            try: self.irod_cfg_spk_spin.setValue(float(p["cfg_scale_speaker"]))
            except (TypeError, ValueError): pass
        if "caption" in p:
            self.irod_caption_edit.setText(str(p["caption"] or ""))
        idx = self.voice_combo.findText(p.get("voice", ""))
        if idx >= 0:
            self.voice_combo.setCurrentIndex(idx)

    def _restore_settings(self):
        s = self.settings
        # Backend: setChecked() before voice reload so the right list loads
        if s.value(self.K_BACKEND, "sbv2") == "irodori":
            self.bk_irod.setChecked(True)
        else:
            self.bk_sbv2.setChecked(True)
        try:
            self.chars_spin.setValue(int(s.value(self.K_MAX_CHARS, 1500)))
            self.irod_steps_spin.setValue(int(s.value(self.K_NUM_STEPS, 40)))
            self.irod_cfg_spk_spin.setValue(float(s.value(self.K_CFG_SPK, 5.0)))
            self.irod_caption_edit.setText(str(s.value(self.K_CAPTION, "") or ""))
        except (TypeError, ValueError):
            # If the stored value is garbage, fall back silently to defaults.
            pass

    # ---- drag and drop ----
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls() and any(u.isLocalFile() for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        paths = [Path(u.toLocalFile()) for u in event.mimeData().urls() if u.isLocalFile()]
        for p in paths:
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            if ext == ".epub":
                self.epub_edit.setText(str(p))
                event.acceptProposedAction()
                return
            if ext in AUDIO_EXTS and self._current_backend() == "irodori":
                self.ref_audio_edit.setText(str(p))
                self._quick_clone_from_ref(p)
                event.acceptProposedAction()
                return

    # ---- helpers ----
    def _current_backend(self) -> str:
        return "sbv2" if self.bk_sbv2.isChecked() else "irodori"

    def _reload_voices(self):
        backend = self._current_backend()
        voices = _list_voices(backend)
        self.voice_combo.clear()
        if voices:
            self.voice_combo.addItems(voices)
            self.voice_combo.setEnabled(True)
        else:
            placeholder = ("No SBV2 voices installed. Click 'Install preset…' to download some."
                           if backend == "sbv2"
                           else "No Irodori voices yet. Click 'Clone from audio…' to add one.")
            self.voice_combo.addItem(placeholder)
            self.voice_combo.setEnabled(False)

    def _on_backend_changed(self):
        backend = self._current_backend()
        if backend == "sbv2":
            self.manage_btn.setText("Install preset…")
        else:
            self.manage_btn.setText("Clone from audio…")
        # Irodori-only widgets hide when SBV2 is active
        if hasattr(self, "irod_box"):
            self.irod_box.setVisible(backend == "irodori")
        if hasattr(self, "ref_audio_row"):
            self.ref_audio_row.setVisible(backend == "irodori")
        self._reload_voices()

    def _on_manage_clicked(self):
        if self._current_backend() == "sbv2":
            installed = set(_list_voices("sbv2"))
            dlg = InstallPresetDialog(self, already_installed=installed)
            if dlg.exec() != QDialog.Accepted:
                return
            picked = dlg.selected()
            if not picked:
                return
            args_batches = [["install-preset", "--name", p] for p in picked]
        else:
            dlg = CloneVoiceDialog(self)
            if dlg.exec() != QDialog.Accepted:
                return
            v = dlg.values()
            args_batches = [[
                "clone", "--audio", v["audio"], "--name", v["name"],
                "--start", str(v["start"]), "--duration", str(v["duration"]),
                "--overwrite",
            ]]

        self._append_log(f"Voice op started ({len(args_batches)} job(s))…")
        self.run_btn.setEnabled(False)
        self.manage_btn.setEnabled(False)
        self.thread = QThread()
        self.worker = _VoiceActionWorker(args_batches)
        self.worker.moveToThread(self.thread)
        self.worker.log.connect(self._append_log)
        self.worker.done.connect(self._on_voice_op_done)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def _on_voice_op_done(self, ok: bool):
        self._teardown_thread()
        self.run_btn.setEnabled(True)
        self.manage_btn.setEnabled(True)
        self._reload_voices()
        if ok:
            self._append_log("✅  Voice op complete.")
        else:
            self._append_log("❌  Voice op failed (see log above).")

    def _pick_epub(self):
        p, _ = QFileDialog.getOpenFileName(self, "Pick epub", "", "Epub (*.epub);;All files (*)")
        if p:
            self.epub_edit.setText(p)

    def _pick_ref_audio(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Pick reference audio", "",
            "Audio (*.mp3 *.m4a *.m4b *.mp4 *.aac *.flac *.ogg *.wav *.opus);;All files (*)",
        )
        if p:
            self.ref_audio_edit.setText(p)
            self._quick_clone_from_ref(Path(p))

    def _quick_clone_from_ref(self, audio_path: Path):
        """Synchronously clone (overwrites) and auto-select in the voice combo.
        ffmpeg trim is ~1s so blocking the UI thread is fine; longer than that
        and we'd push this to _VoiceActionWorker."""
        voice = self._voice_name_from_audio(audio_path)
        self.status_label.setText(f"Cloning '{voice}'…")
        self._append_log(f"Quick-cloning '{voice}' from {audio_path.name}…")
        ok = _run_voice_cmd(
            ["clone", "--audio", str(audio_path), "--name", voice,
             "--start", "60", "--duration", "20", "--overwrite"],
            self._append_log,
        )
        if not ok:
            self.status_label.setText("❌  Clone failed (see log)")
            return
        self._reload_voices()
        idx = self.voice_combo.findText(voice)
        if idx >= 0:
            self.voice_combo.setCurrentIndex(idx)
        self.status_label.setText(f"✅  Voice '{voice}' ready — click Generate")

    def _pick_out(self):
        p = QFileDialog.getExistingDirectory(self, "Pick output folder")
        if p:
            self.out_edit.setText(p)

    @staticmethod
    def _voice_name_from_audio(audio_path: Path) -> str:
        """Derive a sane voice-library name from an audio filename.
        Strips common decorations like ' -cbr', surrounding brackets, and
        whitespace. Falls back to the bare stem if the cleanup empties it."""
        stem = audio_path.stem
        # Strip "(... )", "[...]", and " -cbr" / "_cbr" tails
        stem = re.sub(r"\s*[\[\(][^\]\)]*[\]\)]", "", stem)
        stem = re.sub(r"[\s_-]+cbr$", "", stem, flags=re.I)
        stem = stem.strip()
        return stem or audio_path.stem

    def _on_run(self):
        if not SUBPLZ_BIN.exists():
            QMessageBox.critical(
                self, "subplz not found",
                f"Expected the subplz CLI at:\n  {SUBPLZ_BIN}\n\n"
                f"Run `.venv/bin/pip install -e .` from the project root first.",
            )
            return
        epub = self.epub_edit.text().strip()
        if not epub or not Path(epub).exists():
            QMessageBox.warning(self, "Missing epub", "Pick an .epub source.")
            return

        backend = self._current_backend()
        # Fallback path: if Irodori, the field has a path, but the derived
        # voice isn't in the library yet (paste/edit case — drop and Browse
        # already cloned on-the-spot). Clone now before generating.
        if backend == "irodori":
            ref_audio = self.ref_audio_edit.text().strip()
            if ref_audio:
                ref_path = Path(ref_audio)
                if not ref_path.exists():
                    QMessageBox.warning(self, "Missing audio", f"Reference audio not found:\n{ref_path}")
                    return
                expected = self._voice_name_from_audio(ref_path)
                if self.voice_combo.findText(expected) < 0:
                    self._quick_clone_from_ref(ref_path)
        if not self.voice_combo.isEnabled() or self.voice_combo.count() == 0:
            QMessageBox.warning(self, "No voice", "Install or clone a voice first.")
            return
        voice = self.voice_combo.currentText()

        out = self.out_edit.text().strip()
        out_dir = Path(out) if out else Path(epub).parent
        max_chars = int(self.chars_spin.value())  # 0 = entire book

        # Sentence cache: parse the epub once per (path, mtime), reuse across
        # runs. Lets the user A/B different voices on the same book without
        # re-parsing each time.
        epub_path = Path(epub)
        try:
            cache_key = (str(epub_path.resolve()), epub_path.stat().st_mtime)
        except OSError:
            cache_key = (str(epub_path), 0.0)
        sentences = self._sentence_cache.get(cache_key)
        if sentences is None:
            self.status_label.setText("Parsing epub…")
            QApplication.processEvents()
            from subplz.tts import extract_sentences
            try:
                sentences = extract_sentences(epub_path, lang="ja")
            except Exception as e:
                QMessageBox.critical(self, "Parse failed", f"{type(e).__name__}: {e}")
                self.status_label.setText("❌ Parse failed.")
                return
            self._sentence_cache[cache_key] = sentences
            self._append_log(f"Parsed {len(sentences)} sentences from {epub_path.name} (cached)")
        else:
            self._append_log(f"Reusing {len(sentences)} cached sentences from {epub_path.name}")
        # Write sentences to a temp file the subprocess will read.
        if self._sentence_temp is None:
            self._sentence_temp = Path(tempfile.mkstemp(prefix="subplz-sents-", suffix=".txt")[1])
        self._sentence_temp.write_text("\n".join(sentences), encoding="utf-8")

        self.log_view.clear()
        self.progress.setValue(0)
        self.progress.setRange(0, 0)  # busy bar until we know sentence count
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.open_btn.setEnabled(False)
        self.stats_label.setText("")
        self._last_out_dir = out_dir
        self._run_start = time.monotonic()
        self._run_audio_s = None

        # Record this run in the registry so we can auto-offer resume next
        # launch if the run crashes / the user closes the app mid-run.
        self._save_active_job({
            "epub": str(epub_path),
            "backend": backend,
            "voice": voice,
            "out_dir": str(out_dir),
            "max_chars": max_chars,
            "num_steps": int(self.irod_steps_spin.value()),
            "cfg_scale_speaker": float(self.irod_cfg_spk_spin.value()),
            "caption": self.irod_caption_edit.text().strip() or "",
        })

        # Irodori-only knobs are passed regardless of backend; the worker
        # only emits them on the CLI when backend == "irodori".
        self.thread = QThread()
        self.worker = TTSWorker(
            epub_path, backend, voice, out_dir, max_chars,
            sentences_file=self._sentence_temp,
            output_stem=epub_path.stem,
            num_steps=int(self.irod_steps_spin.value()),
            cfg_scale_speaker=float(self.irod_cfg_spk_spin.value()),
            caption=self.irod_caption_edit.text().strip() or None,
        )
        self.worker.moveToThread(self.thread)
        self.worker.log.connect(self._append_log)
        self.worker.status.connect(self.status_label.setText)
        self.worker.sentence_progress.connect(self._on_sentence_progress)
        self.worker.done.connect(self._on_done)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

    # Matches our CLI's final log line:  "Wrote ... (X.X min audio, RTF Y.YYx)"
    _AUDIO_LEN_RE = re.compile(r"\((\d+(?:\.\d+)?)\s*min\s+audio", re.I)

    def _on_sentence_progress(self, done: int, total: int):
        if self.progress.maximum() != total:
            self.progress.setRange(0, total)
        self.progress.setValue(done)

    def _on_cancel(self):
        if self.worker:
            self.worker.stop()
            self.status_label.setText("Cancelling…")

    def _on_done(self, ok: bool, msg: str):
        self._teardown_thread()
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if ok and self._last_out_dir and self._last_out_dir.exists():
            self.open_btn.setEnabled(True)
        # Clear the resume registry on success. On failure we keep it so the
        # next launch can offer to pick up where we left off (or so the user
        # can just hit Generate again — the pipeline already supports both).
        if ok:
            self._clear_active_job()
        marker = "✅" if ok else "❌"
        self.status_label.setText(f"{marker} {msg}")
        self._append_log(f"{marker} {msg}")
        if ok and self._run_start is not None:
            wall_s = time.monotonic() - self._run_start
            self.stats_label.setText(self._fmt_stats(wall_s, self._run_audio_s))
        self._run_start = None
        self._run_audio_s = None

    @staticmethod
    def _fmt_dur(sec: float) -> str:
        s = int(round(sec))
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        if h: return f"{h}h {m:02d}m {s:02d}s"
        if m: return f"{m}m {s:02d}s"
        return f"{s}s"

    def _fmt_stats(self, wall_s: float, audio_s: Optional[float]) -> str:
        if not audio_s or wall_s <= 0:
            return f"Wall time: {self._fmt_dur(wall_s)}"
        rtf = audio_s / wall_s
        return (f"Wall time: {self._fmt_dur(wall_s)}  ·  "
                f"Audio: {self._fmt_dur(audio_s)}  ·  "
                f"RTF: {rtf:.2f}× realtime")

    def _on_open_output(self):
        if self._last_out_dir and self._last_out_dir.exists():
            subprocess.run(["open", str(self._last_out_dir)], check=False)

    def _append_log(self, line: str):
        # Snoop the audio-length number out of the CLI's final summary line so
        # we can show wall-vs-audio stats on completion. Cheap; ignored on miss.
        m = self._AUDIO_LEN_RE.search(line)
        if m:
            try:
                self._run_audio_s = float(m.group(1)) * 60.0
            except ValueError:
                pass
        if self.log_view.document().blockCount() > 2000:
            cursor = self.log_view.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            for _ in range(500):
                cursor.select(cursor.SelectionType.LineUnderCursor)
                cursor.removeSelectedText()
                cursor.deleteChar()
        self.log_view.append(line)

    def _teardown_thread(self):
        if self.thread:
            self.thread.quit()
            self.thread.wait(2000)
            self.thread = None
            self.worker = None


class MainWindow(QWidget):
    """Top-level window holding the Sync and TTS tabs."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SubPlz")
        self.resize(820, 720)
        self.setAcceptDrops(True)

        self.sync_tab = SyncTab()
        self.tts_tab = TTSTab()

        self.tabs = QTabWidget()
        self.tabs.addTab(self.sync_tab, "Sync / Anki")
        self.tabs.addTab(self.tts_tab, "Audiobook (TTS)")

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.tabs)
        self.setLayout(layout)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
