# SubPlz🫴 — macOS Apple Silicon fork

🫴 Generate per-sentence subtitles for any audiobook by aligning it to its source ebook. Designed for **Japanese immersion reading** but works on any language Whisper supports.

> This is a macOS Apple Silicon fork of [SubPlz by kanjieater](https://github.com/kanjieater/SubPlz). It adds an MLX-Whisper backend (Metal + Neural Engine), a rewritten alignment algorithm, VAD-based timing refinement, and a drag-and-drop GUI. The upstream project supports Linux/CUDA/Docker; this fork is built and validated end-to-end on Apple Silicon.

---

## ✨ What's new in this fork

| | Upstream SubPlz | This fork |
|---|---|---|
| **Apple Silicon** | CPU-only via faster-whisper (5.6× realtime) | Native MLX (Metal + Neural Engine), **~76× realtime** on M-series Max |
| **Alignment** | `nc_align` (recursive); collapses in dialog | `greedy_align` (monotonic + adaptive-n + partial-ratio fallback); 1 sentence per cue |
| **Timing** | Word-level from Whisper | VAD-snapped + char-rate-rebalanced; cues bracket actual speech |
| **Front/back matter** | Included in SRT | Filtered (title pages, copyright, TOC, author bios) |
| **GUI** | CLI only | Drag-and-drop PySide6 app with batch mode |

### Benchmark — Japanese 9 h 9 min audiobook, `turbo` model

| Backend | Transcription | Realtime ratio | Wall clock |
|---|---|---|---|
| Reference: RTX 3090 + CUDA | ~30 min | ~18× | ~30 min |
| This fork: M-series Max CPU (faster-whisper) | 98 min | 5.6× | ~1.6 h |
| **This fork: M-series Max MLX (Metal + ANE)** | **6.7 min** | **82×** | **~7 min** |

---

## 🍎 Quick start (macOS Apple Silicon)

```bash
# 1. Install Homebrew if you don't have it: https://brew.sh
# 2. Install Python 3.11 and ffmpeg
brew install python@3.11 ffmpeg

# 3. Clone and set up a venv
git clone https://github.com/helica1/subplz-mac.git
cd subplz-mac
/opt/homebrew/bin/python3.11 -m venv .venv

# 4. Install with MLX + VAD + GUI extras
.venv/bin/pip install -e ".[mac]"
```

**Sync an audiobook from the CLI:**

```bash
.venv/bin/subplz sync \
  --audio "/path/to/book.mp3" \
  --text  "/path/to/book.epub" \
  --output-dir "/path/to/output" \
  --lang ja --model turbo --mlx --respect-grouping
```

**Or use the GUI:**

```bash
.venv/bin/python subplz_gui.py
```

Drag and drop a single audio + epub pair, or drop a folder of audiobook subfolders for batch processing. Progress bars per-book and per-batch; "Open output folder" when complete.

---

## 🎛️ Flags worth knowing

- `--mlx` — use MLX-Whisper backend. Apple Silicon only. Vastly faster than CPU; recommended.
- `--vad-snap` / `--no-vad-snap` — snap cue boundaries to silero-vad-detected speech (default on).
- `--respect-grouping` — re-time each script sentence as one cue (uses `greedy_align`).
- `--model` — `turbo` is the sweet spot on MLX (large-v3-turbo). `large-v3` for max accuracy at half the speed. `tiny` works but produces interpolation gaps in dialog-dense regions on Japanese; use turbo unless you have a reason.
- `--lang` — `ja`, `en`, etc. Default `ja`.
- `--overwrite` / `--rerun` — overwrite existing `.srt` / re-process files marked as already done.

Run `subplz sync -h` for the full list.

---

## Technologies & techniques

- **MLX-Whisper** — Apple's MLX framework, runs Whisper on Metal GPU + Neural Engine
- **faster-whisper / CTranslate2** — fallback transcription backend (CPU on Mac)
- **stable-ts** — Whisper word-timestamp refinement
- **silero-vad** — voice activity detection for cue boundary snapping
- **rapidfuzz** — fast fuzzy string matching for alignment
- **pysbd** — sentence boundary detection (works for JA)
- **EbookLib** + **BeautifulSoup** — epub parsing

---

## Other cool projects

The best Japanese reading experience — ttu-reader paired with SubPlz subs:

- https://github.com/Renji-XD/ttu-whispersync
- Demo: https://x.com/kanjieater/status/1834309526129930433

A tool to turn audiobook subs into Visual Novels:

- https://github.com/asayake-b5/audiobooksync2renpy

---

## Credits

This is a community fork of [SubPlz by kanjieater](https://github.com/kanjieater/SubPlz). All credit for the original project, the alignment pipeline foundation, the chapter-fuzzy-matching, the Anki integration, and the broader project vision belongs to the upstream author.

For general SubPlz support and the upstream community, see [KanjiEater's Discord](https://discord.com/invite/agbwB4p) and [The Moe Way Discord](https://learnjapanese.moe/join/).

### Changes in this fork (beyond upstream v4.0.0)

- macOS Apple Silicon support: native MLX-Whisper backend (Metal + Neural Engine), platform-aware device defaults, guarded Linux-only `alass` binary.
- New `greedy_align` algorithm replacing `nc_align`: monotonic left-to-right matching with adaptive sub-grouping, partial-ratio fallback for noisy transcription, model-dependent score threshold.
- VAD-based timing refinement via silero-vad: trim leading/trailing silence within cues, snap boundaries to actual speech transitions within tolerance.
- Char-rate rebalance: redistributes audio between adjacent cues when Whisper's segmentation gives one sub way more audio than its text justifies.
- Epub front/back matter filters: drops title pages, copyright disclaimers, table-of-contents, colophon, author bios.
- PySide6 drag-and-drop GUI with batch mode, progress bars, and an "Open output folder" reveal-in-Finder action.

## License

MIT, matching the upstream project.
