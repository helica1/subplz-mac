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

## Usage

### Sync an audiobook to its epub (primary use case)

Put one audio file and one text file in a folder, point `subplz` at it:

```bash
/sync/
└── /My Audiobook/
   ├── book.m4b      # or .mp3, .m4a, .mkv, .mp4, etc.
   └── book.epub     # or .txt, .srt, .vtt, .ass
```

```bash
.venv/bin/subplz sync -d "/sync/My Audiobook" --lang ja --model turbo --mlx --respect-grouping
```

Output: `/sync/My Audiobook/book.srt` with one cue per epub sentence, timed to the audio.

Use it in [ttu-reader](https://github.com/Renji-XD/ttu-whispersync), MPV, or any subtitle-capable player. For dictionary-popup immersion reading, see the [Other Cool Projects](#other-cool-projects) below.

### Generate subs from audio without a script (`gen`)

For media where you only have audio and want Whisper-transcribed subs:

```bash
.venv/bin/subplz gen -d "/path/to/folder" --lang ja --model turbo --mlx
```

`turbo` is recommended for `gen` since there's no reference text to fall back on — accuracy matters more than for `sync`.

### Batch a whole folder

The GUI handles batch natively: drop a folder containing audiobook subfolders (each with one audio + one text), and each gets processed sequentially with per-book and batch progress.

From the CLI:

```bash
.venv/bin/subplz sync -d "/sync/Book1" "/sync/Book2" "/sync/Book3" --lang ja --model turbo --mlx --respect-grouping
```

### Tips

- **One audio file + one text file per folder.** If a folder has multiple of either, things get ambiguous.
- **Sort order matters** when an audiobook is split into multiple files: ensure your OS sorts them in playback order (by name, usually).
- **By default we overwrite** any existing `.srt`. Use `--no-overwrite` if you don't want that.
- **First MLX run downloads a model** (~1.5 GB for turbo). Cached at `~/.cache/huggingface/` for all subsequent runs.

---

## Tuning recommendations

### For audiobooks (the main use case)

```bash
.venv/bin/subplz sync -d "/sync/My Book" --lang ja --model turbo --mlx --respect-grouping
```

- A chaptered `m4b` allows the pipeline to split work by chapter. Not required — single MP3s work fine, just slightly less RAM-efficient.
- `epub` is the easiest text source. `.txt` works if you have one (more manual control over what's included).
- Stick with `--respect-grouping` to get 1 sentence per cue. The new `greedy_align` algorithm doesn't suffer from the dialog collapses of the original `nc_align`.
- `turbo` is the default recommendation. `large-v3` is more accurate but ~2× slower. `tiny` and `small` are noisier and produce more interpolated cues on Japanese — only use them if you specifically need the speed gain (rarely needed since MLX-turbo is already ~7 min/book).

### For realigning existing subtitles to a video

```bash
.venv/bin/subplz sync --model turbo --mlx -d "/path/to/Anime Show"
```

- Always use turbo (or large-v3) for video subs — sound effects, music, and non-speech audio benefit from the larger model.
- Try with `--respect-grouping` first. If cues end up grouping unnaturally for video (theme songs, very fast dialog), try `--no-respect-grouping`.

---

## Alass (subtitle-to-subtitle alignment)

`alass` is a separate tool the upstream project bundles for shifting an untimed subtitle file to match a video's actual timing. The bundled binary is Linux-only and doesn't run on Mac.

If you need this feature on Mac:

```bash
brew install alass
```

Then `subplz` will find it on PATH automatically. See the upstream readme for the `--alass` flag usage.

---

## Anki support

`subplz` can generate subs2srs-style Anki decks from your aligned `.m4b` + `.srt` using AnkiConnect. Setup hasn't been re-tested on Mac in this fork — should work but YMMV. From the project root:

```bash
.venv/bin/pip install -e ".[anki]"
```

Then configure `./anki_importer/mapping.json` and run `./anki_importer/anki.sh "<folder>"`. See the [original upstream readme](https://github.com/kanjieater/SubPlz#anki-support) for the full setup walkthrough — that section hasn't been changed in this fork.

---

## Automation suite (advanced)

The upstream project has a config-driven automation suite (`watch` for real-time file watching, `scanner` for scheduled library scans, integration with Bazarr) for processing media libraries unattended. All of that exists in this fork too — the code is unchanged and it should work the same. The Mac-specific changes are scoped to the sync/gen pipeline, not the automation layer.

If you want to use it, see the upstream readme's [Automation Suite section](https://github.com/kanjieater/SubPlz#automation-suite) for the full `config.yml` format and `subplz watch` / `subplz scanner` usage. The pieces are still present; this readme just doesn't duplicate them.

---

## FAQ

### Can I run this with multiple audio files and one script?

Not recommended for most cases. Break long audiobooks into one-text-per-one-audio chunks: each audio file should have its own text file with matching content. The pipeline assumes 1:1 audio↔text pairing.

Exception: if you have one epub and multiple chaptered MP3s where the epub chapters align cleanly to the MP3 boundaries, the chapter-fuzzy-matcher can handle it. `.txt` files don't work as well for this case.

### How do I get a bunch of MP3s into one m4b?

`m4b-tool` (Docker image) is the standard. See [m4b-tool](https://github.com/sandreas/m4b-tool#installation). The Docker version includes the better codecs — use it. The `helpers/merge2.sh` script in this repo wraps it for batch use.

Quick alternative with ffmpeg (works but lower-quality codec):

```bash
for f in "/path/to/mp3s/"*.mp3; do echo "file '$f'" >> mylist.txt; done
ffmpeg -f concat -safe 0 -i mylist.txt -c copy output.mp3
```

### MLX backend says "model not found" or 401 on download

You're hitting the HuggingFace anonymous rate limit. Either retry after a few minutes, or set `HF_TOKEN` from your HuggingFace account.

### Cue 126-style issue: short text spans 10s of audio

That's a Whisper segmentation artifact where one sub gets a wide audio range for short text. The fork's `rebalance_skewed_neighbors` post-process catches the common case (when the next cue has way too little audio) and redistributes. If you see this still happening on edge cases, open an issue.

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
