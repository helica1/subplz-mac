# Sharing drafts

Three audience-tailored drafts. Pick whichever matches where you're posting, edit freely.

---

## Draft A — r/LearnJapanese (and similar)

**Title:** I made a Mac fork of SubPlz that generates per-sentence SRTs for Japanese audiobooks — 9 hour book → SRT in ~7 minutes via MLX

I've been wanting Kindle-Whispersync-style immersion reading for Japanese audiobooks I own, where the subtitle is the actual epub text (not Whisper's transcription) lined up with the audio. There's a great existing tool for this called [SubPlz by kanjieater](https://github.com/kanjieater/SubPlz) — but it's built around CUDA. I'm on Apple Silicon.

So I forked it and rewrote the parts that didn't fit Mac:

- **MLX-Whisper backend** (Metal GPU + Neural Engine). On my M-series Max it transcribes the `turbo` model at **~82× realtime** — i.e. a 9 hour audiobook is transcribed in about 7 minutes wall clock. For reference, an RTX 3090 does the same job in ~30 minutes via CUDA. The combination of unified memory + Metal + ANE is genuinely competitive.
- **New alignment algorithm** (`greedy_align`). The original recursive matcher collapsed in dialog-dense regions — I'd get cues where one subtitle contained an entire chapter's worth of text. The new algorithm walks the Whisper output monotonically, fuzzy-matches each script sentence with adaptive sub-grouping, and uses a partial-ratio fallback for noisy transcription. On a Murakami audiobook with `tiny`, match rate went from 3% to 50%.
- **VAD-based timing refinement** (silero-vad). Whisper's word-end timestamps are systematically truncated by ~200ms, and cues sometimes include leading silence. I snap each cue to actual speech onsets/offsets within tolerance, and trim leading/trailing silence inside cues. Result: cues bracket exactly the audio of one sentence.
- **Char-rate rebalance**. Detects the specific Whisper artifact where one sub gets 10 seconds of audio for 15 characters of text (the next sub gets squeezed), and redistributes proportionally.
- **Front and back matter filtering**. Drops title page, copyright disclaimer, TOC, colophon, "1949 born in Kyoto..." author bios before they show up as cues at the start/end.
- **PySide6 GUI with drag-and-drop**. Drop a folder of audiobook subdirs, get SRTs out. Real-time progress bars, "Open output folder" on completion.

Validated end-to-end on two Japanese audiobooks (one Kadokawa thriller, one Murakami short story collection). The output is per-sentence cues with sub-second timing — what you'd use with [ttu-reader](https://github.com/Renji-XD/ttu-whispersync) for full immersion-reading, or in a player with subtitles overlaid.

Repo: https://github.com/helica1/subplz-mac

Known limitations: tiny model on Japanese still has interpolation chains in continuous-monologue regions (turbo is genuinely better there); long quoted dialog blocks are kept as single cues (the script splits don't reach inside Japanese「」quotes); SRT cues placed past the end of audio when narrator skips epub content.

Happy to answer questions. PRs welcome (especially if anyone has ideas for the dialog-quote-splitting case).

---

## Draft B — Hacker News (Show HN)

**Title:** Show HN: SubPlz-Mac – Audiobook-to-SRT alignment running 82× realtime on Apple Silicon

I forked [SubPlz](https://github.com/kanjieater/SubPlz), an open-source tool that aligns an audiobook to its source epub to produce sentence-level subtitles. The original is CUDA-only via faster-whisper. I needed it to work on macOS Apple Silicon.

The interesting result: by swapping in MLX-Whisper (Apple's MLX framework using Metal + Neural Engine) and rewriting the alignment algorithm, the same workload that takes an RTX 3090 ~30 minutes via CUDA takes ~7 minutes on an M-series Max. Apple's unified-memory + Metal + ANE stack is well-suited to Whisper's encoder-heavy workload, and the turbo model (32-layer encoder, 4-layer decoder) is the sweet spot.

The alignment side was also interesting. The original recursive divide-and-conquer matcher anchored at midpoints and recursed — fine in clean narration, but in dialog-dense regions it would commit to a bad anchor and collapse 30+ script sentences into one wall-of-text cue. I replaced it with a monotonic greedy matcher with adaptive sub-grouping (Whisper sometimes splits a long sentence into 10+ comma fragments), partial-ratio fallback for noisy transcription (recovers matches that exact-ratio rejects), and a per-cue char-rate rebalance to fix Whisper segmentation artifacts where one sub spans 10s of audio for 3s of speech.

Final layer is silero-vad-based boundary refinement: snap to actual speech onsets/offsets within ±300ms, plus trim leading/trailing silence inside cues. Combined, cues bracket the actual sentence audio with ~100-200ms precision.

Tested end-to-end on Japanese audiobooks. Per-sentence cues, sub-second timing, 88.9% sentences directly matched to audio on the larger turbo model.

Code, README, and benchmarks: https://github.com/helica1/subplz-mac

---

## Draft C — Short Twitter/Mastodon/Bluesky

I forked SubPlz to run on Apple Silicon Macs. M-series Max generates per-sentence subtitles for a 9-hour Japanese audiobook in ~7 minutes via MLX (Metal + Neural Engine). For reference, RTX 3090 takes ~30 min for the same job. Repo: https://github.com/helica1/subplz-mac

---

## Posting checklist

Before posting, double-check the repo URL (`https://github.com/helica1/subplz-mac`) actually exists publicly.

Consider adding:
- A screenshot of the GUI with a real book loaded
- A short screen recording (15–30 sec) of the GUI processing a book and showing the resulting SRT in a player
- The actual SRT file from a public-domain Japanese audiobook so people can preview output without running anything

Subreddits worth posting to:
- `r/LearnJapanese` — primary audience
- `r/japanese` — secondary
- `r/anki` — if you frame the post around the Anki mining use case
- `r/macapps` — for the Apple Silicon angle
- `r/LocalLLaMA` — they like MLX content for the "running ML on Mac" angle (even though Whisper isn't an LLM)
- Hacker News (Show HN) — for the technical/perf angle. Better received with a writeup-style post than a bare repo link.
