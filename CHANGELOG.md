# Changelog

## 2.1.0

### Added
- **Automatic splitting of over-long subtitle cues.** The aligner emits one cue
  per script sentence, so long Japanese monologues or quotes (a single sentence)
  could become one unwieldy wall-of-text cue. A new pass (`subplz/cue_split.py`)
  splits any cue over a soft character limit into 2+ roughly-equal pieces, cut at
  Japanese punctuation — sentence-enders (。！？…) preferred, then clause commas
  (、，), then weaker breaks. Timing is interpolated within the original cue's
  span by character count, so existing alignment is preserved (we only subdivide
  spans the aligner already placed correctly). Closing brackets stay attached;
  runs with no punctuation are left intact rather than chopped mid-word.
- **`--max-cue-length` CLI flag** (`sync` and `gen`). Soft limit, default **120**;
  set `0` to disable. Applied on both the grouped (`--respect-grouping`) and
  non-grouped sync paths, and in `gen`.
- **GUI "Max line length" control.** New spinbox in the Sync settings bar,
  default **100**, range 0–400, which passes `--max-cue-length` to the sync
  subprocess.

### Changed
- Version bumped to 2.1.0.
