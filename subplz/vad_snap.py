"""Silero-VAD-based cue boundary refinement.

Whisper's word-level timestamps come from cross-attention and have known
artifacts: word ends are systematically truncated by ~100-300 ms, and cue
starts sometimes include leading silence. This module post-processes the
SRT cues by snapping their [start, end] to the nearest VAD-detected
speech-onset/offset within a small tolerance.

The snap only fires when a VAD transition exists within the tolerance —
in continuous narration without breaks, the cue is left alone. So this is
a "tighten if possible, leave alone if not" pass.

Expected impact: cues become audio-grounded to ~50 ms precision for any
sentence with a perceptible pause around it (typical for audiobooks).
Continuous-narration regions keep Whisper's word-level precision (~200 ms).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def ensure_vad_model():
    """Lazy-load silero-vad. Cached after first call."""
    if not hasattr(ensure_vad_model, "_model"):
        from silero_vad import load_silero_vad
        ensure_vad_model._model = load_silero_vad()
    return ensure_vad_model._model


def get_speech_timestamps_for_file(audio_path: str, threshold: float = 0.5):
    """Run silero-vad over a whole audio file. Returns sorted list of
    `(start_sec, end_sec)` speech segments."""
    from silero_vad import read_audio, get_speech_timestamps
    model = ensure_vad_model()
    wav = read_audio(audio_path, sampling_rate=16000)
    ts = get_speech_timestamps(
        wav,
        model,
        threshold=threshold,
        return_seconds=True,
        # Defaults below are tuned for general speech; keeping them.
        # min_speech_duration_ms=250,
        # min_silence_duration_ms=100,
    )
    return [(float(t["start"]), float(t["end"])) for t in ts]


# Back-compat alias for any existing internal callers.
_get_speech_timestamps = get_speech_timestamps_for_file


def is_speech_at_edges(
    audio_path: str,
    edge_ms: float = 30.0,
    threshold: float = 0.5,
) -> tuple[bool, bool]:
    """Check whether speech is active at the very start and end of an audio
    file. Returns `(speech_at_start, speech_at_end)`.

    A `True` for either edge means the cut likely chopped a word — the
    speaker was still talking when the clip began or ended. Caller can
    respond by extending the pad and re-slicing.

    `edge_ms` is the tolerance window: any VAD-detected speech overlapping
    the first/last `edge_ms` milliseconds counts as "active at the edge".
    """
    from silero_vad import read_audio
    model = ensure_vad_model()
    wav = read_audio(audio_path, sampling_rate=16000)
    duration = len(wav) / 16000.0
    if duration <= 0:
        return (False, False)

    from silero_vad import get_speech_timestamps
    ts = get_speech_timestamps(wav, model, threshold=threshold, return_seconds=True)
    if not ts:
        return (False, False)

    edge_s = edge_ms / 1000.0
    speech_at_start = any(float(t["start"]) < edge_s for t in ts)
    speech_at_end = any(float(t["end"]) > duration - edge_s for t in ts)
    return (speech_at_start, speech_at_end)


def _snap_to_speech(t: float, transitions: list[float], tolerance: float) -> float:
    """Return the transition closest to `t` within `tolerance`, else `t`.

    `transitions` must be sorted. Uses bisect for O(log n) lookup over the
    thousands of speech-boundary transitions a typical audiobook produces.
    """
    if not transitions:
        return t
    import bisect
    i = bisect.bisect_left(transitions, t)
    candidates = []
    if i > 0:
        candidates.append(transitions[i - 1])
    if i < len(transitions):
        candidates.append(transitions[i])
    best = min(candidates, key=lambda x: abs(x - t)) if candidates else t
    if abs(best - t) <= tolerance:
        return best
    return t


def vad_snap_cues(
    cues,
    audio_path: str,
    tolerance: float = 0.3,
    trim_margin: float = 0.2,
) -> int:
    """Tighten each cue's boundaries to actual speech via VAD.

    Two complementary operations:

    1. **Trim within cue**: find VAD-detected speech segments overlapping
       [cue.start, cue.end] and shrink the cue to span only those.
       Eliminates leading/trailing silence inside the cue — this is what
       handles Whisper's bad segmentations where a 10s sub contains only
       3s of actual speech (the rest being silence/music the segmenter
       grouped in).

    2. **Snap to nearby transition**: for cues whose Whisper boundary
       falls within `tolerance` of a VAD transition, snap. Fixes Whisper's
       ~200ms word-end truncation in clean cases.

    The two work together: trim shrinks oversized cues; snap polishes the
    rest. `trim_margin` allows the cue to extend slightly past a speech
    segment's edge to include word tails the VAD threshold cut off.

    Mutates cues in place. Returns count of cues whose timing changed.
    """
    if not cues or not Path(audio_path).exists():
        return 0

    speech_segments = get_speech_timestamps_for_file(audio_path)
    if not speech_segments:
        return 0

    speech_segments.sort()
    starts_list = [s for s, _ in speech_segments]
    ends_list = [e for _, e in speech_segments]
    starts_sorted = sorted(starts_list)
    ends_sorted = sorted(ends_list)

    import bisect

    def _speech_in_range(cue_s: float, cue_e: float):
        """Return list of speech segments overlapping [cue_s, cue_e]."""
        # First segment that could overlap: end > cue_s
        lo = bisect.bisect_right(ends_list, cue_s)
        result = []
        for j in range(lo, len(speech_segments)):
            s_start, s_end = speech_segments[j]
            if s_start >= cue_e:
                break
            result.append((s_start, s_end))
        return result

    changed = 0
    for i, cue in enumerate(cues):
        prev_end = cues[i - 1].end if i > 0 else 0.0
        next_start = cues[i + 1].start if i + 1 < len(cues) else float("inf")

        original_start, original_end = cue.start, cue.end
        overlapping = _speech_in_range(cue.start, cue.end)

        if overlapping:
            # Trim to actual speech, with small margin to capture word tails.
            new_start = max(overlapping[0][0] - trim_margin, cue.start)
            new_end = min(overlapping[-1][1] + trim_margin, cue.end)
        else:
            # No speech detected in this cue's range — leave the start alone
            # and just try a small snap on the boundaries.
            new_start = _snap_to_speech(cue.start, starts_sorted, tolerance)
            new_end = _snap_to_speech(cue.end, ends_sorted, tolerance)

        # Now apply small-tolerance snap to refine further (catches Whisper
        # truncation when the speech segment edge is just past the cue end).
        new_start = _snap_to_speech(new_start, starts_sorted, tolerance)
        new_end = _snap_to_speech(new_end, ends_sorted, tolerance)

        # Guards: prevent overlap and zero-duration
        new_start = max(new_start, prev_end)
        new_end = min(new_end, next_start)
        if new_end <= new_start + 0.05:
            continue  # would collapse; keep original

        if new_start != original_start or new_end != original_end:
            cue.start = new_start
            cue.end = new_end
            changed += 1

    return changed


def rebalance_skewed_neighbors(cues, target_rate: float = 4.0) -> int:
    """Redistribute audio between adjacent cues when one is implausibly slow
    and the next is implausibly fast.

    This handles a Whisper-segmentation artifact: sometimes a sub's text is
    short but its [start, end] spans a long audio range (e.g. 15 chars of
    text labeled with 10 seconds of audio that actually contains *two*
    sentences). The next sub then gets squeezed — short audio for long
    text. greedy_align faithfully transcribes the bad boundaries from
    Whisper, producing cues like "15 chars, 10s" then "38 chars, 1.8s".

    Fix: when adjacent cues share a boundary (no gap) and have wildly
    different char-rates (one ≪ target, one ≫ target), re-split their
    combined audio range proportionally to char count.

    `target_rate` is approximately characters-per-second of normal JA
    narration (~4-5 char/sec). We trigger only on egregious skew
    (slow < target/2 AND fast > target*2) to avoid touching reasonable
    variation in pacing.

    Returns count of boundaries that were moved.
    """
    if not cues or len(cues) < 2:
        return 0

    moved = 0
    for i in range(len(cues) - 1):
        a, b = cues[i], cues[i + 1]
        if abs(b.start - a.end) > 0.05:
            continue  # don't reshuffle across an explicit gap

        a_chars = max(len(a.text), 1)
        b_chars = max(len(b.text), 1)
        a_dur = a.end - a.start
        b_dur = b.end - b.start
        if a_dur <= 0.05 or b_dur <= 0.05:
            continue
        a_rate = a_chars / a_dur
        b_rate = b_chars / b_dur

        # Only trigger on egregious skew in either direction.
        a_too_slow_b_too_fast = a_rate < target_rate / 2 and b_rate > target_rate * 2
        a_too_fast_b_too_slow = a_rate > target_rate * 2 and b_rate < target_rate / 2
        if not (a_too_slow_b_too_fast or a_too_fast_b_too_slow):
            continue

        total_dur = a_dur + b_dur
        total_chars = a_chars + b_chars
        new_a_dur = total_dur * a_chars / total_chars
        new_boundary = a.start + new_a_dur
        a.end = new_boundary
        b.start = new_boundary
        moved += 1

    return moved
