"""Split over-long subtitle cues into shorter, roughly-equal ones.

The aligner emits one cue per script *sentence*. A long monologue or quote is
often a single sentence, so it becomes one unwieldy wall-of-text cue. This pass
runs right before the final write and breaks any cue whose text exceeds a soft
character limit into 2+ pieces at Japanese punctuation — sentence-enders
(。！？…) preferred, then clause commas (、，), then weaker breaks — choosing
cut points near the equal-division offsets so the pieces come out about the same
size rather than one long + one short.

Timing is interpolated within the original cue's [start, end] span, weighted by
character count. This is the same proportional approach the codebase already
uses for un-anchored cues (see align.emit_split_collapse /
_redistribute_interpolated_runs); we never invent timing outside a cue that the
aligner already placed, so the "excellent" alignment is preserved — we only
subdivide a span that was already correct.
"""

import math

# Tier 0 — sentence-final punctuation. Strongly preferred as a break point.
SENTENCE_END = "。．！？!?…"
# Tier 1 — clause-level breaks (commas, etc.).
CLAUSE_BREAK = "、，,；;：:"
# Tier 2 — weaker/soft breaks, used only when nothing better is near the ideal.
SOFT_BREAK = "・‥―─—–／　"
# Closing quotes/brackets that should stay attached to the text *before* a cut,
# e.g. cut after the 」 in 「…だ。」 rather than before it.
CLOSERS = "」』）〉》】｝］〕｠’”〟＞)]}"

_TIER = {}
for _c in SENTENCE_END:
    _TIER[_c] = 0
for _c in CLAUSE_BREAK:
    _TIER.setdefault(_c, 1)
for _c in SOFT_BREAK:
    _TIER.setdefault(_c, 2)

# Characters we absorb into the left piece when extending a cut point past the
# break char itself (repeated punctuation + trailing closing brackets).
_ABSORB = set(SENTENCE_END) | set(CLAUSE_BREAK) | set(SOFT_BREAK) | set(CLOSERS)

# A cue is left alone until it exceeds the limit by this factor, so a cue right
# at the limit (e.g. 125 when the limit is 120) isn't split needlessly.
_SLACK = 1.08


def _candidate_breaks(text):
    """Return [(cut_index, tier), ...] — positions where text may be split.

    cut_index is the index *after* the break (and after any trailing closing
    brackets / repeated punctuation), i.e. text[:cut_index] | text[cut_index:].
    """
    breaks = []
    n = len(text)
    i = 0
    while i < n:
        tier = _TIER.get(text[i])
        if tier is None:
            i += 1
            continue
        j = i + 1
        # Absorb any immediately following punctuation / closers so the cut
        # lands after them; keep the strongest tier seen across that run.
        while j < n and text[j] in _ABSORB:
            t2 = _TIER.get(text[j])
            if t2 is not None and t2 < tier:
                tier = t2
            j += 1
        breaks.append((j, tier))
        i = j
    return breaks


def split_text_balanced(text, max_len):
    """Split text into roughly-equal pieces, each ideally <= max_len chars.

    Picks ceil(len / max_len) pieces and, for each interior division, the
    punctuation break nearest the equal-division offset — biased toward stronger
    punctuation (a comma must be ~max_len/2 chars closer than a period to win).
    Returns [text] unchanged when it's within the soft limit or has no usable
    break point.
    """
    text = text.strip()
    L = len(text)
    if not max_len or max_len <= 0 or L <= max_len * _SLACK:
        return [text]

    breaks = [b for b in _candidate_breaks(text) if 0 < b[0] < L]
    if not breaks:
        return [text]

    n = max(2, math.ceil(L / max_len))
    cuts = []
    prev = 0
    for k in range(1, n):
        ideal = L * k / n
        best = None  # (score, pos)
        for pos, tier in breaks:
            if pos <= prev:
                continue
            score = abs(pos - ideal) + tier * (max_len * 0.5)
            if best is None or score < best[0]:
                best = (score, pos)
        if best and best[1] > prev:
            cuts.append(best[1])
            prev = best[1]

    if not cuts:
        return [text]

    pieces = []
    start = 0
    for c in cuts:
        piece = text[start:c].strip()
        if piece:
            pieces.append(piece)
        start = c
    tail = text[start:].strip()
    if tail:
        pieces.append(tail)

    if len(pieces) <= 1:
        return [text]

    # A piece can still overshoot if punctuation forced an uneven cut; recurse
    # into it. Recursion terminates because each piece is strictly shorter than
    # its parent (or has no break and returns unchanged).
    out = []
    for p in pieces:
        if len(p) > max_len * _SLACK:
            sub = split_text_balanced(p, max_len)
            out.extend(sub)
        else:
            out.append(p)
    return out


def split_long_segments(segments, max_len):
    """Return a new segment list with over-long cues split in place.

    Each split cue's [start, end] span is divided among its pieces in proportion
    to character count; the final piece keeps the original end to avoid float
    drift. Segments are reconstructed with the same class as the input so the
    writer's duck-typed .vtt()/.start/.end/.text all keep working.
    """
    if not max_len or max_len <= 0:
        return segments

    out = []
    for seg in segments:
        pieces = split_text_balanced(seg.text, max_len)
        if len(pieces) <= 1:
            out.append(seg)
            continue

        total = max(seg.end - seg.start, 0.001)
        weights = [max(len(p), 1) for p in pieces]
        total_w = sum(weights)
        cursor = seg.start
        new = []
        for p, w in zip(pieces, weights):
            dur = total * (w / total_w)
            new.append(type(seg)(p, cursor, cursor + dur))
            cursor += dur
        new[-1].end = seg.end
        out.extend(new)
    return out
