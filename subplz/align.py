from rapidfuzz import fuzz
import re
from typing import List

from tqdm import tqdm
from ats.main import Segment
from subplz.cli import PUNCTUATION, START_PUNC, END_PUNC

# Trim script for quick testing
# script = script[:500]
# subs = subs[:1000]

# Use dynamic programming to pick best subs mapping
memo = {}


class ScriptLine:
    def __init__(self, line):
        self.text = line
        # self.txt = re.sub("「|」|『|』|、|。|・|？|…|―|─|！|（|）", "", line)

    def __repr__(self):
        return "ScriptLine(%s)" % self.text


def read_script(file):
    for line in file:
        line = line.rstrip("\n")
        if line == "":
            continue
        yield line


def get_script(script, script_pos, num_used, sep=""):
    end = min(len(script), script_pos + num_used)
    return sep.join([sub.text for sub in script[script_pos:end]])


def get_base(subs, sub_pos, num_used, sep=""):
    end = min(len(subs), sub_pos + num_used)
    return sep.join([sub.text for sub in subs[sub_pos:end]])


def get_best_sub_n(
    script,
    subs,
    script_pos,
    num_used_script,
    last_script_pos,
    sub_pos,
    max_subs,
    last_sub_to_test,
    max_merge_count,
):
    t_best_score = 0
    t_best_used_sub = 1

    line = get_script(script, script_pos, num_used_script)

    remaining_subs = last_sub_to_test - sub_pos

    for num_used_sub in range(1, min(max_subs, remaining_subs) + 1):
        base = get_base(subs, sub_pos, num_used_sub)
        curr_score = fuzz.ratio(base, line) / 100.0 * min(len(line), len(base))
        tot_score = curr_score + calc_best_score(
            script,
            subs,
            script_pos + num_used_script,
            last_script_pos,
            sub_pos + num_used_sub,
            last_sub_to_test,
            max_merge_count,
        )
        if tot_score > t_best_score:
            t_best_score = tot_score
            t_best_used_sub = num_used_sub

    return (t_best_score, t_best_used_sub)


best_script_score_and_sub = {}


def calc_best_score(
    script,
    subs,
    script_pos,
    last_script_pos,
    sub_pos,
    last_sub_to_test,
    max_merge_count,
):
    if script_pos >= len(script) or sub_pos >= len(subs):
        return 0

    key = (script_pos, sub_pos)
    if key in memo:
        return memo[key][0]

    best_score = 0
    best_used_sub = 1
    best_used_script = 1

    remaining_script = last_script_pos - script_pos

    for num_used_script in range(1, min(max_merge_count, remaining_script) + 1):
        max_subs = max_merge_count if num_used_script == 1 else 1
        t_best_score, t_best_used_sub = get_best_sub_n(
            script,
            subs,
            script_pos,
            num_used_script,
            last_script_pos,
            sub_pos,
            max_subs,
            last_sub_to_test,
            max_merge_count,
        )

        if t_best_score > best_score:
            best_score = t_best_score
            best_used_sub = t_best_used_sub
            best_used_script = num_used_script

    if best_used_script > 1:
        # Do one more fitting
        t_best_score, t_best_used_sub = get_best_sub_n(
            script,
            subs,
            script_pos,
            best_used_script,
            last_script_pos,
            sub_pos,
            max_merge_count,
            last_sub_to_test,
            max_merge_count,
        )
        if t_best_score > best_score:
            best_score = t_best_score
            best_used_sub = t_best_used_sub

    key = (script_pos, sub_pos)
    memo[key] = (best_score, best_used_sub, best_used_script)

    # Save best sub pos for this script pos
    best_prev_score, best_sub = best_script_score_and_sub.get(script_pos, (0, None))
    if best_score >= best_prev_score:
        best_script_score_and_sub[script_pos] = (best_score, key)

    return best_score


def get_best_sub_path(script_pos, n, last_script_pos, last_sub_to_test):
    _, key = best_script_score_and_sub[script_pos]
    ret = []
    sub_pos = key[1]

    i = 0
    while i < n and script_pos < last_script_pos and sub_pos < last_sub_to_test:
        ret.append((script_pos, sub_pos))
        decision = memo[(script_pos, sub_pos)]
        num_used_sub = decision[1]
        num_used_script = decision[2]
        sub_pos += num_used_sub
        script_pos += num_used_script
        i += 1
    return ret


def test_sub_pos(
    script,
    subs,
    script_pos,
    last_script_pos,
    first_sub_to_test,
    last_sub_to_test,
    max_merge_count,
):
    for sub_pos in range(last_sub_to_test - 1, first_sub_to_test - 1, -1):
        calc_best_score(
            script,
            subs,
            script_pos,
            last_script_pos,
            sub_pos,
            last_sub_to_test,
            max_merge_count,
        )


def recursively_find_match(
    script,
    subs,
    result,
    first_script,
    last_script,
    first_sub,
    last_sub,
    max_merge_count,
    bar=None,
):
    if bar is None:
        bar = tqdm(total=1, position=0, leave=True)

    if first_script == last_script or first_sub == last_sub:
        bar.close()
        return

    memo.clear()
    best_script_score_and_sub.clear()
    max_search_context = max_merge_count * 2
    mid = (first_script + last_script) // 2
    start = max(first_script, mid - max_search_context)
    end = min(mid + max_search_context, last_script)

    for script_pos in tqdm(range(end - 1, start - 1, -1), position=1, leave=False):
        test_sub_pos(
            script, subs, script_pos, end, first_sub, last_sub, max_merge_count
        )

    best_path = get_best_sub_path(start, end - start, end, last_sub)
    if len(best_path) > 0:
        for p in best_path:
            if p[0] > mid:
                break
            mid_key = p

        mid_memo = memo[mid_key]
        script_pos = mid_key[0]
        sub_pos = mid_key[1]
        num_used_script = mid_memo[2]
        num_used_sub = mid_memo[1]

        recursively_find_match(
            script,
            subs,
            result,
            first_script,
            script_pos,
            first_sub,
            sub_pos,
            max_merge_count,
            bar,
        )

        scr_out = get_script(script, script_pos, num_used_script, "")
        scr = get_script(script, script_pos, num_used_script, " ‖ ")
        base = get_base(subs, sub_pos, num_used_sub, " ‖ ")

        result.append((script_pos, num_used_script, sub_pos, num_used_sub))

        recursively_find_match(
            script,
            subs,
            result,
            script_pos + num_used_script,
            last_script,
            sub_pos + num_used_sub,
            last_sub,
            max_merge_count,
            bar,
        )
    bar.close()


def remove_tags(line):
    return re.sub("<[^>]*>", "", line)


def get_lines(file):
    for line in file:
        yield line.rstrip("\n")


def read_subtitles(file):
    lines = get_lines(file)
    subs = []
    first_line = next(lines)
    is_vtt = first_line == "WEBVTT"
    if is_vtt:
        assert next(lines) == ""
    last_sub = " "
    while True:
        line = next(lines, None)
        if line is None:  # EOF
            break
        # Match timestamp lines for both VTT and SRT formats
        m = re.findall(
            r"(\d\d:\d\d:\d\d.\d\d\d) --> (\d\d:\d\d:\d\d.\d\d\d)|(\d\d:\d\d.\d\d\d) --> (\d\d:\d\d.\d\d\d)|(\d\d:\d\d.\d\d\d) --> (\d\d:\d\d:\d\d.\d\d\d)|(\d\d:\d\d:\d\d,\d\d\d) --> (\d\d:\d\d:\d\d,\d\d\d)|(\d\d:\d\d,\d\d\d) --> (\d\d:\d\d,\d\d\d)|(\d\d:\d\d,\d\d\d) --> (\d\d:\d\d:\d\d,\d\d\d)",
            line,
        )
        if not m:
            if not line.isdigit() and line:
                print(
                    f'Warning: Line "{line}" did not look like a valid VTT/SRT input. There could be issues parsing this sub'
                )
            continue

        match_pair = [list(filter(None, x)) for x in m][0]
        sub_start = match_pair[0].replace(",", ".")  # Convert SRT to VTT format
        sub_end = match_pair[1].replace(",", ".")

        # Read the subtitle text
        line = next(lines)
        sub_text = []
        while line:
            sub_text.append(remove_tags(line))
            try:
                line = next(lines)
            except StopIteration:
                line = None
            if line == "":
                break

        sub = " ".join(sub_text)
        if sub and last_sub != sub and sub not in [" ", "[音楽]"]:
            last_sub = sub
            subs.append(Segment(sub, sub_start, sub_end))
        elif last_sub == sub and subs:
            subs[-1].end = sub_end

    return subs


def to_float(time_str):
    time_components = time_str.split(":")[::-1]
    total_seconds = 0
    for i, component in enumerate(time_components):
        total_seconds += float(component) * (60**i)
    return total_seconds


def greedy_align(split_script, subs_file, lookahead=30, min_score=45):
    """Re-time script sentences against the step-3 (pre-grouping) SRT.

    The previous implementation (nc_align) used recursive divide-and-conquer to
    pick anchor points, which collapses badly in dialog-dense regions: a wrong
    top-level anchor can poach sub cues that should belong to a deeper
    recursion, leaving the deeper region with too few subs and forcing tens of
    script sentences into one wall-of-text cue.

    Greedy matching is the simpler, more robust alternative:
      - For each script sentence (in order), scan a `lookahead`-sized window
        of step-3 subs starting at the cursor.
      - Try matching the sentence against 1, 2, or 3 consecutive subs
        (Whisper sometimes splits a sentence across cues).
      - Pick the highest-scoring match. If above `min_score`, emit a cue with
        the matched sub(s)' timing and advance the cursor past them.
      - If no match is good enough, interpolate timing just after the previous
        cue (charge ~50 ms per char, min 1 s).

    Monotonic order means we never re-use a sub, so two adjacent script
    sentences can't both grab the same audio range. The window keeps us
    resilient to occasional missed transcription chunks (we can skip ahead a
    few subs to find a better anchor).
    """
    with open(split_script, encoding="utf-8") as s:
        script = [line.rstrip("\n") for line in s if line.strip()]
    with open(subs_file, encoding="utf-8") as srt:
        subs = read_subtitles(srt)

    if not subs or not script:
        return []

    new_subs = []
    sub_cursor = 0
    last_end = to_float(subs[0].start)
    matched = 0
    interpolated = 0

    print(
        f"🤝 Greedy alignment: {len(script)} script sentences against "
        f"{len(subs)} sub cues (lookahead={lookahead}, min_score={min_score})"
    )
    for s_idx, script_text in enumerate(tqdm(script)):
        window_end = min(len(subs), sub_cursor + lookahead)
        best = (-1.0, -1, 1)  # (adjusted_score, sub_idx, n_consecutive)

        if sub_cursor >= len(subs):
            # No more audio. Interpolate the rest.
            duration = max(len(script_text) * 0.05, 1.0)
            new_subs.append(Segment(script_text, last_end, last_end + duration))
            last_end += duration
            interpolated += 1
            continue

        # Whisper happily splits a single long sentence into 10+ fragments at
        # every comma. Fixed n=1..3 misses these. Instead, grow `combined`
        # incrementally and keep extending until either the combined text
        # exceeds ~1.5x the script length OR the combined audio duration
        # exceeds what could plausibly narrate the script sentence (with a
        # minimum so brief utterances next to silence still find a match).
        script_len = max(len(script_text), 1)
        # JA narration ≈ 0.2 s/char in steady prose; allow up to 0.5 s/char as
        # the ceiling, with a 15 s floor for short utterances. Cue-126 had a
        # 30-char sentence spanning 87 s of audio — this cap would have
        # rejected it.
        max_audio_dur = max(15.0, script_len * 0.5)
        for start_idx in range(sub_cursor, window_end):
            combined = ""
            start_t = to_float(subs[start_idx].start)
            for n in range(1, window_end - start_idx + 1):
                combined += subs[start_idx + n - 1].text
                if not combined:
                    continue
                # Audio span of subs[start_idx : start_idx+n]
                end_t = to_float(subs[start_idx + n - 1].end)
                combined_dur = end_t - start_t
                # Tolerant scoring: low-quality models (e.g. `tiny` on JA)
                # mistranscribe characters, dropping fuzz.ratio below the
                # acceptance threshold even when the boundary is correct.
                # partial_ratio finds the best substring of combined that
                # matches script_text — more forgiving of noise — but we
                # only use it when the lengths are comparable. Otherwise a
                # short script can spuriously match any long combined that
                # happens to contain a similar substring, causing the cursor
                # to advance past audio that belonged to subsequent script
                # sentences (we saw 130-cue interpolation cascades from this).
                score_ratio = fuzz.ratio(script_text, combined)
                len_ratio = len(combined) / max(script_len, 1)
                if script_len >= 12 and 0.6 <= len_ratio <= 1.4:
                    score_partial = fuzz.partial_ratio(script_text, combined) * 0.75
                    raw_score = max(score_ratio, score_partial)
                else:
                    raw_score = score_ratio
                # Penalize for skipping ahead — we'd rather take an in-order
                # match than jump past many subs for a marginally better one.
                # Penalty is small (0.5/sub) so it doesn't dominate when a
                # later position genuinely matches much better.
                position_penalty = (start_idx - sub_cursor) * 0.5
                adjusted = raw_score - position_penalty
                # Tie-break preference for smaller n: when two candidates
                # both pass min_score and are nearly equal, prefer the one
                # that consumes fewer subs (less likely to swallow audio
                # that belongs to subsequent script sentences). Critically,
                # the new candidate must itself clear min_score — otherwise
                # we'd downgrade a passing match into a failing one.
                tie_break = (
                    adjusted >= min_score
                    and best[0] >= min_score
                    and (best[0] - adjusted) < 1.0
                    and n < best[2]
                )
                if combined_dur <= max_audio_dur and (adjusted > best[0] or tie_break):
                    best = (adjusted, start_idx, n)
                # Stop extending n once either text length or audio duration
                # has overshot — further extension only makes the match worse.
                if (
                    len(combined) >= script_len * 1.5
                    or combined_dur >= max_audio_dur
                ):
                    break

        adjusted_score, best_idx, best_n = best
        if best_idx >= 0 and adjusted_score >= min_score:
            start_t = to_float(subs[best_idx].start)
            end_t = to_float(subs[best_idx + best_n - 1].end)
            # Guarantee non-decreasing timestamps even if Whisper produced
            # tiny overlaps.
            if start_t < last_end:
                start_t = last_end
            if end_t <= start_t:
                end_t = start_t + max(len(script_text) * 0.05, 0.5)
            new_subs.append(Segment(script_text, start_t, end_t))
            sub_cursor = best_idx + best_n
            last_end = end_t
            matched += 1
        else:
            # No usable match in the window. Interpolate.
            duration = max(len(script_text) * 0.05, 1.0)
            new_subs.append(Segment(script_text, last_end, last_end + duration))
            last_end += duration
            interpolated += 1

    print(
        f"✅ Greedy alignment: matched {matched}/{len(script)} sentences "
        f"({interpolated} interpolated, {len(subs) - sub_cursor} subs unused)"
    )

    # Post-process: smooth out runs of interpolated cues sandwiched between
    # two matched anchors. Without this, an interpolated cue gets ~50ms/char
    # of duration and is placed immediately after the previous one — so a 10s
    # audio gap between two anchors can show all the in-between sentences in
    # the first 2 seconds and leave 8 seconds of audio with no subtitle.
    return _redistribute_interpolated_runs(new_subs, subs)


def _redistribute_interpolated_runs(new_subs, original_subs):
    """Smooth interpolated cue runs between matched anchors.

    A cue is "matched" if its [start, end] lines up with one or more
    contiguous original sub cues' timestamps. Anything else is interpolated.
    For each run of interpolated cues between two matched anchors, redistribute
    the run's timing to span the audio gap proportionally to character count.
    """
    # Build a set of (start, end) tuples that came from original subs for fast
    # lookup. We allow approximate match within 50 ms to tolerate the small
    # nudges greedy_align applies.
    matched_starts = sorted(
        set(round(to_float(s.start), 3) for s in original_subs)
    )
    matched_ends = sorted(
        set(round(to_float(s.end), 3) for s in original_subs)
    )

    def _is_matched(cue):
        return (
            round(cue.start, 3) in matched_starts
            or round(cue.end, 3) in matched_ends
        )

    i = 0
    n = len(new_subs)
    while i < n:
        if _is_matched(new_subs[i]):
            i += 1
            continue
        # Found start of an interpolated run.
        run_start = i
        while i < n and not _is_matched(new_subs[i]):
            i += 1
        run_end = i  # exclusive; new_subs[run_end] is the next anchor or off the end

        if run_end >= n:
            # Trailing interpolated run (audio ran out). Leave as-is — the
            # heuristic timing is fine since there's no next anchor to span to.
            break

        # We have an interpolated run [run_start, run_end) bounded by:
        #   - previous anchor: new_subs[run_start - 1].end (or 0 if at file start)
        #   - next anchor: new_subs[run_end].start
        gap_start = new_subs[run_start - 1].end if run_start > 0 else new_subs[run_start].start
        gap_end = new_subs[run_end].start
        gap = gap_end - gap_start
        if gap <= 0.05:
            continue  # No real audio gap; leave as-is.

        run = new_subs[run_start:run_end]
        weights = [max(len(c.text), 1) for c in run]
        total_w = sum(weights)
        # Cap each interpolated cue at a sensible display duration. If the
        # proportional share exceeds this, the excess becomes a silent gap
        # (better UX than showing a 4-char utterance for 87 seconds — that
        # bug used to show up in the full audiobook output as e.g. cue 126).
        # JA narration: ~0.2 s/char. Allow up to 0.5 s/char as ceiling, with
        # a 6 s floor so short sentences still get readable display time.
        def _cue_cap(text):
            return max(6.0, len(text) * 0.5)

        cursor = gap_start
        for c, w in zip(run, weights):
            proportional = gap * (w / total_w)
            dur = min(proportional, _cue_cap(c.text))
            c.start = cursor
            c.end = cursor + dur
            cursor += dur
        # Any leftover time at the end of the run becomes a silent gap before
        # the next matched anchor. That's fine — the audio there likely
        # contains music, silence, or content Whisper couldn't transcribe.

    return new_subs


def nc_align(split_script, subs_file, max_merge_count):
    with open(split_script, encoding="utf-8") as s:
        script = [ScriptLine(line) for line in read_script(s)]
    print(subs_file)
    with open(subs_file, encoding="utf-8") as vtt:
        subs = read_subtitles(vtt)
    new_subs = []

    result = []
    print("🤝 Grouping based on transcript...")
    bar = tqdm(total=0)
    recursively_find_match(
        script, subs, result, 0, len(script), 0, len(subs), max_merge_count, bar
    )
    bar.close()

    # Precompute the median num_used_script across non-final entries so we can
    # detect — and clamp — a runaway first/final cue. Without this guard the
    # original code dumps ALL leading/trailing script into the bounding cues
    # whenever alignment fails to cover one end (subplz/align.py historical bug).
    if len(result) >= 2:
        script_gaps = [result[j + 1][0] - result[j][0] for j in range(len(result) - 1)]
        sub_gaps = [result[j + 1][2] - result[j][2] for j in range(len(result) - 1)]
        script_gaps_sorted = sorted(script_gaps)
        sub_gaps_sorted = sorted(sub_gaps)
        median_script_per_cue = script_gaps_sorted[len(script_gaps_sorted) // 2]
        median_sub_per_cue = sub_gaps_sorted[len(sub_gaps_sorted) // 2]
    else:
        median_script_per_cue = 1
        median_sub_per_cue = 1
    max_edge_script = max(median_script_per_cue * 5, 3)
    # A "collapse" is a result-pair where the script gap is much bigger than
    # the sub gap relative to typical cues — i.e., alignment failed to find
    # anchors in that region and emitted one cue covering many script sentences
    # against few sub entries. We expand collapses below using the available
    # sub cues' own timestamps as boundaries.
    collapse_script_threshold = max(median_script_per_cue * 4, 3)

    def emit_split_collapse(script_pos, num_used_script, sub_pos, num_used_sub, out):
        """Distribute many script sentences across the available sub cues.

        Each output cue gets one script sentence (in order); its timing comes
        from a proportional slice of the [sub_pos, sub_pos+num_used_sub] range.
        Falls back to one big cue if the sub range is empty.
        """
        if num_used_sub <= 0 or num_used_script <= 0:
            return False
        # Collect the audio range from the available sub entries.
        start_t = to_float(subs[sub_pos].start)
        end_t = to_float(subs[sub_pos + num_used_sub - 1].end)
        total_span = max(end_t - start_t, 0.001)
        # Weight each script sentence by character count so long sentences
        # get proportionally more display time.
        sentences = [script[script_pos + k].text for k in range(num_used_script)]
        weights = [max(len(s), 1) for s in sentences]
        total_w = sum(weights)
        cursor = start_t
        for s, w in zip(sentences, weights):
            dur = total_span * (w / total_w)
            out.append(Segment(s, cursor, cursor + dur))
            cursor += dur
        return True

    for i, (script_pos, num_used_script, sub_pos, num_used_sub) in enumerate(
        tqdm(result)
    ):
        if i == 0:
            # Symmetric leading-edge guard. The original code forced
            # script_pos=0 and sub_pos=0 here, which collapsed every unmatched
            # leading script sentence (and audio chunk) into cue 1. If the
            # audio doesn't start at the very beginning of the script (e.g.
            # publisher intro, music bed, narrator's own preface), we now drop
            # the unmatched lead-in instead of dumping it into one giant cue.
            first_script_pos = result[0][0]
            first_sub_pos = result[0][2]
            if first_script_pos > max_edge_script:
                print(
                    f"⚠️  nc_align: alignment first matched at script sentence "
                    f"{first_script_pos}/{len(script)}; dropping the unmatched lead-in "
                    f"({first_script_pos - max_edge_script} sentence(s)) instead of "
                    f"dumping them into one giant cue. This usually means the audio "
                    f"starts later than the script, or the front matter wasn't fully "
                    f"stripped."
                )
                script_pos = first_script_pos - max_edge_script
            else:
                script_pos = 0
            sub_pos = max(0, first_sub_pos - max_edge_script)

        if i + 1 < len(result):
            num_used_script = result[i + 1][0] - script_pos
            num_used_sub = result[i + 1][2] - sub_pos
        else:
            remaining_script = len(script) - script_pos
            remaining_sub = len(subs) - sub_pos
            if remaining_script > max_edge_script:
                print(
                    f"⚠️  nc_align: alignment covered {script_pos}/{len(script)} script "
                    f"sentences; dropping the unmatched tail ({remaining_script - max_edge_script} "
                    f"sentence(s)) instead of dumping them into one giant cue. "
                    f"This usually means the audio is shorter than the script, or "
                    f"the back half didn't align cleanly."
                )
                num_used_script = max_edge_script
            else:
                num_used_script = remaining_script
            num_used_sub = remaining_sub

        # Mid-gap collapse mitigation: when alignment punted on this region
        # (many script sentences mapped onto few sub cues), spread the script
        # sentences across the sub range proportionally instead of dumping
        # them all into a single multi-paragraph cue.
        if (
            num_used_script > collapse_script_threshold
            and num_used_sub <= max(median_sub_per_cue * 2, 2)
        ):
            print(
                f"⚠️  nc_align: collapse at script[{script_pos}:{script_pos + num_used_script}] "
                f"({num_used_script} sentences) onto only {num_used_sub} sub cue(s); "
                f"distributing across the audio range instead of one wall-of-text cue."
            )
            if emit_split_collapse(
                script_pos, num_used_script, sub_pos, num_used_sub, new_subs
            ):
                continue

        scr_out = get_script(script, script_pos, num_used_script, "")
        scr = get_script(script, script_pos, num_used_script, " ‖ ")
        base = get_base(subs, sub_pos, num_used_sub, " ‖ ")

        # print('Record:', script_pos, scr, '==', base)
        new_subs.append(
            Segment(
                scr_out,
                to_float(subs[sub_pos].start),
                to_float(subs[sub_pos + num_used_sub - 1].end),
            )
        )

    return new_subs


def double_check_misaligned_pairs(segments):
    if not segments or len(segments) < 2:
        return segments

    adjusted_segments = []
    for i, segment in enumerate(segments):
        segment = handle_specific_pattern(segment, segments, i)
        segment = handle_starting_punctuation(segment, adjusted_segments, i)
        segment = handle_ending_punctuation(segment, segments, i)
        adjusted_segments.append(segment)

    return adjusted_segments


def handle_specific_pattern(segment, segments, index):
    PATTERN_1 = r"」「(.{1,2})、$"  # Pattern for '」「ばか、'
    PATTERN_2 = r"」「(.{1})、$"  # Pattern for '」「ば、'
    combined_pattern = f"({PATTERN_1}|{PATTERN_2})"
    match = re.search(combined_pattern, segment.text)
    if match:
        matched_text = match.group(0)
        if index < len(segments) - 1:
            segments[index + 1].text = matched_text + segments[index + 1].text
            segment.text = segment.text[: match.start()]

    return segment


def handle_starting_punctuation(segment, adjusted_segments, index):
    if segment.text and segment.text[0] in END_PUNC and index > 0:
        adjusted_segments[-1].text += segment.text[0]
        segment.text = segment.text[1:]
    return segment


def handle_ending_punctuation(segment, segments, index):
    if segment.text and segment.text[-1] in START_PUNC and index < len(segments) - 1:
        segments[index + 1].text = segment.text[-1] + segments[index + 1].text
        segment.text = segment.text[:-1]
    return segment


def find_punctuation_index(s: str) -> int:
    indices = [i for i, char in enumerate(s) if char in PUNCTUATION]
    return indices


def has_ending_punctuation(s: str) -> bool:
    indices = [i for i, char in enumerate(s) if char in END_PUNC]
    return bool(indices)


def has_punctuation(s: str) -> bool:
    return bool(find_punctuation_index(s))


def has_double_comma(str_starts: str, str_ends: str) -> bool:
    comma = """、"""
    return str_starts[-1] == comma and str_ends[-1] == comma


def count_non_punctuation(s: str) -> int:
    return len([char for char in s if char not in PUNCTUATION])


def find_index_with_non_punctuation_start(indices: List[int]) -> List[int]:
    """Removes sequential indices, keeping only the first occurrence in a sequence."""
    if not indices:
        return []

    result = [indices[0]]

    for i in range(1, len(indices)):
        # Add index if it's not consecutive with the previous index
        if indices[i] != indices[i - 1] + 1:
            result.append(indices[i])
        elif i == 1:  # Keep the first in a consecutive sequence
            result.append(indices[i])
        else:  # Replace the last item with the current index
            result[-1] = indices[i]

    return result


def find_index_with_non_punctuation_end(indices: List[int]) -> List[int]:
    """Removes sequential indices, keeping only the last occurrence in a sequence."""
    if not indices:
        return []
    result = []
    for i in range(len(indices) - 1):
        if indices[i] != indices[i + 1] - 1:
            result.append(indices[i])
    result.append(indices[-1])
    return result


def trim_segments(segments: List["Segment"]) -> List["Segment"]:
    return [
        Segment(text=segment.text.strip(), start=segment.start, end=segment.end)
        for segment in segments
    ]


def print_modified_segments(
    segments,
    new_segments,
    final_segments,
    modified_new_segment_debug_log,
    modified_final_segment_debug_log,
):
    print("Modified Start segments:")
    for index in modified_new_segment_debug_log:
        print(
            f"""
            Original: {segments[index].text}
            Modified: {new_segments[index].text}
            """
        )

    print("Modified End segments:")
    for index in modified_final_segment_debug_log:
        print(
            f"""
            Original: {segments[index].text}
            Modified: {final_segments[index].text}
            """
        )


def shift_align(segments: List[Segment]) -> List[Segment]:
    modified_new_segment_debug_log = []
    modified_final_segment_debug_log = []
    new_segments = []
    for i, segment in enumerate(segments):
        text = segment.text

        # If no punctuation is present, keep the segment unchanged
        if not has_punctuation(text):
            new_segments.append(segment)
            continue

        # Case 2.a: Handle case for starting index
        indices = find_punctuation_index(text)
        if indices:
            start_index = find_index_with_non_punctuation_start(indices)[0]
            # Case 2.c: Handle empty non-punctuation chunks
            # if the substring would result in an empty string if it were removed
            non_punc_count = count_non_punctuation(text[0:start_index])
            if (
                non_punc_count == 0
                or count_non_punctuation(text[start_index + 1 :]) == 0
            ):
                new_segments.append(segment)
                continue
            if non_punc_count <= 2:
                # If the first segment has 2 or fewer non-punctuation characters
                if (
                    i > 0
                    and len(new_segments) > 0
                    and not has_ending_punctuation(new_segments[-1].text[-1])
                    and not has_double_comma(
                        new_segments[-1].text, text[: start_index + 1]
                    )
                ):
                    # Move part of the text to the previous segment
                    prev_segment = new_segments.pop()
                    prev_segment.text += text[
                        : start_index + 1
                    ]  # Include the punctuation
                    new_segments.append(prev_segment)
                    text = text[start_index + 1 :]  # Exclude the punctuation
                    modified_new_segment_debug_log.append(i)
        new_segments.append(Segment(text, segment.start, segment.end))

    final_segments = []
    for i, segment in enumerate(new_segments):
        text = segment.text
        indices = find_punctuation_index(text)
        if indices:
            last_index = find_index_with_non_punctuation_end(indices)[-1]
            non_punc_count = count_non_punctuation(text[last_index:])
            if non_punc_count == 0 or (len(text[last_index:])) == len(text):
                final_segments.append(segment)
                continue
            if count_non_punctuation(text[indices[-1] :]) <= 2:
                # Move part of the text to the next segment
                if (
                    i + 1 < len(new_segments)
                    and not has_ending_punctuation(new_segments[i + 1].text[0])
                    and not has_double_comma(
                        new_segments[i + 1].text[0], text[: start_index + 1]
                    )
                ):
                    next_segment = segments[i + 1]
                    next_segment.text = (
                        text[last_index + 1 :] + next_segment.text
                    )  # Keep the punctuation
                    text = text[: last_index + 1]  # Exclude the punctuation
                    final_segments.append(Segment(text, segment.start, segment.end))
                    new_segments[i + 1] = next_segment
                    modified_final_segment_debug_log.append(i)
                    continue
        final_segments.append(Segment(text, segment.start, segment.end))

    # print_modified_segments(segments, new_segments, final_segments, modified_new_segment_debug_log, modified_final_segment_debug_log)
    return trim_segments(double_check_misaligned_pairs(final_segments))
