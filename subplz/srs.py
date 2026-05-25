"""Build an Anki .apkg deck from an audiobook + SRT (subs2srs-style).

For each subtitle cue, slice the audio with a small lead/trail pad and a
short fade in/out (avoids clicks), then optionally verify with silero-vad
that the clip didn't chop a word at either edge. Clips are packaged with
their text into a single .apkg the user can transfer to AnkiDroid — no
AnkiConnect, no Anki sync.

The deck uses a single front=audio, back=text Japanese sentence-mining
note type. Each note's unique GUID is derived from `audio_path + cue_index
+ text` so re-running on the same input doesn't duplicate cards on import.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .logger import logger
from .utils import get_tqdm

tqdm, _ = get_tqdm()


@dataclass
class Cue:
    index: int
    start: float
    end: float
    text: str


def _parse_srt(srt_path: Path) -> list[Cue]:
    """Parse SRT into Cues. Uses the `srt` library if available, else a
    minimal hand-rolled parser so the module still works without the extra."""
    try:
        import srt as srt_lib

        with open(srt_path, "r", encoding="utf-8-sig") as f:
            parsed = list(srt_lib.parse(f.read()))
        return [
            Cue(
                index=i + 1,
                start=s.start.total_seconds(),
                end=s.end.total_seconds(),
                text=s.content.strip(),
            )
            for i, s in enumerate(parsed)
            if s.content.strip()
        ]
    except ImportError:
        return _parse_srt_minimal(srt_path)


def _parse_srt_minimal(srt_path: Path) -> list[Cue]:
    """Fallback SRT parser. Handles the standard `index\\ntimecodes\\ntext`
    block format; tolerates BOM and CRLF."""
    import re

    text = srt_path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    blocks = re.split(r"\n\s*\n", text.strip())
    ts_re = re.compile(
        r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)"
    )

    cues = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 2:
            continue
        # First line is index, second is timecode (or first is timecode if no idx)
        ts_match = ts_re.search(lines[0]) or (ts_re.search(lines[1]) if len(lines) > 1 else None)
        if not ts_match:
            continue
        ts_line_idx = 0 if ts_re.search(lines[0]) else 1
        h1, m1, s1, ms1, h2, m2, s2, ms2 = (int(g) for g in ts_match.groups())
        start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0
        end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0
        body = "\n".join(lines[ts_line_idx + 1 :]).strip()
        if body:
            cues.append(Cue(index=len(cues) + 1, start=start, end=end, text=body))
    return cues


def _slice_audio(
    audio_path: Path,
    out_path: Path,
    start: float,
    end: float,
    fade_ms: int,
    codec: str,
    bitrate: str,
    channels: int,
) -> None:
    """Cut [start, end] from audio_path, apply symmetric fade in/out, encode."""
    duration = max(end - start, 0.05)
    fade_s = fade_ms / 1000.0
    # Don't fade longer than half the clip — would silence short clips.
    fade_s = min(fade_s, duration / 4)

    af_chain = []
    if fade_s > 0:
        af_chain.append(f"afade=t=in:st=0:d={fade_s:.3f}")
        af_chain.append(f"afade=t=out:st={max(duration - fade_s, 0):.3f}:d={fade_s:.3f}")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(audio_path),
    ]
    if af_chain:
        cmd += ["-af", ",".join(af_chain)]
    cmd += ["-vn", "-c:a", codec, "-ac", str(channels)]
    if bitrate:
        cmd += ["-b:a", bitrate]
    cmd += [str(out_path)]

    subprocess.run(cmd, check=True, capture_output=True)


def _probe_audio_layout(audio_path: Path) -> tuple[Optional[int], Optional[int]]:
    """Return `(bitrate_bps, channels)` for the first audio stream of
    `audio_path`, or `(None, None)` on probe failure.

    Tries the audio stream's own bit_rate first (the most honest number);
    falls back to the container's overall bit_rate when the stream doesn't
    advertise one (common for some lossless / container-wrapped sources).
    """
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=bit_rate,channels:format=bit_rate",
                "-of", "default=nw=1",
                str(audio_path),
            ],
            check=True, capture_output=True, text=True,
        ).stdout
    except subprocess.CalledProcessError:
        return (None, None)

    stream_br: Optional[int] = None
    format_br: Optional[int] = None
    channels: Optional[int] = None
    seen_stream = False
    for line in out.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        if v in ("", "N/A"):
            continue
        if k == "bit_rate":
            # Same key on both stream and format scopes; ffprobe lists the
            # stream first under `-select_streams a:0`.
            if not seen_stream:
                try:
                    stream_br = int(v)
                    seen_stream = True
                except ValueError:
                    pass
            else:
                try:
                    format_br = int(v)
                except ValueError:
                    pass
        elif k == "channels":
            try:
                channels = int(v)
            except ValueError:
                pass

    return (stream_br or format_br, channels)


def _resolve_codec_params(
    audio_path: Path,
    requested_bitrate: str,
    requested_channels: str,
) -> tuple[str, int]:
    """Turn `--bitrate auto` / `--channels auto` into concrete values.

    Lossless sources advertise an absurd bitrate (~1.4 Mbps for CD-quality
    FLAC). We cap auto at 192k for the lossy clip target — anything above
    that is wasted on speech.
    """
    src_bps, src_channels = _probe_audio_layout(audio_path)

    if requested_bitrate == "auto":
        if src_bps is None:
            br_str = "96k"  # safe default when we can't tell
        else:
            # Round to the nearest k for a clean filename/log; cap at 192k.
            kbps = min(max(src_bps // 1000, 32), 192)
            br_str = f"{kbps}k"
    else:
        br_str = requested_bitrate

    if requested_channels == "auto":
        ch = src_channels or 1
    elif requested_channels == "mono":
        ch = 1
    elif requested_channels == "stereo":
        ch = 2
    else:
        try:
            ch = int(requested_channels)
        except ValueError:
            ch = src_channels or 1
    ch = max(1, min(ch, 2))

    return br_str, ch


def _extract_cover(audio_path: Path, out_path: Path) -> Optional[Path]:
    """Pull embedded cover art from an audio file. Returns the path to the
    written image on success, or None if there's no cover (or ffmpeg fails).

    Works for ID3 APIC frames (mp3), MP4 cover atoms (m4b/m4a), and FLAC
    pictures. We just ask ffmpeg to copy the attached picture stream out;
    it figures out the format. Output extension determines the encoding
    (jpg/png — we use jpg so it stays small in the .apkg).
    """
    # `-an` strips audio; `-vcodec copy` keeps the original picture if it's
    # already jpeg/png. For non-jpeg sources, ffmpeg transcodes to jpg via
    # the output extension.
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(audio_path),
        "-an", "-vframes", "1",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        return None
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    return None


def _slugify(s: str) -> str:
    """Shortish filesystem-safe slug from a free-form string."""
    keep = "".join(c if c.isalnum() else "_" for c in s)
    return keep.strip("_")[:64] or "deck"


def _clip_filename(audio_stem: str, cue_index: int, ext: str) -> str:
    return f"{audio_stem}_{cue_index:05d}.{ext}"


def _hash_guid(audio_name: str, cue_index: int, text: str) -> str:
    h = hashlib.sha1(f"{audio_name}|{cue_index}|{text}".encode("utf-8")).hexdigest()
    return h[:16]


def _build_deck(
    cues: list[Cue],
    clip_paths: dict[int, Path],
    deck_name: str,
    audio_stem: str,
    clip_ext: str,
    cover_path: Optional[Path] = None,
):
    """Construct a genanki Deck + Package referencing the audio clip files."""
    import genanki

    # Fixed model + deck IDs derived from a stable string so re-imports
    # update the same note type / deck rather than creating duplicates.
    # v2 model: added Image field for audiobook cover art.
    model_id = int(hashlib.sha1(b"subplz-srs-model-v2").hexdigest()[:8], 16)
    deck_id = int(hashlib.sha1(f"subplz-srs-deck::{deck_name}".encode()).hexdigest()[:8], 16)

    model = genanki.Model(
        model_id,
        "SubPlz SRS",
        fields=[
            {"name": "Audio"},
            {"name": "Image"},
            {"name": "Expression"},
            {"name": "Source"},
        ],
        templates=[
            {
                "name": "Listening",
                "qfmt": "{{Audio}}{{#Image}}<br>{{Image}}{{/Image}}",
                "afmt": '{{FrontSide}}<hr id="answer">{{Expression}}<br><br><span style="color:#888;font-size:0.8em">{{Source}}</span>',
            },
        ],
        css=(
            ".card { font-family: -apple-system, sans-serif; font-size: 28px; "
            "text-align: center; color: #222; background: #fafafa; }"
            " .card img { max-width: 60%; max-height: 240px; border-radius: 6px; }"
        ),
    )

    deck = genanki.Deck(deck_id, deck_name)
    media_files: list[str] = []
    image_html = ""
    if cover_path is not None and cover_path.exists():
        media_files.append(str(cover_path))
        image_html = f'<img src="{cover_path.name}">'

    for cue in cues:
        clip = clip_paths.get(cue.index)
        if clip is None:
            continue
        clip_name = clip.name
        media_files.append(str(clip))
        note = genanki.Note(
            model=model,
            fields=[
                f"[sound:{clip_name}]",
                image_html,
                cue.text.replace("\n", "<br>"),
                audio_stem,
            ],
            guid=_hash_guid(audio_stem, cue.index, cue.text),
        )
        deck.add_note(note)

    pkg = genanki.Package(deck)
    pkg.media_files = media_files
    return pkg


def run_srs(inputs):
    """Entry point for the `subplz srs` subcommand.

    `inputs` is the argparse Namespace (run.py routes non-sync/gen
    subcommands as raw args). Required: audio, text, output_dir. Optional:
    pad_ms, fade_ms, vad_check, codec, bitrate, deck_name.
    """
    # --audio / --text use nargs="+", so argparse may hand us a list.
    audio_in = inputs.audio[0] if isinstance(inputs.audio, list) else inputs.audio
    text_in = inputs.text[0] if isinstance(inputs.text, list) else inputs.text
    if not audio_in or not text_in or not inputs.output_dir:
        raise ValueError("srs requires --audio, --text, and --output-dir")
    audio_path = Path(audio_in).expanduser().resolve()
    srt_path = Path(text_in).expanduser().resolve()
    output_dir = Path(inputs.output_dir).expanduser().resolve()

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    if not srt_path.is_file():
        raise FileNotFoundError(f"SRT not found: {srt_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    audio_stem = audio_path.stem
    deck_name = inputs.deck_name or audio_stem
    pad_ms = inputs.pad_ms
    fade_ms = inputs.fade_ms
    vad_check = inputs.vad_check
    codec = inputs.codec
    bitrate, channels = _resolve_codec_params(
        audio_path,
        inputs.bitrate,
        getattr(inputs, "channels", "auto"),
    )
    layout = "mono" if channels == 1 else f"{channels}ch"
    logger.info(f"🎚  Encoding clips: {codec} @ {bitrate} {layout}")
    clip_ext = "opus" if codec == "libopus" else ("mp3" if codec == "libmp3lame" else "m4a")

    cues = _parse_srt(srt_path)
    if not cues:
        raise RuntimeError(f"No cues parsed from {srt_path}")
    logger.info(f"📜 Parsed {len(cues)} cues from {srt_path.name}")

    # Clamp end-of-audio so the last cue's pad doesn't run past EOF.
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(audio_path)],
            check=True, capture_output=True, text=True,
        )
        media_duration = float(probe.stdout.strip())
    except Exception:
        media_duration = float("inf")

    media_dir = output_dir / f"{_slugify(deck_name)}_media"
    media_dir.mkdir(parents=True, exist_ok=True)

    # Pull embedded cover art (album art / book cover) from the audio file,
    # if any. Used as a per-card image so the deck has visual identity
    # even though the source is audio-only.
    cover_path: Optional[Path] = None
    if getattr(inputs, "cover", True):
        cover_candidate = media_dir / f"cover_{_slugify(deck_name)}.jpg"
        cover_path = _extract_cover(audio_path, cover_candidate)
        if cover_path is not None:
            logger.info(f"🖼  Extracted cover art ({cover_path.stat().st_size // 1024} KB)")
        else:
            logger.info("🖼  No embedded cover art found; cards will have no image.")

    pad_s = pad_ms / 1000.0
    clip_paths: dict[int, Path] = {}
    skipped_cuts: list[tuple[int, str]] = []
    flagged_cuts: list[int] = []

    def cut_one(cue: Cue, extra_pad: float = 0.0) -> tuple[Cue, Optional[Path], bool, Optional[str]]:
        start = max(cue.start - pad_s - extra_pad, 0.0)
        end = min(cue.end + pad_s + extra_pad, media_duration)
        # Cue falls past the end of the media (e.g. SRT was generated from a
        # longer source than the audio we have here) — skip cleanly.
        if end <= start + 0.05:
            return cue, None, False, "past end of media"
        out = media_dir / _clip_filename(audio_stem, cue.index, clip_ext)
        try:
            _slice_audio(audio_path, out, start, end, fade_ms, codec, bitrate, channels)
        except subprocess.CalledProcessError as e:
            # Single bad cue (often: ffprobe-reported duration exceeds actual
            # audio data, so a late seek hits nothing) shouldn't kill the rest
            # of the deck. Capture ffmpeg's stderr for the skip reason.
            stderr_blob = e.stderr or b""
            if isinstance(stderr_blob, bytes):
                stderr_blob = stderr_blob.decode("utf-8", errors="replace")
            msg = stderr_blob.strip().splitlines()[-1] if stderr_blob.strip() else f"exit {e.returncode}"
            return cue, None, False, msg[:200]
        return cue, out, True, None

    logger.info(f"🔪 Slicing {len(cues)} clips → {media_dir}")
    workers = max(2, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(cut_one, c): c for c in cues}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="cutting"):
            cue, path, ok, reason = fut.result()
            if ok and path is not None:
                clip_paths[cue.index] = path
            else:
                skipped_cuts.append((cue.index, reason or "unknown"))

    if skipped_cuts:
        # Group by reason so a flood of identical errors collapses into one line.
        by_reason: dict[str, list[int]] = {}
        for idx, reason in skipped_cuts:
            by_reason.setdefault(reason, []).append(idx)
        logger.warning(
            f"⏭  Skipped {len(skipped_cuts)} of {len(cues)} cue(s) during slicing:"
        )
        for reason, idxs in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
            sample = idxs[:5]
            more = f" (+{len(idxs)-5} more)" if len(idxs) > 5 else ""
            logger.warning(f"     • {len(idxs)}× {reason!r} — e.g. cue {sample}{more}")

    if not clip_paths:
        raise RuntimeError("All cues failed to slice — refusing to build an empty deck.")

    if vad_check:
        try:
            from .vad_snap import is_speech_at_edges
        except ImportError:
            logger.warning("silero-vad not installed; skipping edge check. Install the `vad` extra to enable.")
            is_speech_at_edges = None
    else:
        is_speech_at_edges = None

    if is_speech_at_edges is not None:
        logger.info("🎚  VAD edge check (auto-retrying clips that chopped a word)")
        retry_pad_s = 0.2
        for cue in tqdm(cues, desc="vad-check"):
            path = clip_paths.get(cue.index)
            if path is None:
                continue
            try:
                speech_start, speech_end = is_speech_at_edges(str(path))
            except Exception as e:
                logger.debug(f"VAD failed on cue {cue.index}: {e}")
                continue
            if speech_start or speech_end:
                # Retry once with extra pad.
                try:
                    cut_one(cue, extra_pad=retry_pad_s)
                    speech_start2, speech_end2 = is_speech_at_edges(str(path))
                    if speech_start2 or speech_end2:
                        flagged_cuts.append(cue.index)
                except Exception as e:
                    logger.debug(f"VAD retry failed on cue {cue.index}: {e}")
                    flagged_cuts.append(cue.index)

        if flagged_cuts:
            logger.warning(
                f"⚠️  {len(flagged_cuts)} clip(s) still show speech at an edge after retry — "
                f"likely cut mid-word. First few: {flagged_cuts[:10]}"
            )
        else:
            logger.info("✅ VAD edge check passed for all clips")

    apkg_path = output_dir / f"{_slugify(deck_name)}.apkg"
    try:
        pkg = _build_deck(cues, clip_paths, deck_name, audio_stem, clip_ext, cover_path)
        pkg.write_to_file(str(apkg_path))
    except Exception as e:
        # If the cover is what broke us, drop it and try again — the audio
        # clips are the irreplaceable part, the image is just decoration.
        if cover_path is not None:
            logger.warning(f"⚠️  Deck build failed with cover ({e}); retrying without it.")
            pkg = _build_deck(cues, clip_paths, deck_name, audio_stem, clip_ext, None)
            pkg.write_to_file(str(apkg_path))
        else:
            raise
    logger.info(f"📦 Wrote {apkg_path}")

    # Clean up the loose media dir — everything's inside the .apkg now.
    if not getattr(inputs, "keep_media", False):
        shutil.rmtree(media_dir, ignore_errors=True)

    return str(apkg_path)
