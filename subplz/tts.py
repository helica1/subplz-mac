"""TTS for epub → audiobook generation with two pluggable backends.

Backends
--------
- **sbv2**  Style-Bert-VITS2 JP-Extra, in-process (main subplz venv).
  Fast (~7x realtime on M-series CPU), narrator voices are pretrained
  checkpoints downloaded once per voice.
- **irodori**  Aratako/Irodori-TTS-500M-v3, run as a subprocess in a
  sibling venv (Irodori pins `transformers<5`, subplz needs >=5 for SBV2).
  Slower (~0.7x realtime) but zero-shot voice cloning from any wav.

Voice storage
-------------
~/.subplz/voices/
  sbv2/<name>/{config.json, *.safetensors, style_vectors.npy}
  irodori/<name>/{reference.wav, meta.json}

Public CLI surface (wired in subplz.cli + subplz.run):
- `subplz tts --epub PATH --backend [sbv2|irodori] --voice NAME --out DIR`
- `subplz voice list`
- `subplz voice clone --audio PATH --name NAME [--start S] [--duration S]`
- `subplz voice install-preset --name PRESET` (sbv2 narrator presets)
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .logger import logger
from .utils import get_tqdm

tqdm, _ = get_tqdm()


def default_device_for(backend: str) -> str:
    """Best device per backend on this machine.

    SBV2 is small (~250MB) and the MPS padding ops fall back to slow view-ops,
    so CPU wins on Apple Silicon. Irodori is a 500M-param transformer where MPS
    measured ~73% faster than CPU (RTF 1.76 vs 1.02) on M5 Max."""
    is_mac = platform.system() == "Darwin"
    if backend == "irodori":
        return "mps" if is_mac else "cpu"
    return "cpu"


# ----------------------------------------------------------------------- paths

def voices_root() -> Path:
    """Where per-backend voice assets live. Override with SUBPLZ_VOICES_DIR."""
    env = os.environ.get("SUBPLZ_VOICES_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".subplz" / "voices"


def irodori_repo() -> Path:
    """Where the Irodori source tree lives. Override with SUBPLZ_IRODORI_REPO."""
    env = os.environ.get("SUBPLZ_IRODORI_REPO")
    if env:
        return Path(env).expanduser()
    # fallback: tts_spike/irodori sibling of this repo (dev install layout)
    return Path(__file__).resolve().parent.parent / "tts_spike" / "irodori"


def irodori_python() -> Path:
    """Python executable for the Irodori venv. Override with SUBPLZ_IRODORI_PYTHON."""
    env = os.environ.get("SUBPLZ_IRODORI_PYTHON")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parent.parent / "tts_spike" / "irodori_venv" / "bin" / "python"


# --------------------------------------------------------------- voice catalog

# Known SBV2 JP-Extra voice presets. All verified `use_jp_extra: true`.
# Tagged: narrator (calm long-form), character (anime/expressive), or
# emotional (multi-style; useful when we get to per-character voicing).
SBV2_PRESETS: dict[str, dict] = {
    # ---- narrator-grade (calm, suitable for long-form reading) ----
    "rikka_botan_cool": {
        "config":      "https://huggingface.co/RikkaBotan/style_bert_vits2_jp_extra_cool_original/resolve/main/config.json",
        "safetensors": "https://huggingface.co/RikkaBotan/style_bert_vits2_jp_extra_cool_original/resolve/main/rikka_botan_cool.safetensors",
        "style":       "https://huggingface.co/RikkaBotan/style_bert_vits2_jp_extra_cool_original/resolve/main/style_vectors.npy",
        "license":     "CC-BY-SA-4.0",
        "weights_filename": "rikka_botan_cool.safetensors",
        "tag":         "narrator",
        "description": "Female, soft/unhurried (おっとり); author markets for 朗読",
    },
    "koharune-ami": {
        "config":      "https://huggingface.co/litagin/sbv2_koharune_ami/resolve/main/koharune-ami/config.json",
        "safetensors": "https://huggingface.co/litagin/sbv2_koharune_ami/resolve/main/koharune-ami/koharune-ami.safetensors",
        "style":       "https://huggingface.co/litagin/sbv2_koharune_ami/resolve/main/koharune-ami/style_vectors.npy",
        "license":     "amitaro.net terms (credit required, no R-18)",
        "weights_filename": "koharune-ami.safetensors",
        "tag":         "narrator",
        "description": "Female (young adult), corpus recording; 6 styles, calm",
    },
    "amitaro": {
        "config":      "https://huggingface.co/litagin/sbv2_amitaro/resolve/main/amitaro/config.json",
        "safetensors": "https://huggingface.co/litagin/sbv2_amitaro/resolve/main/amitaro/amitaro.safetensors",
        "style":       "https://huggingface.co/litagin/sbv2_amitaro/resolve/main/amitaro/style_vectors.npy",
        "license":     "amitaro.net terms (credit required, no R-18)",
        "weights_filename": "amitaro.safetensors",
        "tag":         "narrator",
        "description": "Same VA as koharune-ami, livestream-trained (more naturalistic)",
    },
    # ---- character-leaning (variety for multi-voice / dialogue) ----
    "lux": {
        "config":      "https://huggingface.co/Lami/Lux-Style-Bert-VITS2-JP-Extra/resolve/main/config.json",
        "safetensors": "https://huggingface.co/Lami/Lux-Style-Bert-VITS2-JP-Extra/resolve/main/Lux_e100_s21300.safetensors",
        "style":       "https://huggingface.co/Lami/Lux-Style-Bert-VITS2-JP-Extra/resolve/main/style_vectors.npy",
        "license":     "CC-BY-4.0",
        "weights_filename": "Lux_e100_s21300.safetensors",
        "tag":         "character",
        "description": "Original female character voice, JP-Extra v2.6.1",
    },
    "rikka_botan_sweet": {
        "config":      "https://huggingface.co/RikkaBotan/style_bert_vits2_jp_extra_sweet_original/resolve/main/config.json",
        "safetensors": "https://huggingface.co/RikkaBotan/style_bert_vits2_jp_extra_sweet_original/resolve/main/rikka_botan_mokyumokyu.safetensors",
        "style":       "https://huggingface.co/RikkaBotan/style_bert_vits2_jp_extra_sweet_original/resolve/main/style_vectors.npy",
        "license":     "Unknown (check repo before commercial use)",
        "weights_filename": "rikka_botan_mokyumokyu.safetensors",
        "tag":         "character",
        "description": "Female, sweet/cute register — pairs with rikka_botan_cool",
    },
    "fumifumi": {
        "config":      "https://huggingface.co/kokushing/style_bert_vits2_fumifumi/resolve/main/config.json",
        "safetensors": "https://huggingface.co/kokushing/style_bert_vits2_fumifumi/resolve/main/fumifumi_e100_s5200.safetensors",
        "style":       "https://huggingface.co/kokushing/style_bert_vits2_fumifumi/resolve/main/style_vectors.npy",
        "license":     "Unknown (no model card)",
        "weights_filename": "fumifumi_e100_s5200.safetensors",
        "tag":         "character",
        "description": "Female, single-voice character (v2.2-JP-Extra)",
    },
    # ---- emotional (multi-style — best for per-character voicing later) ----
    "jvnv-f1-jp": {
        "config":      "https://huggingface.co/litagin/style_bert_vits2_jvnv/resolve/main/jvnv-F1-jp/config.json",
        "safetensors": "https://huggingface.co/litagin/style_bert_vits2_jvnv/resolve/main/jvnv-F1-jp/jvnv-F1-jp_e160_s14000.safetensors",
        "style":       "https://huggingface.co/litagin/style_bert_vits2_jvnv/resolve/main/jvnv-F1-jp/style_vectors.npy",
        "license":     "CC-BY-SA-4.0",
        "weights_filename": "jvnv-F1-jp_e160_s14000.safetensors",
        "tag":         "emotional",
        "description": "Female, 7 emotion styles (anger/sad/happy/etc.)",
    },
    "jvnv-f2-jp": {
        "config":      "https://huggingface.co/litagin/style_bert_vits2_jvnv/resolve/main/jvnv-F2-jp/config.json",
        "safetensors": "https://huggingface.co/litagin/style_bert_vits2_jvnv/resolve/main/jvnv-F2-jp/jvnv-F2_e166_s20000.safetensors",
        "style":       "https://huggingface.co/litagin/style_bert_vits2_jvnv/resolve/main/jvnv-F2-jp/style_vectors.npy",
        "license":     "CC-BY-SA-4.0",
        "weights_filename": "jvnv-F2_e166_s20000.safetensors",
        "tag":         "emotional",
        "description": "Second female, 7 emotion styles",
    },
    "jvnv-m1-jp": {
        "config":      "https://huggingface.co/litagin/style_bert_vits2_jvnv/resolve/main/jvnv-M1-jp/config.json",
        "safetensors": "https://huggingface.co/litagin/style_bert_vits2_jvnv/resolve/main/jvnv-M1-jp/jvnv-M1-jp_e158_s14000.safetensors",
        "style":       "https://huggingface.co/litagin/style_bert_vits2_jvnv/resolve/main/jvnv-M1-jp/style_vectors.npy",
        "license":     "CC-BY-SA-4.0",
        "weights_filename": "jvnv-M1-jp_e158_s14000.safetensors",
        "tag":         "emotional",
        "description": "Adult male, 7 emotion styles",
    },
    "jvnv-m2-jp": {
        "config":      "https://huggingface.co/litagin/style_bert_vits2_jvnv/resolve/main/jvnv-M2-jp/config.json",
        "safetensors": "https://huggingface.co/litagin/style_bert_vits2_jvnv/resolve/main/jvnv-M2-jp/jvnv-M2-jp_e159_s17000.safetensors",
        "style":       "https://huggingface.co/litagin/style_bert_vits2_jvnv/resolve/main/jvnv-M2-jp/style_vectors.npy",
        "license":     "CC-BY-SA-4.0",
        "weights_filename": "jvnv-M2-jp_e159_s17000.safetensors",
        "tag":         "emotional",
        "description": "Second adult male, 7 emotion styles",
    },
    "mofa-girls": {
        "config":      "https://huggingface.co/Mofa-Xingche/girl-style-bert-vits2-JPExtra-models/resolve/main/config.json",
        "safetensors": "https://huggingface.co/Mofa-Xingche/girl-style-bert-vits2-JPExtra-models/resolve/main/NotAnimeJPManySpeaker_e120_s22200.safetensors",
        "style":       "https://huggingface.co/Mofa-Xingche/girl-style-bert-vits2-JPExtra-models/resolve/main/style_vectors.npy",
        "license":     "MIT",
        "weights_filename": "NotAnimeJPManySpeaker_e120_s22200.safetensors",
        "tag":         "emotional",
        "description": "Multi-speaker pack: 4 young females + 1 male, 26 styles each",
    },
}


@dataclass
class Voice:
    backend: str  # "sbv2" | "irodori"
    name: str
    path: Path    # the voice directory
    meta: dict


def list_voices(backend: Optional[str] = None) -> list[Voice]:
    out: list[Voice] = []
    root = voices_root()
    for be in ("sbv2", "irodori") if backend is None else (backend,):
        bdir = root / be
        if not bdir.exists():
            continue
        for vdir in sorted(p for p in bdir.iterdir() if p.is_dir()):
            meta_path = vdir / "meta.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            out.append(Voice(backend=be, name=vdir.name, path=vdir, meta=meta))
    return out


def find_voice(backend: str, name: str) -> Voice:
    vdir = voices_root() / backend / name
    if not vdir.exists():
        raise FileNotFoundError(
            f"Voice '{name}' not found for backend '{backend}'. "
            f"Expected at {vdir}. Run `subplz voice list` to see available voices."
        )
    meta_path = vdir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return Voice(backend=backend, name=name, path=vdir, meta=meta)


# ----------------------------------------------------------- voice management

def _download(url: str, dest: Path) -> None:
    """Streamed download via urllib (no extra dep)."""
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"  ↓ {url}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def install_sbv2_preset(name: str) -> Voice:
    """Download a known SBV2 narrator preset into the voice store."""
    if name not in SBV2_PRESETS:
        raise ValueError(
            f"Unknown SBV2 preset '{name}'. Known: {sorted(SBV2_PRESETS)}"
        )
    preset = SBV2_PRESETS[name]
    vdir = voices_root() / "sbv2" / name
    if vdir.exists() and any(vdir.iterdir()):
        logger.info(f"SBV2 voice '{name}' already installed at {vdir}")
    else:
        _download(preset["config"],      vdir / "config.json")
        _download(preset["safetensors"], vdir / preset["weights_filename"])
        _download(preset["style"],       vdir / "style_vectors.npy")
    meta = {
        "weights_filename": preset["weights_filename"],
        "license": preset["license"],
        "preset": name,
    }
    (vdir / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"Installed SBV2 voice '{name}' -> {vdir}")
    return find_voice("sbv2", name)


def clone_irodori_voice(
    audio_path: Path,
    name: str,
    start_s: float = 60.0,
    duration_s: float = 15.0,
    overwrite: bool = False,
) -> Voice:
    """Auto-trim a clean reference clip from any audio source and register
    it as an Irodori voice. Uses ffmpeg to extract `duration_s` seconds
    starting at `start_s`, mono, 24 kHz.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)
    vdir = voices_root() / "irodori" / name
    if vdir.exists() and not overwrite:
        raise FileExistsError(
            f"Voice '{name}' already exists at {vdir}. Pass overwrite=True to replace."
        )
    vdir.mkdir(parents=True, exist_ok=True)
    ref_path = vdir / "reference.wav"
    cmd = [
        "ffmpeg", "-y", "-ss", str(start_s), "-t", str(duration_s),
        "-i", str(audio_path), "-ac", "1", "-ar", "24000",
        str(ref_path),
    ]
    # NOT text=True: ffmpeg's stderr can include non-UTF-8 bytes (locale
    # control codes, filename echoes) and text=True decodes eagerly,
    # raising even on rc=0. Capture as bytes; decode only if we need to.
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        stderr = r.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg failed: {stderr[-500:]}")
    meta = {
        "source": str(audio_path),
        "start_s": start_s,
        "duration_s": duration_s,
        "sample_rate": 24000,
    }
    (vdir / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"Cloned Irodori voice '{name}' from {audio_path.name} -> {ref_path}")
    return find_voice("irodori", name)


# ---------------------------------------------------------------- text → bits

def extract_sentences(epub_path: Path, lang: str = "ja") -> list[str]:
    """Parse an epub and return a flat list of sentences ready to synthesize.

    Public so callers (GUI, tests) can pre-parse once and reuse across multiple
    TTS runs on the same book — see `--sentences-file` on the CLI."""
    import pysbd
    from .text import Epub
    book = Epub.from_file(str(epub_path))
    paragraphs = [p.text() for p in book.text()]
    full = "\n".join(s for s in paragraphs if s.strip())
    seg = pysbd.Segmenter(language=lang, clean=False)
    sentences: list[str] = []
    for line in full.split("\n"):
        line = line.strip()
        if not line:
            continue
        sentences.extend(s.strip() for s in seg.segment(line) if s.strip())
    return sentences


def _sentences_from_epub(epub_path: Path, lang: str = "ja") -> list[str]:
    """Back-compat shim — call extract_sentences() instead."""
    return extract_sentences(epub_path, lang=lang)


def _format_srt_time(t: float) -> str:
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _write_srt(cues: list[tuple[float, float, str]], path: Path) -> None:
    lines = []
    for i, (start, end, text) in enumerate(cues, 1):
        lines.append(f"{i}\n{_format_srt_time(start)} --> {_format_srt_time(end)}\n{text}\n")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- SBV2 backend

class SBV2Backend:
    """In-process SBV2 JP-Extra synthesis with the fp32 coercion baked in."""

    _bert_loaded = False

    def __init__(self, voice: Voice, device: Optional[str] = None):
        if device is None:
            device = default_device_for("sbv2")
        self.voice = voice
        self.device = device
        self._load()

    def _load(self) -> None:
        from style_bert_vits2.constants import Languages
        from style_bert_vits2.nlp import bert_models
        from style_bert_vits2.tts_model import TTSModel

        if not SBV2Backend._bert_loaded:
            logger.info("Loading BERT (deberta-v2-large-japanese-char-wwm)…")
            m = bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
            m.float()  # transformers 5.x saves in fp16; SBV2 synth is fp32
            bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
            SBV2Backend._bert_loaded = True

        weights = self.voice.meta.get("weights_filename")
        if not weights:
            candidates = list(self.voice.path.glob("*.safetensors"))
            if len(candidates) != 1:
                raise RuntimeError(
                    f"Could not infer SBV2 weights file in {self.voice.path}; "
                    f"set 'weights_filename' in meta.json."
                )
            weights = candidates[0].name
        self._model = TTSModel(
            model_path=self.voice.path / weights,
            config_path=self.voice.path / "config.json",
            style_vec_path=self.voice.path / "style_vectors.npy",
            device=self.device,
        )
        self._model.load()
        # fp16 checkpoint into fp32 model -> dtype mismatch on biases. Coerce.
        net_g = getattr(self._model, "_TTSModel__net_g")
        net_g.float()

    def synthesize_many(self, sentences: list[str]) -> list[tuple[np.ndarray, int]]:
        from style_bert_vits2.constants import Languages
        out: list[tuple[np.ndarray, int]] = []
        for text in tqdm(sentences, desc="SBV2"):
            sr, audio = self._model.infer(text=text, language=Languages.JP)
            out.append((audio, sr))
        return out


# -------------------------------------------------------- Irodori backend (subprocess)

class IrodoriBackend:
    """Subprocess Irodori in its own venv (transformers<5). Spawns once per
    `synthesize_many` call so the model loads once and amortizes."""

    HF_CHECKPOINT = "Aratako/Irodori-TTS-500M-v3"

    def __init__(
        self,
        voice: Voice,
        device: Optional[str] = None,
        num_steps: int = 24,
        cfg_scale_speaker: float = 5.0,
        caption: Optional[str] = None,
    ):
        if device is None:
            device = default_device_for("irodori")
        self.voice = voice
        self.device = device
        self.num_steps = num_steps
        self.cfg_scale_speaker = cfg_scale_speaker
        self.caption = caption
        self._ref_wav = voice.path / "reference.wav"
        if not self._ref_wav.exists():
            raise FileNotFoundError(f"Reference wav missing: {self._ref_wav}")
        py = irodori_python()
        if not py.exists():
            raise FileNotFoundError(
                f"Irodori venv python not found at {py}. "
                f"Set SUBPLZ_IRODORI_PYTHON or install the Irodori env."
            )
        repo = irodori_repo()
        if not (repo / "irodori_tts").exists():
            raise FileNotFoundError(
                f"Irodori repo not found at {repo}. "
                f"Set SUBPLZ_IRODORI_REPO or clone https://github.com/Aratako/Irodori-TTS"
            )
        self._py = py
        self._repo = repo

    def synthesize_to_dir(self, items: list[tuple[int, str]], out_dir: Path) -> None:
        """Synthesize each (index, text) into out_dir/<index:06d>.wav atomically.

        Used by the streaming pipeline (epub_to_audiobook): one subprocess
        call handles the whole batch, and the worker writes per-sentence WAVs
        with absolute indices so resume can pick up exactly where we left off."""
        worker = Path(__file__).resolve().parent / "_irodori_worker.py"
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            req_path = Path(f.name)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            res_path = Path(f.name)
        try:
            req_path.write_text(json.dumps({
                "irodori_repo": str(self._repo),
                "hf_checkpoint": self.HF_CHECKPOINT,
                "ref_wav": str(self._ref_wav),
                "device": self.device,
                "output_dir": str(out_dir),
                "items": [{"index": idx, "text": text} for idx, text in items],
                "num_steps": self.num_steps,
                "cfg_scale_speaker": self.cfg_scale_speaker,
                "caption": self.caption,
            }))
            cmd = [str(self._py), str(worker),
                   "--request-json", str(req_path),
                   "--result-json", str(res_path)]
            logger.info(f"Spawning Irodori worker for {len(items)} sentence(s)…")
            r = subprocess.run(cmd)
            if r.returncode != 0:
                raise RuntimeError(f"Irodori worker exited {r.returncode}")
            if res_path.exists():
                result = json.loads(res_path.read_text())
                t = result.get("timings", {})
                logger.info(
                    f"Irodori: model_load={t.get('model_load_s', 0):.1f}s  "
                    f"synth={t.get('total_synth_s', 0):.1f}s for {len(items)} sentences"
                )
        finally:
            for p in (req_path, res_path):
                p.unlink(missing_ok=True)


# --------------------------------------------------------------- backend factory

def load_backend(
    backend: str,
    voice_name: str,
    device: Optional[str] = None,
    *,
    num_steps: Optional[int] = None,
    cfg_scale_speaker: Optional[float] = None,
    caption: Optional[str] = None,
):
    voice = find_voice(backend, voice_name)
    if backend == "sbv2":
        return SBV2Backend(voice, device=device)
    if backend == "irodori":
        kw: dict = {}
        if num_steps is not None:           kw["num_steps"] = num_steps
        if cfg_scale_speaker is not None:   kw["cfg_scale_speaker"] = cfg_scale_speaker
        if caption is not None:             kw["caption"] = caption
        return IrodoriBackend(voice, device=device, **kw)
    raise ValueError(f"Unknown backend '{backend}'. Expected 'sbv2' or 'irodori'.")


# --------------------------------------------------------- pipeline orchestration

INTER_SENTENCE_PAUSE_S = 0.25
# 64k mono is the audiobook-industry sweet spot for narration (Audible's
# mid-tier). Transparent for voice with no music. Override via --mp3-bitrate
# on the CLI or SUBPLZ_MP3_BITRATE in the environment.
DEFAULT_MP3_BITRATE = os.environ.get("SUBPLZ_MP3_BITRATE", "64k")


def _atomic_write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    """Write a single mono 16-bit WAV via tmp+rename so a crash never leaves
    a half-written file the resume logic might mistake for complete."""
    import soundfile as sf
    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.dtype == np.int16:
        pcm = audio
    elif np.issubdtype(audio.dtype, np.floating):
        f = np.clip(audio.astype(np.float32), -1.0, 1.0)
        pcm = (f * 32767.0).astype(np.int16)
    else:
        raise RuntimeError(f"Unexpected audio dtype: {audio.dtype}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Explicit format= because the .wav.tmp double-suffix defeats sf's
    # extension-based auto-detection.
    sf.write(tmp, pcm, sr, subtype="PCM_16", format="WAV")
    tmp.replace(path)


def _existing_sentence_indices(work_dir: Path) -> set[int]:
    """Indices we've already synthesized (per-sentence file present on disk)."""
    out: set[int] = set()
    if not work_dir.exists():
        return out
    for f in work_dir.glob("[0-9]" * 6 + ".wav"):
        try:
            out.add(int(f.stem))
        except ValueError:
            pass
    return out


def _sentences_signature(sentences: list[str]) -> str:
    """Short stable hash of the sentence list — used to detect when a resume
    candidate's source text has changed and the work dir is no longer valid."""
    import hashlib
    h = hashlib.md5()
    for s in sentences:
        h.update(s.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def _finalize_to_mp3(
    work_dir: Path, n_sentences: int, mp3_path: Path, bitrate: str = DEFAULT_MP3_BITRATE,
) -> tuple[int, list[tuple[float, float]]]:
    """ffmpeg-concat all per-sentence WAVs (with a shared silence wav between
    them) into one MP3. Returns (sample_rate, per-sentence (start,end) spans)
    derived from the actual per-sentence file durations."""
    import soundfile as sf

    first = work_dir / f"{0:06d}.wav"
    info = sf.info(first)
    sr = int(info.samplerate)

    # Shared silence file referenced once per inter-sentence pause.
    silence_path = work_dir / "silence.wav"
    if not silence_path.exists():
        silence = np.zeros(int(INTER_SENTENCE_PAUSE_S * sr), dtype=np.int16)
        sf.write(silence_path, silence, sr, subtype="PCM_16")

    # Concat list (ffmpeg concat demuxer) + per-sentence cue spans.
    cue_spans: list[tuple[float, float]] = []
    cursor = 0.0
    list_lines: list[str] = []
    for i in range(n_sentences):
        wav = work_dir / f"{i:06d}.wav"
        if not wav.exists():
            raise RuntimeError(f"Cannot finalize: missing sentence {i} at {wav}")
        dur = sf.info(wav).frames / sr
        list_lines.append(f"file '{wav.name}'")
        cue_spans.append((cursor, cursor + dur))
        cursor += dur
        if i < n_sentences - 1:
            list_lines.append(f"file '{silence_path.name}'")
            cursor += INTER_SENTENCE_PAUSE_S
    concat_list = work_dir / "concat.txt"
    concat_list.write_text("\n".join(list_lines), encoding="utf-8")

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-codec:a", "libmp3lame", "-b:a", bitrate, "-ar", str(sr),
        str(mp3_path),
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed: {r.stderr.decode('utf-8', errors='replace')[-500:]}")
    return sr, cue_spans


def _truncate_to_chars(sentences: list[str], max_chars: int) -> list[str]:
    """Take whole sentences from the start until adding the next would exceed
    max_chars. Always returns at least the first sentence if one exists."""
    if max_chars <= 0:
        return sentences
    out: list[str] = []
    total = 0
    for s in sentences:
        if out and total + len(s) > max_chars:
            break
        out.append(s)
        total += len(s)
    return out


def epub_to_audiobook(
    epub_path: Optional[Path],
    backend_name: str,
    voice_name: str,
    out_dir: Path,
    *,
    sentences_override: Optional[list[str]] = None,
    output_stem: Optional[str] = None,
    device: Optional[str] = None,
    max_sentences: Optional[int] = None,
    max_chars: Optional[int] = None,
    lang: str = "ja",
    num_steps: Optional[int] = None,
    cfg_scale_speaker: Optional[float] = None,
    caption: Optional[str] = None,
    mp3_bitrate: Optional[str] = None,
) -> tuple[Path, Path]:
    """epub → (mp3_path, srt_path). Generates continuous audio plus a sentence-level SRT.

    `max_chars` caps how much of the book to synthesize (counting JA chars in
    extracted sentences). `max_sentences` is the older sentence-count cap;
    if both are given, both apply. Pass neither to synthesize the whole book.

    `sentences_override` skips epub parsing entirely (callers like the GUI
    can pre-parse once and reuse across multiple voice runs). In that case
    `epub_path` is only used to derive the output filename; pass `output_stem`
    to override.
    """
    if sentences_override is None and epub_path is None:
        raise ValueError("epub_to_audiobook needs either epub_path or sentences_override")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if sentences_override is not None:
        sentences = list(sentences_override)
        logger.info(f"Using {len(sentences)} pre-parsed sentences (cache hit)")
    else:
        epub_path = Path(epub_path)
        logger.info(f"Extracting sentences from {epub_path.name}…")
        sentences = extract_sentences(epub_path, lang=lang)
    full_count = len(sentences)
    full_chars = sum(len(s) for s in sentences)
    if max_chars:
        sentences = _truncate_to_chars(sentences, max_chars)
    if max_sentences:
        sentences = sentences[:max_sentences]
    used_chars = sum(len(s) for s in sentences)
    logger.info(
        f"  {len(sentences)}/{full_count} sentences  "
        f"{used_chars}/{full_chars} chars"
    )
    if not sentences:
        raise RuntimeError("No sentences extracted from epub.")

    base_stem = output_stem or (epub_path.stem if epub_path else "tts")
    stem = f"{base_stem}.{backend_name}.{voice_name}"
    mp3_path = out_dir / f"{stem}.mp3"
    srt_path = out_dir / f"{stem}.srt"
    # Per-sentence WAVs live here until we ffmpeg-concat into MP3 at the end.
    # Hidden dotfolder so it stays out of finder listings; deleted on success.
    work_dir = out_dir / f".{stem}.subplz-work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Resume sanity: if the work dir was created by an aborted run with a
    # *different* config (different sentences / voice), nuke it. Otherwise
    # the per-sentence files would mismatch the current sentence list.
    meta_path = work_dir / "meta.json"
    config = {
        "backend": backend_name,
        "voice": voice_name,
        "n_sentences": len(sentences),
        "sentences_signature": _sentences_signature(sentences),
        "num_steps": num_steps,
        "cfg_scale_speaker": cfg_scale_speaker,
        "caption": caption,
    }
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text())
        except Exception:
            existing = None
        if existing != config:
            logger.warning(
                "Work dir has a different config; discarding partial output and starting fresh."
            )
            shutil.rmtree(work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))

    # Resume: skip sentence indices already on disk
    completed = _existing_sentence_indices(work_dir)
    remaining: list[tuple[int, str]] = [
        (i, s) for i, s in enumerate(sentences) if i not in completed
    ]
    if completed and remaining:
        logger.info(
            f"Resuming: {len(completed)}/{len(sentences)} sentences already on disk; "
            f"{len(remaining)} to go."
        )
    elif not remaining:
        logger.info(
            f"All {len(sentences)} sentences already synthesized from a previous run — "
            f"finalizing only."
        )

    synth_s = 0.0
    if remaining:
        backend = load_backend(
            backend_name, voice_name, device=device,
            num_steps=num_steps, cfg_scale_speaker=cfg_scale_speaker, caption=caption,
        )
        t0 = time.perf_counter()
        if backend_name == "sbv2":
            # One sentence per call: in-process, atomic write per sentence so
            # Ctrl-C / crash never leaves a half-written file.
            for k, (idx, text) in enumerate(tqdm(remaining, desc="SBV2")):
                results = backend.synthesize_many([text])
                audio, sr_ = results[0]
                _atomic_write_wav(work_dir / f"{idx:06d}.wav", audio, sr_)
                if k % 25 == 24:
                    logger.info(f"[{k+1}/{len(remaining)}] SBV2 progress …")
        elif backend_name == "irodori":
            # One subprocess call for the whole remaining batch; the worker
            # writes per-sentence WAVs with absolute indices, also atomically.
            backend.synthesize_to_dir(remaining, work_dir)
        else:
            raise ValueError(f"Unknown backend {backend_name!r}")
        synth_s = time.perf_counter() - t0

    # Verify all sentences got produced before we try to finalize.
    missing = [i for i in range(len(sentences)) if not (work_dir / f"{i:06d}.wav").exists()]
    if missing:
        raise RuntimeError(
            f"Synthesis incomplete — {len(missing)} sentence(s) missing "
            f"(first missing: {missing[0]}). Rerun to resume from where we left off; "
            f"work dir kept at: {work_dir}"
        )

    sr, spans = _finalize_to_mp3(
        work_dir, len(sentences), mp3_path,
        bitrate=(mp3_bitrate or DEFAULT_MP3_BITRATE),
    )
    cues = [(s, e, text) for (s, e), text in zip(spans, sentences)]
    _write_srt(cues, srt_path)
    audio_s = spans[-1][1] if spans else 0.0
    rtf = audio_s / synth_s if synth_s > 0 else 0.0
    # Same "X.X min audio, RTF Y.YYx" pattern the GUI parses for its stats line.
    logger.info(
        f"Wrote {mp3_path.name} ({audio_s/60:.1f} min audio, RTF {rtf:.2f}x) + {srt_path.name}"
    )
    # Cleanup on success — keeps the user's output dir tidy. On failure we
    # leave the work dir intact so the next run can resume.
    shutil.rmtree(work_dir, ignore_errors=True)
    return mp3_path, srt_path


# ------------------------------------------------------------ run.py entrypoints

def run_tts(args) -> None:
    """`subplz tts` handler."""
    sentences_override = None
    sentences_file = getattr(args, "sentences_file", None)
    if sentences_file:
        # Newline-separated; empty lines skipped. Matches what the GUI writes.
        text = Path(sentences_file).read_text(encoding="utf-8")
        sentences_override = [s for s in (line.strip() for line in text.splitlines()) if s]
    epub_path = Path(args.epub) if getattr(args, "epub", None) else None
    output_stem = getattr(args, "output_stem", None)
    if not epub_path and sentences_override is None:
        raise ValueError("Pass --epub or --sentences-file")
    epub_to_audiobook(
        epub_path=epub_path,
        sentences_override=sentences_override,
        output_stem=output_stem,
        backend_name=args.backend,
        voice_name=args.voice,
        out_dir=Path(args.output_dir),
        device=getattr(args, "device", None) or None,
        max_sentences=getattr(args, "max_sentences", None),
        max_chars=getattr(args, "max_chars", None),
        num_steps=getattr(args, "num_steps", None),
        cfg_scale_speaker=getattr(args, "cfg_scale_speaker", None),
        caption=getattr(args, "caption", None) or None,
        mp3_bitrate=getattr(args, "mp3_bitrate", None),
    )


def run_voice(args) -> None:
    """`subplz voice` handler. Dispatches on `args.voice_op`."""
    op = args.voice_op
    if op == "list":
        for v in list_voices(args.backend):
            print(f"  {v.backend:8s}  {v.name:24s}  {v.path}")
        return
    if op == "install-preset":
        install_sbv2_preset(args.name)
        return
    if op == "clone":
        clone_irodori_voice(
            audio_path=Path(args.audio),
            name=args.name,
            start_s=args.start,
            duration_s=args.duration,
            overwrite=args.overwrite,
        )
        return
    raise ValueError(f"Unknown voice op: {op}")
