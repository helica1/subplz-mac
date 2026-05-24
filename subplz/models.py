import gc
from faster_whisper import WhisperModel
import whisper
from types import MethodType
import torch
from copy import copy
import numpy as np

from .logger import logger
from .utils import get_tqdm, find_and_show_lingering_tensors

tqdm = get_tqdm()[0]


def unload_model(model):
    """
    Explicitly unloads a model from VRAM to free up memory.
    """
    if model is None:
        return
    try:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("Model successfully unloaded from memory.")
    except Exception as e:
        logger.warning(f"An error occurred while unloading the model: {e}")
    # if torch.cuda.is_available():
    #     logger.debug("--- VRAM DIAGNOSTICS (Post-Job) ---")
    #     allocated_gb = torch.cuda.memory_allocated() / 1e9
    #     reserved_gb = torch.cuda.memory_reserved() / 1e9
    #     logger.debug(f"torch.cuda.memory_allocated(): {allocated_gb:.2f} GB")
    #     logger.debug(f"torch.cuda.memory_reserved(): {reserved_gb:.2f} GB")

    #     if allocated_gb > 0.1: # If more than 100MB is still actively allocated
    #             logger.warning("High VRAM allocation detected after cleanup. A tensor reference is likely lingering.")
    #     else:
    #             logger.debug("VRAM allocation is low, as expected.")

    #     if reserved_gb > 3.0: # If more than 3GB is still reserved (adjust based on your baseline)
    #         logger.warning(f"High VRAM reservation ({reserved_gb:.2f} GB) persists. This confirms a lingering reference or context issue.")

    #     logger.debug("------------------------------------")
    #     find_and_show_lingering_tensors()


def faster_transcribe(self, audio, name, **args):
    args["log_prob_threshold"] = args.pop("logprob_threshold")
    args["beam_size"] = args["beam_size"] if args["beam_size"] else 1
    args["patience"] = args["patience"] if args["patience"] else 1
    args["length_penalty"] = args["length_penalty"] if args["length_penalty"] else 1
    result = self.transcribe(audio, best_of=1, **args)
    # result = self.refine(audio, result, **args)
    result.pad(0.5, 0.5, word_level=False)
    segments, prev_end = [], 0
    with tqdm(total=result.duration, unit_scale=True, unit=" seconds") as pbar:
        pbar.set_description(f"{name}")
        for segment in result.segments:
            segments.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                }
            )
            pbar.update(segment.end - prev_end)
            prev_end = segment.end
        pbar.update(result.duration - prev_end)
        pbar.refresh()

    return {
        "segments": segments,
        "language": args["language"] if "language" in args else result.language,
    }


def get_temperature(inputs):
    temperature = copy(inputs.temperature)
    temperature_increment_on_fallback = copy(inputs.temperature_increment_on_fallback)
    if (temperature_increment_on_fallback) is not None:
        temperature = tuple(
            np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
        )
    else:
        temperature = [temperature]
    return temperature


MLX_MODEL_ALIASES = {
    "tiny": "mlx-community/whisper-tiny",
    "tiny.en": "mlx-community/whisper-tiny.en",
    "base": "mlx-community/whisper-base",
    "base.en": "mlx-community/whisper-base.en",
    "small": "mlx-community/whisper-small",
    "small.en": "mlx-community/whisper-small.en",
    "medium": "mlx-community/whisper-medium",
    "medium.en": "mlx-community/whisper-medium.en",
    "large": "mlx-community/whisper-large-v3-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}

# kwargs that mlx_whisper.transcribe accepts. Anything else from the subplz
# CLI is dropped silently. (faster-whisper exposes more knobs than MLX does.)
_MLX_SUPPORTED_KWARGS = frozenset(
    {
        "language",
        "initial_prompt",
        "temperature",
        "compression_ratio_threshold",
        "logprob_threshold",
        "no_speech_threshold",
        "condition_on_previous_text",
        "word_timestamps",
        "prepend_punctuations",
        "append_punctuations",
    }
)


class MLXWhisperBackend:
    """Apple Silicon Metal + Neural Engine backend.

    Wraps `mlx_whisper.transcribe()` to match the `faster_transcribe(audio,
    name, **kwargs)` contract that subplz/files.py:AudioSub.transcribe expects.
    MLX caches the loaded model internally across calls keyed on the repo path,
    so sequential chapter transcriptions reuse the loaded weights.
    """

    def __init__(self, model_repo: str):
        self.model_repo = model_repo
        # Eagerly resolve to give the user a clear error early on a typo'd repo.
        import mlx_whisper  # noqa: F401  (defer heavyweight import to call time)

        logger.info(f"🧠 MLX-Whisper backend: model='{model_repo}'")

    def faster_transcribe(self, audio, name, **kwargs):
        import mlx_whisper

        # Strip kwargs MLX doesn't understand; map a few names through.
        mlx_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in _MLX_SUPPORTED_KWARGS and v is not None
        }
        # MLX requires temperature to be a float or tuple of floats; subplz
        # may pass a list/tuple already from get_temperature(). Normalize.
        if "temperature" in mlx_kwargs:
            t = mlx_kwargs["temperature"]
            if not isinstance(t, (int, float)):
                mlx_kwargs["temperature"] = tuple(float(x) for x in t)

        logger.info(f"📝 MLX transcribe: {name}")
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model_repo,
            verbose=None,
            **mlx_kwargs,
        )

        # Reshape to the contract subplz expects: only {start, end, text} per
        # segment. mlx_whisper segments also include `id`, `tokens`,
        # `avg_logprob`, etc. — extra fields are harmless but we trim to keep
        # the cache file small and match the existing pipeline shape.
        segments = [
            {"start": float(s["start"]), "end": float(s["end"]), "text": s["text"]}
            for s in result.get("segments", [])
        ]
        return {
            "segments": segments,
            "language": result.get("language", kwargs.get("language") or "en"),
        }


def get_model(backend):
    model_name = backend.model_name
    device = backend.device
    faster_whisper = backend.faster_whisper
    stable_ts = backend.stable_ts
    local_files_only = backend.local_only
    quantize = backend.quantize
    num_workers = backend.threads
    # MLX is opt-in via --mlx; on Apple Silicon it's the fastest option since
    # faster-whisper/CTranslate2 has no Metal backend.
    use_mlx = getattr(backend, "mlx", False)

    if use_mlx:
        # Map short aliases like "turbo" → an mlx-community HF repo.
        # If the user passed a full repo path with "/", trust it as-is.
        if "/" in model_name:
            model_repo = model_name
        else:
            model_repo = MLX_MODEL_ALIASES.get(model_name, model_name)
            if model_repo == model_name and model_name not in MLX_MODEL_ALIASES:
                logger.warning(
                    f"⚠️  '{model_name}' not in MLX alias table; will try as HF repo. "
                    f"Known: {sorted(MLX_MODEL_ALIASES.keys())}"
                )
        return MLXWhisperBackend(model_repo)

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    logger.info(
        f"🖥️  We're using {device}. Results will be faster using Cuda with GPU than just CPU. Lots of RAM needed no matter what."
    )
    compute_type = (
        "float32" if not quantize else ("int8" if device == "cpu" else "float16")
    )
    if faster_whisper:
        model = WhisperModel(
            model_name, device, local_files_only, compute_type, num_workers
        )
        model.transcribe2 = model.transcribe
        model.faster_transcribe = MethodType(faster_transcribe, model)
    elif stable_ts:
        import stable_whisper

        model = stable_whisper.load_faster_whisper(
            model_name,
            device=device,
            local_files_only=local_files_only,
            compute_type=compute_type,
            num_workers=num_workers,
        )
        # model.transcribe2 = model.transcribe_stable
        # model.transcribe2 = model.transcribe
        # TODO: Don't monkeypatch this - unnecessary
        model.faster_transcribe = MethodType(faster_transcribe, model)

    else:
        model = whisper.load_model(model).to(device)
        if quantize and device != "cpu":
            model = model.half()
    return model
