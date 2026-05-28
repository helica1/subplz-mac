"""Irodori inference worker — runs inside the Irodori venv (transformers<5),
spawned by IrodoriBackend in the main subplz process.

Protocol: read a JSON request from --request-json, write a JSON result to
--result-json. Loads the model once, generates audio for every sentence,
saves each as a WAV.

Request (two shapes accepted):
    Legacy (sentences as a list of strings, indexed 0..N-1):
        {"...": "...", "sentences": ["...", "..."]}
    Indexed (list of {"index": N, "text": "..."} — supports resume with
    arbitrary indices and atomic per-sentence write):
        {"...": "...", "items": [{"index": 17, "text": "..."}, ...]}

    Other fields:
      irodori_repo, hf_checkpoint, ref_wav, device, output_dir,
      num_steps, cfg_scale_speaker, caption, seed

Result:
    {
      "items": [{"index": 17, "text": "...", "wav": "...", "duration_s": 7.3}, ...],
      "sample_rate": 44100,
      "timings": {"model_load_s": ..., "total_synth_s": ...}
    }
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--request-json", required=True)
    ap.add_argument("--result-json", required=True)
    args = ap.parse_args()

    req = json.loads(Path(args.request_json).read_text())
    sys.path.insert(0, req["irodori_repo"])

    from huggingface_hub import hf_hub_download
    from irodori_tts.inference_runtime import (
        InferenceRuntime, RuntimeKey, SamplingRequest, save_wav,
    )

    t0 = time.perf_counter()
    checkpoint_path = hf_hub_download(repo_id=req["hf_checkpoint"], filename="model.safetensors")
    runtime = InferenceRuntime.from_key(RuntimeKey(
        checkpoint=str(checkpoint_path),
        model_device=req["device"],
        codec_device=req["device"],
    ))
    model_load_s = time.perf_counter() - t0
    sample_rate = int(runtime.codec.sample_rate)

    out_dir = Path(req["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    num_steps = int(req.get("num_steps", 24))
    cfg_scale_speaker = float(req.get("cfg_scale_speaker", 5.0))
    caption = req.get("caption") or None
    seed = req.get("seed")  # None for random

    # Normalize input to indexed shape so the rest of the worker has one path.
    if "items" in req:
        work = [(int(it["index"]), str(it["text"])) for it in req["items"]]
    else:
        work = [(i, t) for i, t in enumerate(req["sentences"])]
    n = len(work)

    items = []
    t_synth = time.perf_counter()
    for k, (idx, text) in enumerate(work):
        result = runtime.synthesize(SamplingRequest(
            text=text,
            caption=caption,
            ref_wav=req["ref_wav"],
            num_steps=num_steps,
            cfg_scale_speaker=cfg_scale_speaker,
            seed=None if seed is None else int(seed),
        ))
        # Atomic per-sentence write: write to a `_pending_` prefix (still .wav
        # so torchcodec/soundfile recognize the extension) then rename. The
        # parent's resume globber matches "[0-9]{6}.wav" so it ignores any
        # leftover pending files from a previous crash.
        final_path = out_dir / f"{idx:06d}.wav"
        tmp_path = out_dir / f"_pending_{idx:06d}.wav"
        save_wav(str(tmp_path), result.audio, result.sample_rate)
        tmp_path.replace(final_path)
        dur = float(result.audio.shape[-1]) / float(result.sample_rate)
        items.append({"index": idx, "text": text, "wav": str(final_path), "duration_s": dur})
        elapsed = time.perf_counter() - t_synth
        rate = (k + 1) / elapsed
        eta_s = (n - k - 1) / rate if rate > 0 else 0
        print(f"[{k+1}/{n}] idx={idx} {dur:.1f}s audio  eta {eta_s/60:.1f}min  {text[:40]}", flush=True)
    total_synth_s = time.perf_counter() - t_synth

    Path(args.result_json).write_text(json.dumps({
        "items": items,
        "sample_rate": sample_rate,
        "timings": {"model_load_s": model_load_s, "total_synth_s": total_synth_s},
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
