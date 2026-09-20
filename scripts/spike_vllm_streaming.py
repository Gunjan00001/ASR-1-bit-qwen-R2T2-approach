"""T0.0 go/no-go spike: Qwen3-ASR vLLM streaming on a CUDA host (target: Kaggle T4).

This is a throwaway verification script, not part of the library. It answers one
question: can we run the *official* `qwen-asr[vllm]` streaming path on the target
GPU host? If yes, Stage 0's streaming gate is feasible there; if no, the gate
moves to Colab Pro (A100/L4) or the `qwenllm/qwen3-asr` Docker image.

See ``docs/kaggle_stage0.md`` for the Kaggle setup that runs this.

Usage (on a CUDA host with ``qwen-asr[vllm]`` installed)::

    python scripts/spike_vllm_streaming.py --model Qwen/Qwen3-ASR-0.6B

Exit codes: 0 pass, 2 no CUDA, 3 vLLM/model init failed, 4 empty streaming output.
"""

from __future__ import annotations

import argparse
import io
import platform
import sys
import time
import urllib.request

SAMPLE_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
TARGET_SAMPLE_RATE = 16000
MIN_CAPABILITY = 7  # vLLM nightly (cu129) needs sm70+; T4 is sm75, P100 is sm60.


def env_report() -> list[str]:
    """Collect version and device facts to record alongside the spike result."""
    lines = [f"python={platform.python_version()}", f"platform={platform.platform()}"]
    try:
        import torch

        lines.append(f"torch={torch.__version__}")
        lines.append(f"torch_cuda={torch.version.cuda}")
        lines.append(f"cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            lines.append(f"device={torch.cuda.get_device_name(0)}")
            lines.append(f"compute_capability=sm{cap[0]}{cap[1]}")
    except Exception as exc:  # noqa: BLE001 - spike reports any import failure
        lines.append(f"torch import failed: {exc!r}")
    for name in ("vllm", "qwen_asr", "transformers"):
        try:
            module = __import__(name)
            lines.append(f"{name}={getattr(module, '__version__', 'unknown')}")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"{name} import failed: {exc!r}")
    return lines


def gpu_check() -> tuple[bool, list[str]]:
    """Return (ok, messages). Fails hard when there is no usable CUDA device."""
    messages: list[str] = []
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return False, [f"torch is not importable: {exc!r}"]
    if not torch.cuda.is_available():
        messages.append("No CUDA device visible. vLLM streaming cannot run here.")
        return False, messages
    capability = torch.cuda.get_device_capability(0)
    if capability[0] < MIN_CAPABILITY:
        messages.append(
            f"GPU sm{capability[0]}{capability[1]} is below the vLLM nightly "
            f"minimum sm{MIN_CAPABILITY}. Use a T4/L4/A100, not P100."
        )
        return False, messages
    messages.append(f"CUDA device OK: {torch.cuda.get_device_name(0)}.")
    return True, messages


def load_wav_16k(url: str, max_seconds: float | None) -> object:
    """Download a wav and return 16 kHz mono float32 numpy audio."""
    import numpy as np
    import soundfile as sf
    import soxr

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    with io.BytesIO(payload) as handle:
        audio, sr = sf.read(handle, dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != TARGET_SAMPLE_RATE:
        audio = soxr.resample(audio, sr, TARGET_SAMPLE_RATE).astype(np.float32)
    if max_seconds is not None:
        audio = audio[: int(max_seconds * TARGET_SAMPLE_RATE)]
    return audio


def run_streaming(asr, audio, chunk_size_sec: float, step_sec: float) -> dict:
    """Feed audio in `step_sec` increments with the given model chunk size."""
    step = max(1, round(step_sec * TARGET_SAMPLE_RATE))
    state = asr.init_streaming_state(
        unfixed_chunk_num=2,
        unfixed_token_num=5,
        chunk_size_sec=chunk_size_sec,
    )
    started = time.perf_counter()
    position = 0
    calls = 0
    while position < audio.shape[0]:
        segment = audio[position : position + step]
        position += segment.shape[0]
        calls += 1
        asr.streaming_transcribe(segment, state)
        print(f"  [call {calls:03d}] language={state.language!r} text={state.text!r}")
    asr.finish_streaming_transcribe(state)
    elapsed = time.perf_counter() - started
    duration = audio.shape[0] / TARGET_SAMPLE_RATE
    print(f"  [final] language={state.language!r} text={state.text!r}")
    return {
        "chunk_size_sec": chunk_size_sec,
        "calls": calls,
        "text": state.text,
        "elapsed_sec": round(elapsed, 2),
        "audio_sec": round(duration, 2),
        "rtf": round(elapsed / duration, 3) if duration > 0 else float("inf"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        help="Cap KV cache; the 65536 default does not fit a T4 (~7 GiB).",
    )
    parser.add_argument("--sample-url", default=SAMPLE_URL)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--step-sec", type=float, default=0.5)
    parser.add_argument(
        "--chunk-sizes",
        type=float,
        nargs="+",
        default=[2.0, 0.32],
        help="model chunk_size_sec values to exercise",
    )
    args = parser.parse_args()

    print("=== environment ===")
    for line in env_report():
        print(" ", line)

    print("=== gpu check ===")
    ok, messages = gpu_check()
    for line in messages:
        print(" ", line)
    if not ok:
        print("SPIKE: FAIL (no usable GPU)")
        return 2

    print("=== loading model (vLLM backend) ===")
    try:
        from qwen_asr import Qwen3ASRModel

        asr = Qwen3ASRModel.LLM(
            model=args.model,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_new_tokens=32,
            max_model_len=args.max_model_len,
        )
    except Exception as exc:  # noqa: BLE001 - spike must surface the reason
        import traceback

        traceback.print_exc()
        print(f"SPIKE: FAIL (could not init vLLM backend): {exc!r}")
        return 3

    print("=== loading audio ===")
    audio = load_wav_16k(args.sample_url, args.max_seconds)
    print(f"  samples={audio.shape[0]} ({audio.shape[0] / TARGET_SAMPLE_RATE:.2f}s)")

    print("=== offline sanity ===")
    offline = asr.transcribe(audio=(audio, TARGET_SAMPLE_RATE), language=None)
    print(f"  offline text={offline[0].text!r}")

    results = []
    for chunk_size in args.chunk_sizes:
        print(f"=== streaming chunk_size_sec={chunk_size} ===")
        results.append(run_streaming(asr, audio, chunk_size, args.step_sec))

    print("=== summary ===")
    for result in results:
        print(" ", result)

    if not results[-1]["text"].strip():
        print("SPIKE: FAIL (streaming produced empty text)")
        return 4
    print("SPIKE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
