"""MiniMax H3 worker (text -> video with sound). Runs inside its own virtual environment (pinned in pyproject.toml).

Speaks the worker protocol of docs/04_architecture.md section 2 on stdin/stdout, like the Qwen worker. Loads the community ConvRot
files with this folder's own loader (no ComfyUI): see docs/04_architecture.md section 6.2.

Memory plan: nothing stays on the GPU between jobs. Each job encodes the prompt (encoder layers read from disk), loads the
transformer for the sampling steps, then decodes; the GPU is capped at the memory physically free when the job starts.
"""

from __future__ import annotations

import faulthandler
import json
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the loader modules next to this file

inbox: queue.Queue[dict] = queue.Queue()
cancelled: set[str] = set()
STARTED = time.monotonic()

# file names inside the model's weights folder (aha_open/install.py writes this layout). The transformer file of the chosen build
# comes from the gateway with `load` (the catalog is the one list of builds); the encoder, both VAEs and the small configs are the
# same for every build and both models.
MODELS = ("minimax-h3-fl2va", "minimax-h3-ref2va")
TRANSFORMER_CONFIG = {"minimax-h3-fl2va": "transformer/config.json", "minimax-h3-ref2va": "transformer_ref/config.json"}
TEXT_ENCODER = "comfyorg/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
VIDEO_VAE = "abiray/vae/minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "abiray/vae/minimax_h3_audio_vae_fp32.safetensors"
FPS = 24
REQUIRED_PARAMS = ("width", "height", "num_frames", "steps", "seed")  # the gateway sends complete params; nothing is guessed here


def install_crash_diagnostics() -> None:
    """A native crash skips every `except`; faulthandler writes every thread's Python stack to the worker log at that moment."""
    faulthandler.enable(file=sys.stderr, all_threads=True)


def stage(label: str) -> None:
    """One log line with elapsed time and memory. Only fixed labels and numbers: never a prompt, a token or a request path."""
    import psutil

    rss = psutil.Process().memory_info().rss / 2**30
    available = psutil.virtual_memory().available / 2**30
    print(f"[worker +{time.monotonic() - STARTED:6.1f}s rss={rss:.1f}GiB ram_available={available:.1f}GiB] {label}", file=sys.stderr, flush=True)


def emit(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def stdin_lines():
    """Protocol lines from fd 0 without ever parking a thread in a blocking read (on Windows that blocks `import torch`)."""
    fd, pending = sys.stdin.fileno(), b""
    if sys.platform == "win32":
        import ctypes
        import msvcrt
        from ctypes import wintypes

        handle = wintypes.HANDLE(msvcrt.get_osfhandle(fd))
        peek, avail = ctypes.windll.kernel32.PeekNamedPipe, wintypes.DWORD()

        def available() -> int | None:
            if not peek(handle, None, 0, None, ctypes.byref(avail), None):
                return None
            return avail.value
    else:
        import select

        def available() -> int | None:
            ready, _, _ = select.select([fd], [], [], 0.1)
            return 1 if ready else 0

    while True:
        count = available()
        if count is None:
            return
        if count == 0:
            time.sleep(0.05)
            continue
        chunk = os.read(fd, min(count, 65536))
        if not chunk:
            return
        pending += chunk
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            yield line.decode("utf-8", "replace")


def reader() -> None:
    for line in stdin_lines():
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("op") == "cancel":
            cancelled.add(msg.get("job_id", ""))
        else:
            inbox.put(msg)
    inbox.put({"op": "unload"})


def weights_for(weights_dir: str, files: list[str], model: str = "minimax-h3-fl2va"):
    import h3_runner

    root = Path(weights_dir)
    if model not in MODELS:
        raise FileNotFoundError(f"model {model!r} is not one this worker knows")
    if len(files) != 1:
        raise FileNotFoundError(f"the load request names no transformer file for {model}; update the gateway and the worker together")
    weights = h3_runner.Weights(
        transformer=root / files[0], text_encoder=root / TEXT_ENCODER, video_vae=root / VIDEO_VAE,
        audio_vae=root / AUDIO_VAE, config_dir=root, transformer_config=TRANSFORMER_CONFIG[model],
        accepts_references=model == "minimax-h3-ref2va",
    )
    missing = [p.name for p in (weights.transformer, weights.text_encoder, weights.video_vae, weights.audio_vae,
                                root / "transformer" / "config.json", root / weights.transformer_config) if not p.is_file()]
    if missing:
        raise FileNotFoundError("model files are missing: " + ", ".join(missing) + "; run `aha-open models install` again")
    return weights


def run(weights, msg: dict) -> None:
    import h3_references
    import h3_runner
    import h3_video

    job_id, jobdir = msg["job_id"], Path(msg["dir"])
    request = json.loads((jobdir / "request.json").read_text(encoding="utf-8"))
    params, prompt = request["params"], request.get("prompt") or ""
    absent = [k for k in REQUIRED_PARAMS if k not in params]
    if absent:
        raise ValueError(f"the accepted params lack {', '.join(absent)}")
    inputs = request.get("inputs") or []
    if inputs and not weights.accepts_references:
        raise ValueError("this model generates from text only; first/last frames are not supported yet")
    if weights.accepts_references and not inputs:
        raise ValueError("the reference model needs at least one reference")
    references = h3_references.from_inputs(inputs) if inputs else None
    steps = int(params["steps"])

    result = h3_runner.generate(
        weights, prompt, int(params["width"]), int(params["height"]), int(params["num_frames"]), steps, int(params["seed"]),
        cancelled=lambda: job_id in cancelled,
        on_stage=lambda s: stage(f"job: {s}"),
        on_step=lambda i, n: emit({"event": "progress", "job_id": job_id, "step": i, "total_steps": n}),
        log=lambda line: stage(line),
        references=references,
    )
    if job_id in cancelled:
        raise h3_runner.Cancelled()
    frames, audio, rate = result["frames"], result["audio"], result["sampling_rate"]
    if frames.shape[1:3] != (int(params["height"]), int(params["width"])) or frames.shape[0] != int(params["num_frames"]):
        raise RuntimeError(f"the pipeline produced {frames.shape[0]} frames of {frames.shape[2]}x{frames.shape[1]}, "
                           f"not the accepted {params['num_frames']} of {params['width']}x{params['height']}")
    name = "out_0.mp4"
    h3_video.write_mp4(jobdir / name, frames, audio, rate, fps=FPS)
    info = h3_video.probe(jobdir / name)  # report what the file holds, checked against what was accepted
    if (info["width"], info["height"], info["frames"]) != (int(params["width"]), int(params["height"]), int(params["num_frames"])):
        raise RuntimeError(f"the written mp4 holds {info['frames']} frames of {info['width']}x{info['height']}")
    if info["audio_channels"] != 2:
        raise RuntimeError("the written mp4 has no stereo sound track")
    stage(f"job: done {result['timings']}")
    emit({"event": "done", "job_id": job_id, "outputs": [{
        "file": name, "kind": "video", "media_type": "video/mp4", "width": info["width"], "height": info["height"],
        "duration_s": info["duration_s"], "fps": info["fps"], "frames": info["frames"], "has_audio": True,
    }]})


def _failure(err: BaseException) -> tuple[str, str]:
    """Different causes, different sentences: the client (and the person behind it) can act on each one."""
    import torch

    text = str(err)
    if isinstance(err, torch.OutOfMemoryError) or "out of memory" in text.lower():
        return "gpu_out_of_memory", ("the GPU ran out of memory: close other programs that use the GPU (games, other AI tools) "
                                     "or choose a smaller size, then try again")
    if isinstance(err, FileNotFoundError):
        return "model_not_installed", text[:400]
    return "generation_failed", f"{type(err).__name__}: {text}"[:400]


def main() -> None:
    install_crash_diagnostics()
    threading.Thread(target=reader, daemon=True).start()
    weights = None
    while True:
        msg = inbox.get()
        op = msg.get("op")
        if op == "load":
            started = time.monotonic()
            try:
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError("no CUDA device is available to this worker")
                torch.cuda.set_device(int(msg.get("gpu") or 0))
                weights = weights_for(msg["weights_dir"], msg.get("files") or [], msg.get("model") or "minimax-h3-fl2va")
                weights.memory_limit_gib = msg.get("memory_limit_gib")
                stage("load: files present; the model is loaded per job (nothing stays on the GPU between jobs)")
                emit({"event": "loaded", "model": msg.get("model"), "variant": msg.get("variant"), "placement": "per_job",
                      "load_s": round(time.monotonic() - started, 1), "vram_peak_mib": 0})
            except Exception as err:  # noqa: BLE001 - reported to the gateway instead of dying silently
                traceback.print_exc(file=sys.stderr)
                code, message = _failure(err)
                emit({"event": "failed", "code": code, "message": f"loading the model failed: {message}"[:400]})
                return
        elif op == "run":
            import h3_runner

            job_id = msg["job_id"]
            try:
                run(weights, msg)
            except h3_runner.Cancelled:
                emit({"event": "cancelled", "job_id": job_id})
            except Exception as err:  # noqa: BLE001 - any failure becomes a typed `failed` event for the client
                traceback.print_exc(file=sys.stderr)
                code, message = _failure(err)
                emit({"event": "failed", "job_id": job_id, "code": code, "message": message})
            finally:
                cancelled.discard(job_id)
                import gc

                import torch

                gc.collect()
                torch.cuda.empty_cache()
        elif op == "unload":
            emit({"event": "unloaded"})
            return


if __name__ == "__main__":
    main()
