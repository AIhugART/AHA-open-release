"""Qwen-Image-2.1 worker (plan P4). Runs inside its own virtual environment (pinned in pyproject.toml).

Speaks the worker protocol of docs/04_architecture.md section 2 on stdin/stdout. Uses only public diffusers /
transformers / bitsandbytes APIs; no code is copied from the Qwen or community repositories (plan R4, R21).

Memory plan: the INT8 model always uses model-level CPU offload (the full BF16 model is not offered, plan R25).
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
from types import MethodType

inbox: queue.Queue[dict] = queue.Queue()
cancelled: set[str] = set()
RGBA_PREFIX = "This is an RGBA image with transparency. "
RGBA_SUFFIX = " The image has alpha channel and the background is transparent."


class Cancelled(Exception):
    pass


STARTED = time.monotonic()


def install_crash_diagnostics() -> None:
    """A native crash (access violation, segfault) skips every `except` and leaves the gateway only an exit code.

    faulthandler writes the Python stack of every thread to stderr - the worker log - at that moment, so the log says
    which line was running. Passive: it only fires when the process is already dying.
    """
    faulthandler.enable(file=sys.stderr, all_threads=True)


def stage(label: str) -> None:
    """One log line with elapsed time and memory, so a crash shows how far loading got and how much RAM was left.

    Never carries a prompt, a token or a path from a request; only a fixed label, seconds and gibibytes.
    """
    import psutil

    rss = psutil.Process().memory_info().rss / 2**30
    available = psutil.virtual_memory().available / 2**30
    print(f"[worker +{time.monotonic() - STARTED:6.1f}s rss={rss:.1f}GiB ram_available={available:.1f}GiB] {label}", file=sys.stderr, flush=True)


def emit(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def stdin_lines():
    """Yield protocol lines from fd 0 WITHOUT ever blocking inside a read.

    Root cause (plan R22): on Windows a thread parked in a synchronous ReadFile on the stdin pipe blocks every other
    operation on that handle, including the ones `import torch` / `import numpy` perform, so the worker hung forever on
    its first load. Polling for available bytes first means no thread is ever parked in ReadFile.
    """
    fd, pending = sys.stdin.fileno(), b""
    if sys.platform == "win32":
        import ctypes
        import msvcrt
        from ctypes import wintypes

        handle = wintypes.HANDLE(msvcrt.get_osfhandle(fd))
        peek, avail = ctypes.windll.kernel32.PeekNamedPipe, wintypes.DWORD()

        def available() -> int | None:  # None = the gateway closed the pipe
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


# ---------------------------------------------------------------------------------------------- loading


def keep_int8_state_on_device(model) -> None:
    """bitsandbytes 0.50.2: a parent `.to()` moves the INT8 weight but not the scale/outlier tensors kept beside it.

    Wrap each Linear8bitLt's `_apply` so those tensors follow the weight; instance-local, nothing installed is modified.
    """
    import torch
    from bitsandbytes.nn import Linear8bitLt

    for layer in model.modules():
        if not isinstance(layer, Linear8bitLt) or getattr(layer, "_aha_patched", False):
            continue
        original = layer._apply

        def apply(self, fn, recurse=True, _original=original):
            result = _original(fn, recurse=recurse)
            device = self.weight.device
            if self.weight.CB is not None:
                self.weight.CB = self.weight.data
            if self.weight.SCB is not None:
                self.weight.SCB = self.weight.SCB.to(device)
            for name, value in vars(self.state).items():
                if isinstance(value, torch.Tensor):
                    setattr(self.state, name, self.weight.data if name == "CB" else value.to(device))
            return result

        layer._apply = MethodType(apply, layer)
        layer._aha_patched = True


def load_pipeline(variant: str, weights: str, gpu: int):
    import torch
    from diffusers import QwenImage21Pipeline

    if variant != "int8":
        raise ValueError(f"unsupported variant {variant!r}: only int8 is offered")
    from diffusers import QwenImage21Transformer2DModel
    from transformers import Qwen3VLForConditionalGeneration

    path = str(weights)
    components = {}
    for name, cls in (("transformer", QwenImage21Transformer2DModel), ("text_encoder", Qwen3VLForConditionalGeneration)):
        stage(f"load {name}: reading weights")
        part = cls.from_pretrained(path, subfolder=name, dtype=torch.bfloat16, local_files_only=True)
        stage(f"load {name}: weights read")
        if not getattr(part, "is_loaded_in_8bit", False):
            raise RuntimeError(f"{name} in {path} is not a saved bitsandbytes INT8 component")
        keep_int8_state_on_device(part)
        part.to("cpu")  # one component at a time, so the load never peaks on the GPU
        torch.cuda.empty_cache()
        stage(f"load {name}: on cpu")
        components[name] = part
    stage("load pipeline: assembling")
    pipe = QwenImage21Pipeline.from_pretrained(path, dtype=torch.bfloat16, local_files_only=True, **components)
    for name in ("text_encoder", "transformer"):
        keep_int8_state_on_device(getattr(pipe, name))
    stage("load pipeline: enabling model cpu offload")
    pipe.enable_model_cpu_offload(gpu_id=gpu)
    stage("load pipeline: ready")
    return pipe, "model_cpu_offload"


# ---------------------------------------------------------------------------------------------- LoRA

MAX_RESIDENT_ADAPTERS = 6
adapters: dict[str, str] = {}  # adapter name -> sha256 of the file it was loaded from (insertion order = age)


LORA_PREFIXES = ("diffusion_model.", "transformer.", "base_model.model.")


def lora_state_for(path: str, module_names: set[str]) -> dict:
    """Read a LoRA file into the layout diffusers loads, or refuse - never apply part of it silently.

    Trained LoRAs name the same layers differently: a `diffusion_model.`/`transformer.` prefix, a `.default` adapter
    infix, `.alpha` tensors, and ai-toolkit's fused `img_mlp.gate_up` where this model has two layers (`gate_layer`,
    then `proj`; the fused rows are the two halves in that order - an independent open-source Qwen-Image LoRA loader, a third-party
    statement that the real-run check in docs/05_models.md section 3.2 tests). Measured on a real ai-toolkit file:
    without this step peft dropped the 64 fused tensors with only a log line and the LoRA ran at five sixths strength.
    """
    from safetensors.torch import load_file

    raw = load_file(path)
    pairs: dict[str, dict[str, object]] = {}
    for key, tensor in raw.items():
        for prefix in LORA_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        base, _, leaf = key.rpartition(".")
        if leaf == "alpha":
            pairs.setdefault(base, {})["alpha"] = tensor
            continue
        for marker in ("lora_A", "lora_B"):
            if f".{marker}" in key:
                pairs.setdefault(key.split(f".{marker}")[0], {})[marker] = tensor
                break
        else:
            raise ValueError(f"this LoRA has a tensor named {key!r} that is not a PEFT/ai-toolkit lora_A/lora_B weight")
    state: dict[str, object] = {}
    for module, parts in pairs.items():
        if "lora_A" not in parts or "lora_B" not in parts:
            raise ValueError(f"this LoRA has an incomplete pair for {module!r}")
        a, b = parts["lora_A"], parts["lora_B"]
        if "alpha" in parts:
            b = b * (float(parts["alpha"]) / a.shape[0])
        if module.endswith(".img_mlp.gate_up"):
            prefix, half = module[: -len("gate_up")], b.shape[0] // 2
            targets = {prefix + "gate_layer": b[:half], prefix + "proj": b[half:]}
        else:
            targets = {module: b}
        for name, b_part in targets.items():
            if name not in module_names:
                raise ValueError(f"this LoRA targets {name!r}, which this model does not have; it was trained for another model")
            state[f"transformer.{name}.lora_A.weight"] = a
            state[f"transformer.{name}.lora_B.weight"] = b_part
    return state


def apply_loras(pipe, loras: list[dict]) -> None:
    """Make the pipeline use exactly this job's LoRAs, and nothing left over from the previous job.

    Adapters stay loaded between jobs (loading costs seconds); a changed file under the same id is reloaded. When a job
    uses none, they are switched off, so a job without LoRAs is bit-identical to a server that never loaded one.
    """
    if not loras:
        if adapters:
            pipe.disable_lora()
        return
    for item in loras:
        name = item["id"]
        if adapters.get(name) == item["sha256"]:
            continue
        if name in adapters:
            pipe.delete_adapters(name)
            del adapters[name]
        while len(adapters) >= MAX_RESIDENT_ADAPTERS:
            oldest = next(iter(adapters))
            pipe.delete_adapters(oldest)
            del adapters[oldest]
        state = lora_state_for(item["path"], {n for n, _ in pipe.transformer.named_modules()})  # safetensors only: no pickle
        pipe.load_lora_weights(state, adapter_name=name)
        adapters[name] = item["sha256"]
    pipe.enable_lora()
    pipe.set_adapters([i["id"] for i in loras], adapter_weights=[i["scale"] for i in loras])


# ---------------------------------------------------------------------------------------------- one job


def run(pipe, kv_cache_flag: bool, msg: dict) -> None:
    import torch
    from PIL import Image

    job_id, jobdir = msg["job_id"], Path(msg["dir"])
    request = json.loads((jobdir / "request.json").read_text(encoding="utf-8"))
    params, prompt = request["params"], request.get("prompt") or ""
    if params.get("transparent"):
        prompt = f"{RGBA_PREFIX}{prompt}{RGBA_SUFFIX}"
    images = []
    for role in ("source", "reference"):  # the source first: the pipeline reads the last image's aspect only when no size is given
        for item in request.get("inputs", []):
            if item["role"] == role:
                with Image.open(item["path"]) as img:
                    images.append(img.convert("RGB"))
    total = int(params["steps"])
    apply_loras(pipe, request.get("loras", []))

    def on_step(_pipe, index, _timestep, kwargs):
        if job_id in cancelled:
            raise Cancelled()
        emit({"event": "progress", "job_id": job_id, "step": index + 1, "total_steps": total})
        return kwargs

    outputs = []
    for n in range(int(params.get("num_outputs", 1))):
        generator = torch.Generator("cpu").manual_seed(int(params["seed"]) + n)
        args = {
            "prompt": prompt, "width": params["width"], "height": params["height"], "num_inference_steps": total,
            "generator": generator, "callback_on_step_end": on_step, "true_cfg_scale": 1.0, "use_kv_cache": kv_cache_flag,
        }
        if images:
            args["image"] = images
        result = pipe(**args).images[0]
        name = f"out_{n}.png"
        result.save(jobdir / name)
        outputs.append({"file": name, "kind": "image", "media_type": "image/png", "mode": result.mode})
    emit({"event": "done", "job_id": job_id, "outputs": outputs})


def main() -> None:
    install_crash_diagnostics()
    threading.Thread(target=reader, daemon=True).start()
    pipe = None
    kv_cache = True
    while True:
        msg = inbox.get()
        op = msg.get("op")
        if op == "load":
            started = time.monotonic()
            try:
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError("no CUDA device is available to this worker")
                gpu = int(msg.get("gpu") or 0)
                torch.cuda.set_device(gpu)
                pipe, mode = load_pipeline(msg.get("variant") or "int8", msg["weights_dir"], gpu)
                kv_cache = True
                emit({
                    "event": "loaded", "model": msg.get("model"), "variant": msg.get("variant"), "placement": mode,
                    "load_s": round(time.monotonic() - started, 1),
                    "vram_peak_mib": int(torch.cuda.max_memory_allocated() / 2**20),
                })
            except Exception as err:  # noqa: BLE001 - report to the gateway instead of dying silently (a mute worker is undiagnosable)
                traceback.print_exc(file=sys.stderr)
                code = "gpu_out_of_memory" if "out of memory" in str(err).lower() else "generation_failed"
                emit({"event": "failed", "code": code, "message": f"loading the model failed: {type(err).__name__}: {err}"[:400]})
                return
        elif op == "run":
            job_id = msg["job_id"]
            try:
                run(pipe, kv_cache, msg)
            except Cancelled:
                emit({"event": "cancelled", "job_id": job_id})
            except Exception as err:  # noqa: BLE001 - any pipeline failure becomes a typed `failed` event for the client
                traceback.print_exc(file=sys.stderr)
                import torch

                oom = isinstance(err, torch.OutOfMemoryError) or "out of memory" in str(err).lower()
                emit({
                    "event": "failed", "job_id": job_id, "code": "gpu_out_of_memory" if oom else "generation_failed",
                    "message": ("the GPU ran out of memory; try a smaller size or the int8 variant" if oom else f"{type(err).__name__}: {err}"[:400]),
                })
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
