"""Synthetic worker: no torch, no GPU, stdlib only. It speaks the real worker protocol (docs/04_architecture.md section 2)
so the gateway and the conformance suite can be exercised end to end without a model.

Prompt markers (tests only) make it misbehave on purpose:
  __fail_oom  gpu_out_of_memory     __fail  generation_failed     __slow  200 steps (time to cancel)
  __wrongsize PNG (or video) of the wrong size   __nosize  a sized video that does not report its size
  __ignorecancel  never honours cancel     __crash  dies mid-job
  __crash_native  dies the way a real native crash does (access violation / SIGSEGV)
Environment (tests only): AHA_FAKE_CRASH_ON_LOAD=native  dies natively while loading the model.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import struct
import sys
import threading
import time
import zlib
from pathlib import Path

STEP_S = float(os.environ.get("AHA_FAKE_STEP_S", "0.02"))
LOAD_S = float(os.environ.get("AHA_FAKE_LOAD_S", "0.3"))

inbox: queue.Queue[dict] = queue.Queue()
cancelled: set[str] = set()


def emit(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def die_natively() -> None:
    """Die like a real native crash: no Python exception, no `failed` event, only an exit status."""
    if sys.platform == "win32":
        os._exit(-1073741819)  # STATUS_ACCESS_VIOLATION 0xC0000005 as the signed 32-bit int _exit takes; the parent sees 3221225477
    os.kill(os.getpid(), signal.SIGSEGV)


def reader() -> None:
    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("op") == "cancel":
            cancelled.add(msg.get("job_id", ""))
        else:
            inbox.put(msg)
    inbox.put({"op": "unload"})  # stdin closed: the gateway is gone


def png(width: int, height: int, seed: int) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    r, g, b = seed % 251, (seed * 7) % 251, (seed * 13) % 251
    row = b"\x00" + bytes((r, g, b)) * width
    raw = row * height
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def fake_mp4(seed: int) -> bytes:
    return struct.pack(">I", 24) + b"ftypisom" + struct.pack(">I", 512) + b"isomiso2" + bytes([seed % 251]) * 4096


def run(msg: dict) -> None:
    job_id, jobdir = msg["job_id"], Path(msg["dir"])
    request = json.loads((jobdir / "request.json").read_text(encoding="utf-8"))
    prompt, params = request.get("prompt") or "", request["params"]
    for item in request.get("inputs", []):
        if not Path(item["path"]).is_file():
            emit({"event": "failed", "job_id": job_id, "code": "generation_failed", "message": "an input file is missing"})
            return
    loras = request.get("loras", [])
    for lora in loras:
        if not Path(lora["path"]).is_file():
            emit({"event": "failed", "job_id": job_id, "code": "generation_failed", "message": "a LoRA file is missing"})
            return
    if "__crash_native" in prompt:
        die_natively()
    if "__crash" in prompt:
        os._exit(3)
    if "__fail_oom" in prompt:
        emit({"event": "failed", "job_id": job_id, "code": "gpu_out_of_memory", "message": "simulated out of memory"})
        return
    if "__fail" in prompt:
        emit({"event": "failed", "job_id": job_id, "code": "generation_failed", "message": "simulated failure"})
        return
    total = 200 if "__slow" in prompt else int(params.get("steps", 4))
    for step in range(1, total + 1):
        time.sleep(STEP_S)
        if job_id in cancelled and "__ignorecancel" not in prompt:
            emit({"event": "cancelled", "job_id": job_id})
            return
        emit({"event": "progress", "job_id": job_id, "step": step, "total_steps": total})
    seed = int(params.get("seed", 0)) + sum(int(l["scale"] * 10) + len(l["id"]) for l in loras)  # a LoRA changes the picture
    outputs = []
    for index in range(int(params.get("num_outputs", 1))):
        if msg["task"].startswith("video"):
            name = f"out_{index}.mp4"
            (jobdir / name).write_bytes(fake_mp4(seed + index))
            video = {"file": name, "kind": "video", "media_type": "video/mp4", "duration_s": params.get("duration_s", 1), "fps": 8, "has_audio": False}
            if "width" in params and "__nosize" not in prompt:  # a model with exact sizes says what it made, like the MiniMax worker
                video["width"] = params["width"] + (1 if "__wrongsize" in prompt else 0)
                video["height"] = params["height"]
            outputs.append(video)
        else:
            w, h = params["width"], params["height"]
            if "__wrongsize" in prompt:
                w += 1
            name = f"out_{index}.png"
            (jobdir / name).write_bytes(png(w, h, seed + index))
            outputs.append({"file": name, "kind": "image", "media_type": "image/png"})
    emit({"event": "done", "job_id": job_id, "outputs": outputs})


def main() -> None:
    threading.Thread(target=reader, daemon=True).start()
    while True:
        msg = inbox.get()
        op = msg.get("op")
        if op == "load":
            time.sleep(LOAD_S)
            if os.environ.get("AHA_FAKE_CRASH_ON_LOAD") == "native":
                die_natively()
            emit({"event": "loaded", "model": msg.get("model"), "vram_peak_mib": 0})
        elif op == "run":
            run(msg)
        elif op == "unload":
            emit({"event": "unloaded"})
            return


if __name__ == "__main__":
    main()
