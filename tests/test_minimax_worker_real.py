"""The MiniMax H3 worker over its real protocol, with the real model (plan 2026-11-24_minimax_h3_plan.md, M2).

Needs the weights downloaded under reference/h3_weights and the worker environment, and takes minutes: marked ``gpu`` (skipped by
default, run with ``pytest -m gpu tests/test_minimax_worker_real.py``). Each test starts the worker exactly as the gateway does
(``python worker.py`` with pipes) and talks JSON lines, so what is checked is the protocol the gateway depends on:
``loaded`` -> ``progress`` ... -> ``done`` (or ``cancelled`` / ``failed``), and that a cancelled job leaves no video behind.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from aha_open.catalog import CATALOG

ROOT = Path(__file__).resolve().parents[1]
WORKER_DIR = ROOT / "workers" / "minimax_h3"
PYTHON = WORKER_DIR / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
REF = ROOT / "reference"
SRC = {
    "abiray/MiniMax_H3_FL2VA_pruned_mixed_int4_int8_convrot.safetensors": REF / "h3_weights/abiray/MiniMax_H3_FL2VA_pruned_mixed_int4_int8_convrot.safetensors",
    "abiray/vae/minimax_h3_video_vae_fp16.safetensors": REF / "Minimax-H3-nvfp4-INT4-INT8-Convrot/vae/minimax_h3_video_vae_fp16.safetensors",
    "abiray/vae/minimax_h3_audio_vae_fp32.safetensors": REF / "Minimax-H3-nvfp4-INT4-INT8-Convrot/vae/minimax_h3_audio_vae_fp32.safetensors",
    "comfyorg/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors": REF / "h3_weights/comfyorg/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
}
FL2VA = CATALOG["minimax-h3-fl2va"]
# the other builds' transformers, when downloaded: each gets the same job test
OTHER = {FL2VA.variants[v]["files"][0]: REF / "h3_weights" / FL2VA.variants[v]["files"][0] for v in ("int8", "nvfp4")}
OFFICIAL = REF / "h3_weights" / "official"

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not PYTHON.exists(), reason="the minimax_h3 worker environment is not installed"),
    pytest.mark.skipif(not all(p.exists() for p in SRC.values()) or not OFFICIAL.exists(), reason="the MiniMax weights are not downloaded"),
]

PARAMS = {"width": 384, "height": 224, "num_frames": 124, "seed": 42}  # 384x224: small, so the test is about the protocol, not speed


@pytest.fixture(scope="module")
def weights_root():
    """The folder layout `aha-open models install` writes, built from the files already on disk with hard links (no copy of 46 GB).

    Lives next to the weights (same drive, which a hard link needs); removed afterwards.
    """
    root = REF / "h3_weights" / "_pytest_root"
    if root.exists():
        shutil.rmtree(root)
    for rel, src in {**SRC, **{r: p for r, p in OTHER.items() if p.exists()}}.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(src, target)
    shutil.copytree(OFFICIAL, root, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".cache"))
    yield root
    shutil.rmtree(root, ignore_errors=True)


class Worker:
    def __init__(self, log: Path):
        self.log = open(log, "wb")
        self.proc = subprocess.Popen(
            [str(PYTHON), str(WORKER_DIR / "worker.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, bufsize=1, cwd=str(WORKER_DIR),
        )
        self.lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        for line in self.proc.stdout:
            try:
                self.lines.put(json.loads(line))
            except json.JSONDecodeError:
                self.lines.put({"event": "garbage", "line": line})
        self.lines.put({"event": "exited"})

    def send(self, message: dict) -> None:
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def next(self, timeout: float = 900) -> dict:
        return self.lines.get(timeout=timeout)

    def until(self, *names: str, timeout: float = 900) -> list[dict]:
        seen = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            event = self.next(max(1.0, deadline - time.monotonic()))
            seen.append(event)
            if event["event"] in names:
                return seen
            if event["event"] == "exited":
                break
        raise AssertionError(f"waited for {names}, saw {[e['event'] for e in seen][-6:]}")

    def stop(self) -> None:
        try:
            self.send({"op": "unload"})
            self.proc.wait(timeout=60)
        except Exception:  # noqa: BLE001 - the worker may already be gone
            self.proc.kill()
        self.log.close()


def new_job(tmp_path: Path, name: str, steps: int, prompt: str = "a lighthouse on a cliff at sunset, waves crashing on the rocks") -> tuple[str, Path]:
    jobdir = tmp_path / name
    jobdir.mkdir()
    (jobdir / "request.json").write_text(json.dumps({"prompt": prompt, "params": {**PARAMS, "steps": steps}, "inputs": []}), encoding="utf-8")
    return name, jobdir


@pytest.fixture()
def worker(request, tmp_path, weights_root):
    variant = getattr(request, "param", "mixed")
    files = list(FL2VA.variants[variant]["files"])
    if not (weights_root / files[0]).exists():
        pytest.skip(f"the {variant} transformer is not downloaded")
    w = Worker(tmp_path / "worker.log")
    w.send({"op": "load", "model": "minimax-h3-fl2va", "variant": variant, "weights_dir": str(weights_root), "gpu": 0, "files": files})
    loaded = w.until("loaded", "failed")[-1]
    assert loaded["event"] == "loaded", loaded
    yield w
    w.stop()


@pytest.mark.parametrize("worker", ["mixed", "int8", "nvfp4"], indirect=True)
def test_a_job_reports_progress_and_returns_a_video_that_matches_what_was_accepted(worker, tmp_path):
    job_id, jobdir = new_job(tmp_path, "job1", steps=4)
    worker.send({"op": "run", "job_id": job_id, "dir": str(jobdir), "task": "video.generate"})
    events = worker.until("done", "failed", "cancelled")
    last = events[-1]
    assert last["event"] == "done", last
    progress = [e for e in events if e["event"] == "progress"]
    assert progress and progress[-1]["step"] == progress[-1]["total_steps"] and all(e["job_id"] == job_id for e in progress)
    [out] = last["outputs"]
    assert out["kind"] == "video" and out["media_type"] == "video/mp4" and out["has_audio"] is True
    assert (out["width"], out["height"], out["frames"], out["fps"]) == (384, 224, 124, 24.0)
    assert (jobdir / out["file"]).stat().st_size > 10_000
    # what the worker says is in the file is what is in the file (read back independently of the worker's own probe)
    done = subprocess.run(
        [str(PYTHON), "-c", "import av,sys;c=av.open(sys.argv[1]);v=c.streams.video[0];a=c.streams.audio[0];"
         "print(v.codec_context.width,v.codec_context.height,sum(1 for _ in c.decode(video=0)),a.codec_context.sample_rate,"
         "a.codec_context.channels,c.metadata.get('comment',''))", str(jobdir / out["file"])],
        capture_output=True, text=True, timeout=120,
    )
    assert done.stdout.strip().split(maxsplit=5) == ["384", "224", "124", "32000", "2", "AI-generated (MiniMax H3)"], done.stderr
    log = (tmp_path / "worker.log").read_text(encoding="utf-8", errors="replace")
    assert "lighthouse" not in log  # the prompt is never written to the worker's log


def test_cancel_stops_the_job_at_the_next_step_and_leaves_no_video(worker, tmp_path):
    job_id, jobdir = new_job(tmp_path, "job2", steps=20)
    worker.send({"op": "run", "job_id": job_id, "dir": str(jobdir), "task": "video.generate"})
    first = worker.until("progress", "done", "failed")[-1]
    assert first["event"] == "progress", first
    started = time.monotonic()
    worker.send({"op": "cancel", "job_id": job_id})
    last = worker.until("cancelled", "done", "failed", timeout=300)[-1]
    assert last["event"] == "cancelled" and last["job_id"] == job_id, last
    assert time.monotonic() - started < 120  # at the next sampling step, not after the whole job
    assert not list(jobdir.glob("*.mp4"))  # a cancelled job never has a result
    # and the worker is still usable: nothing is stuck in the cancel set
    job_id2, jobdir2 = new_job(tmp_path, "job3", steps=2)
    worker.send({"op": "run", "job_id": job_id2, "dir": str(jobdir2), "task": "video.generate"})
    assert worker.until("done", "failed", "cancelled")[-1]["event"] == "done"


def test_missing_files_are_reported_by_name_with_what_to_do_not_as_a_crash(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    w = Worker(tmp_path / "worker.log")
    try:
        w.send({"op": "load", "model": "minimax-h3-fl2va", "variant": "mixed", "weights_dir": str(empty), "gpu": 0,
            "files": list(CATALOG["minimax-h3-fl2va"].variants["mixed"]["files"])})
        event = w.until("failed", "loaded")[-1]
    finally:
        w.stop()
    assert event["event"] == "failed" and event["code"] == "model_not_installed"
    assert "MiniMax_H3_FL2VA_pruned_mixed_int4_int8_convrot.safetensors" in event["message"] and "models install" in event["message"]
    assert str(tmp_path) not in event["message"]  # no server path travels to the client


def test_a_request_with_input_images_is_refused_clearly_until_they_are_supported(worker, tmp_path):
    jobdir = tmp_path / "job4"
    jobdir.mkdir()
    (jobdir / "request.json").write_text(json.dumps({
        "prompt": "x", "params": {**PARAMS, "steps": 2}, "inputs": [{"role": "first_frame", "path": str(jobdir / "a.png")}],
    }), encoding="utf-8")
    worker.send({"op": "run", "job_id": "job4", "dir": str(jobdir), "task": "video.generate"})
    event = worker.until("failed", "done", "cancelled")[-1]
    assert event["event"] == "failed" and event["code"] == "generation_failed" and "text only" in event["message"]


# ---------------------------------------------------------------------------------------------- Ref2VA (plan M7)

REF2VA = CATALOG["minimax-h3-ref2va"]
# the first reference build that is downloaded (they differ only in the transformer)
REF_VARIANT = next((v for v in ("mixed", "nvfp4") if (REF / "h3_weights" / REF2VA.variants[v]["files"][0]).exists()), "mixed")
REF2VA_FILE = REF / "h3_weights" / REF2VA.variants[REF_VARIANT]["files"][0]
ref2va_only = pytest.mark.skipif(not REF2VA_FILE.exists() or not (OFFICIAL / "transformer_ref" / "config.json").exists(),
                                 reason="the Ref2VA transformer is not downloaded")


def solid_png(path: Path, rgb: tuple[int, int, int], width: int = 256, height: int = 160) -> Path:
    """A plain-colour PNG made with the standard library only."""
    import struct
    import zlib

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    row = b"\x00" + bytes(rgb) * width
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(row * height)) + chunk(b"IEND", b""))
    return path


@pytest.fixture(scope="module")
def ref_weights_root(weights_root):
    """The text model's folder plus what the reference model adds: its transformer and the transformer_ref config."""
    link = weights_root / "abiray" / REF2VA_FILE.name
    os.link(REF2VA_FILE, link)
    (weights_root / "transformer_ref").mkdir(exist_ok=True)
    shutil.copy2(OFFICIAL / "transformer_ref" / "config.json", weights_root / "transformer_ref" / "config.json")
    yield weights_root


@pytest.fixture()
def ref_worker(tmp_path, ref_weights_root):
    w = Worker(tmp_path / "worker.log")
    w.send({"op": "load", "model": "minimax-h3-ref2va", "variant": REF_VARIANT, "weights_dir": str(ref_weights_root), "gpu": 0,
            "files": list(REF2VA.variants[REF_VARIANT]["files"])})
    loaded = w.until("loaded", "failed")[-1]
    assert loaded["event"] == "loaded", loaded
    yield w
    w.stop()


def ref_job(tmp_path: Path, name: str, inputs: list[dict], steps: int = 4) -> tuple[str, Path]:
    jobdir = tmp_path / name
    jobdir.mkdir()
    (jobdir / "request.json").write_text(json.dumps({
        "prompt": "the red object slowly turns towards the camera", "params": {**PARAMS, "steps": steps}, "inputs": inputs}), encoding="utf-8")
    return name, jobdir


@ref2va_only
def test_a_reference_image_job_returns_a_video_of_the_accepted_size_with_sound(ref_worker, tmp_path):
    image = solid_png(tmp_path / "ref.png", (220, 30, 30))
    job_id, jobdir = ref_job(tmp_path, "ref1", [{"role": "reference", "path": str(image)}])
    ref_worker.send({"op": "run", "job_id": job_id, "dir": str(jobdir), "task": "video.generate"})
    events = ref_worker.until("done", "failed", "cancelled", timeout=1800)
    last = events[-1]
    assert last["event"] == "done", last
    [out] = last["outputs"]
    assert (out["width"], out["height"], out["frames"], out["fps"], out["has_audio"]) == (384, 224, 124, 24.0, True)
    assert [e for e in events if e["event"] == "progress"][-1]["step"] >= 1
    log = (tmp_path / "worker.log").read_text(encoding="utf-8", errors="replace")
    assert str(image) not in log and "turns towards" not in log  # neither a path nor the prompt reaches the worker log


@ref2va_only
def test_the_reference_model_refuses_a_job_without_references(ref_worker, tmp_path):
    job_id, jobdir = ref_job(tmp_path, "ref2", [])
    ref_worker.send({"op": "run", "job_id": job_id, "dir": str(jobdir), "task": "video.generate"})
    event = ref_worker.until("failed", "done", "cancelled")[-1]
    assert event["event"] == "failed" and "at least one reference" in event["message"]
