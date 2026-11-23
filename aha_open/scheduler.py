"""One lane per GPU: a fair queue, one resident model at a time, and the job state machine (docs/04_architecture.md section 3).

Design decisions: lanes are plain threads talking to worker processes through WorkerProc, not asyncio
subprocesses - asyncio subprocess support depends on the event-loop flavour uvicorn picks on Windows.
A failed job is never retried here: a retry costs GPU minutes, so the client decides.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .catalog import CATALOG, WORKERS_DIR, ModelSpec, weights_dir, worker_command
from .config import Config, ModelConfig
from .errors import ApiError
from .store import ACTIVE, TERMINAL, Store
from .util import now, png_size, sha256_file
from .worker_proc import WorkerProc, describe_exit

log = logging.getLogger("aha_open.scheduler")

WORKER_ERROR_CODES = {"gpu_out_of_memory", "generation_failed", "user_cancelled", "model_not_installed"}


class LaneError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class Entry:
    job_id: str
    client_id: str
    model_id: str
    enqueued: float


class Lane:
    def __init__(self, gpu: int) -> None:
        self.gpu = gpu
        self.cond = threading.Condition()
        self.queue: list[Entry] = []
        self.worker: WorkerProc | None = None
        self.model: str | None = None
        self.loading: str | None = None
        self.running: str | None = None
        self.degraded = False
        self.last_served: dict[str, float] = {}
        self.thread: threading.Thread | None = None


class Scheduler:
    def __init__(self, cfg: Config, store: Store) -> None:
        self.cfg = cfg
        self.store = store
        self.lanes: dict[int, Lane] = {}
        self.admission_lock = threading.Lock()  # check-limits-then-insert must be one step
        self._lanes_lock = threading.Lock()
        self._cancel: set[str] = set()
        self._cancel_lock = threading.Lock()
        self._stop = threading.Event()
        self._janitor: threading.Thread | None = None
        self.started = False

    # ------------------------------------------------------------------ paths
    def job_dir(self, job_id: str) -> Path:
        return self.cfg.data_dir / "jobs" / job_id

    def upload_dir(self, upload_id: str) -> Path:
        return self.cfg.data_dir / "uploads" / upload_id

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self._recover()
        self._janitor = threading.Thread(target=self._janitor_loop, name="janitor", daemon=True)
        self._janitor.start()
        self.started = True

    def stop(self) -> None:
        self._stop.set()
        for lane in list(self.lanes.values()):
            with lane.cond:
                lane.cond.notify_all()
        for lane in list(self.lanes.values()):
            if lane.thread:
                lane.thread.join(timeout=10)
            if lane.worker:
                lane.worker.stop()
        self.started = False

    def _recover(self) -> None:
        """Jobs that were alive when the last process died cannot continue; fail them, never re-run them silently."""
        marks = ",".join("?" for _ in ACTIVE)
        for row in self.store.jobs.all(f"SELECT id FROM jobs WHERE status IN ({marks})", ACTIVE):
            self.store.finish_job(
                row["id"], "failed", now(), error={"code": "generation_failed", "message": "the server restarted while this job was waiting or running"}
            )
            shutil.rmtree(self.job_dir(row["id"]), ignore_errors=True)

    def _lane(self, gpu: int) -> Lane:
        with self._lanes_lock:
            lane = self.lanes.get(gpu)
            if lane is None:
                lane = Lane(gpu)
                lane.thread = threading.Thread(target=self._lane_loop, args=(lane,), name=f"lane-{gpu}", daemon=True)
                self.lanes[gpu] = lane
                lane.thread.start()
            return lane

    # ------------------------------------------------------------------ API-facing
    def submit(self, job_id: str, client_id: str, mc: ModelConfig) -> None:
        lane = self._lane(mc.gpu)
        with lane.cond:
            lane.queue.append(Entry(job_id, client_id, mc.id, time.monotonic()))
            lane.cond.notify()

    def queue_position(self, job_id: str) -> int | None:
        for lane in self.lanes.values():
            with lane.cond:
                for index, entry in enumerate(lane.queue):
                    if entry.job_id == job_id:
                        return index + 1
        return None

    def cancel(self, job_id: str) -> None:
        """Queued: gone at once. Loading a model: terminal at once, the lane drops it afterwards. Running: flagged, stops at the next step."""
        for lane in self.lanes.values():
            with lane.cond:
                for entry in lane.queue:
                    if entry.job_id == job_id:
                        lane.queue.remove(entry)
                        self._finish_cancelled(job_id)
                        return
        job = self.store.job(job_id)
        if job is None or job["status"] in TERMINAL:
            return
        with self._cancel_lock:
            self._cancel.add(job_id)
        if job["status"] == "loading_model":
            self._finish_cancelled(job_id)

    def health(self) -> dict:
        lanes = list(self.lanes.values())
        queued = sum(len(lane.queue) for lane in lanes)
        running = sum(1 for lane in lanes if lane.running)
        return {
            "active": {lane.gpu: lane.model for lane in lanes},
            "loading": {lane.loading for lane in lanes if lane.loading},
            "degraded": any(lane.degraded for lane in lanes),
            "queued": queued,
            "running": running,
        }

    # ------------------------------------------------------------------ lane loop
    def _lane_loop(self, lane: Lane) -> None:
        while not self._stop.is_set():
            with lane.cond:
                while not lane.queue and not self._stop.is_set():
                    lane.cond.wait(timeout=0.5)
                if self._stop.is_set():
                    return
                entry = self._pick(lane)
                lane.queue.remove(entry)
                lane.last_served[entry.client_id] = time.monotonic()
            try:
                self._run_entry(lane, entry)
            except Exception:  # a bug here must not kill the lane or leave the job hanging
                log.exception("lane %s: unexpected error on %s", lane.gpu, entry.job_id)
                self._fail(entry.job_id, "generation_failed", "internal error in the scheduler")
                self._drop_worker(lane)

    def _pick(self, lane: Lane) -> Entry:
        """Round-robin over clients (least recently served first); stay on the resident model unless someone waited too long."""
        first: dict[str, Entry] = {}
        for entry in lane.queue:
            first.setdefault(entry.client_id, entry)
        order = sorted(first.values(), key=lambda e: lane.last_served.get(e.client_id, 0.0))
        t = time.monotonic()
        starved = [e for e in order if e.model_id != lane.model and t - e.enqueued > self.cfg.max_wait_for_swap_s]
        if starved:
            return min(starved, key=lambda e: e.enqueued)
        if lane.model:
            for entry in order:
                if entry.model_id == lane.model:
                    return entry
        return order[0]

    def _cancelled(self, job_id: str) -> bool:
        with self._cancel_lock:
            return job_id in self._cancel

    def _forget(self, job_id: str) -> None:
        with self._cancel_lock:
            self._cancel.discard(job_id)

    # ------------------------------------------------------------------ one job
    def _run_entry(self, lane: Lane, entry: Entry) -> None:
        job_id = entry.job_id
        job = self.store.job(job_id)
        if job is None or job["status"] in TERMINAL:
            return
        if self._cancelled(job_id):
            self._finish_cancelled(job_id)
            return
        mc = self.cfg.model(entry.model_id)
        spec = CATALOG.get(entry.model_id)
        jobdir = self.job_dir(job_id)
        if mc is None or spec is None or not (jobdir / "request.json").exists():
            self._fail(job_id, "generation_failed", "the job's request is no longer available")
            return
        try:
            if not (lane.worker and lane.worker.alive() and lane.model == mc.id):
                if not self.store.advance_job(job_id, "loading_model"):
                    return
                self._ensure_model(lane, mc, spec)
            if not self.store.advance_job(job_id, "running", step=0, total_steps=0):
                return  # cancelled while the model loaded
            request = json.loads((jobdir / "request.json").read_text(encoding="utf-8"))
            lane.running = job_id
            self._drive(lane, job_id, jobdir, job["task"], request.get("params", {}), request.get("loras", []))
        except LaneError as err:
            self._fail(job_id, err.code, err.message)
        finally:
            lane.running = None
            self._forget(job_id)
            (jobdir / "request.json").unlink(missing_ok=True)  # the prompt never outlives the job
            if (self.store.job(job_id) or {}).get("status") != "succeeded":
                shutil.rmtree(jobdir, ignore_errors=True)

    def _drive(self, lane: Lane, job_id: str, jobdir: Path, task: str, accepted: dict, loras: list[dict]) -> None:
        worker = lane.worker
        assert worker is not None
        if not worker.send({"op": "run", "job_id": job_id, "dir": str(jobdir), "task": task}):
            self._drop_worker(lane)
            raise LaneError("generation_failed", "the worker is not reachable")
        deadline = time.monotonic() + self.cfg.job_timeout_s
        cancel_sent: float | None = None
        while True:
            event = worker.next_event(0.2)
            t = time.monotonic()
            if cancel_sent is None and self._cancelled(job_id):
                worker.send({"op": "cancel", "job_id": job_id})
                cancel_sent = t
            if cancel_sent is not None and t - cancel_sent > self.cfg.cancel_grace_s:
                self._drop_worker(lane)  # it ignored the cancel; the GPU must be given back
                self._finish_cancelled(job_id)
                return
            if event is None:
                if t > deadline:
                    self._drop_worker(lane)
                    raise LaneError("generation_failed", "the job exceeded job_timeout_s")
                continue
            if event.get("job_id") not in (None, job_id):
                continue  # late event of an earlier job
            kind = event.get("event")
            if kind == "progress":
                self.store.jobs.run(
                    "UPDATE jobs SET step=?, total_steps=? WHERE id=? AND status='running'",
                    (int(event.get("step", 0)), int(event.get("total_steps", 0)), job_id),
                )
            elif kind == "done":
                if self._cancelled(job_id):
                    self._finish_cancelled(job_id)  # a result that arrives after cancel is dropped
                    return
                results = self._collect(jobdir, event, accepted, loras)
                if not self.store.finish_job(job_id, "succeeded", now(), results=results):
                    shutil.rmtree(jobdir, ignore_errors=True)
                else:
                    log.info("job %s succeeded", job_id)
                return
            elif kind in ("failed", "cancelled"):
                code = event.get("code")
                if kind == "cancelled" or code == "user_cancelled" or self._cancelled(job_id):
                    self._finish_cancelled(job_id)
                else:
                    self._fail(job_id, code if code in WORKER_ERROR_CODES else "generation_failed", str(event.get("message", "generation failed")))
                return
            elif kind == "exited":
                lane.worker, lane.model, lane.degraded = None, None, True
                if self._cancelled(job_id):
                    self._finish_cancelled(job_id)
                    return
                raise LaneError("generation_failed", f"while generating: {describe_exit(event.get('code'), worker.log_name)}")

    def _collect(self, jobdir: Path, event: dict, accepted: dict, loras: list[dict]) -> list[dict]:
        outputs = event.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            raise LaneError("generation_failed", "the worker finished without producing an output")
        results = []
        base = jobdir.resolve()
        for index, out in enumerate(outputs):
            path = (jobdir / str(out.get("file", ""))).resolve()
            if base not in path.parents or not path.is_file():
                raise LaneError("generation_failed", "the worker reported an output file that does not exist")
            kind = out.get("kind", "image")
            item = {
                "index": index, "kind": kind, "media_type": out.get("media_type", "image/png"),
                "bytes": path.stat().st_size, "sha256": sha256_file(path), "file": path.name,
            }
            if "seed" in accepted:
                item["seed"] = accepted["seed"] + index  # what the worker seeded this output with
            if loras:  # which exact adapter files shaped this image (never their paths)
                item["loras"] = [{"id": x["id"], "scale": x["scale"], "sha256": x["sha256"]} for x in loras]
            if kind == "image":
                size = png_size(path)
                want = (accepted.get("width"), accepted.get("height"))
                if size is None:
                    raise LaneError("generation_failed", "the worker produced an image that is not a PNG")
                if want != (None, None) and size != want:
                    # Invariant: output size is exactly the accepted size; never resize silently (docs/04 section 8).
                    raise LaneError("generation_failed", f"output is {size[0]}x{size[1]} but {want[0]}x{want[1]} was accepted")
                item["width"], item["height"] = size
            else:
                for key in ("width", "height", "duration_s", "fps", "has_audio"):
                    if key in out:
                        item[key] = out[key]
            results.append(item)
        return results

    # ------------------------------------------------------------------ model residency
    def _drop_worker(self, lane: Lane) -> None:
        worker, lane.worker, lane.model = lane.worker, None, None
        if worker:
            worker.kill()

    def _ensure_model(self, lane: Lane, mc: ModelConfig, spec: ModelSpec) -> None:
        if lane.worker:
            lane.worker.stop()  # one model on the GPU at a time: release the VRAM before loading the next
            lane.worker, lane.model = None, None
        lane.loading = mc.id
        try:
            try:
                cmd = worker_command(self.cfg, mc)
            except ApiError as err:
                raise LaneError("model_not_installed", err.message) from err
            wdir = weights_dir(self.cfg, mc)
            worker = WorkerProc(cmd, self.cfg.data_dir / "logs" / f"worker-{mc.worker}.log", cwd=WORKERS_DIR / mc.worker)
            worker.send({"op": "load", "model": mc.id, "variant": mc.variant, "weights_dir": str(wdir) if wdir else None, "gpu": mc.gpu})
            deadline = time.monotonic() + self.cfg.worker_load_timeout_s
            while True:
                event = worker.next_event(0.5)
                if event is None:
                    if time.monotonic() > deadline:
                        worker.kill()
                        raise LaneError("generation_failed", "loading the model timed out")
                    continue
                kind = event.get("event")
                if kind == "loaded":
                    lane.worker, lane.model, lane.degraded = worker, mc.id, False
                    log.info("lane %s: %s loaded", lane.gpu, mc.id)
                    return
                if kind == "failed":
                    worker.kill()
                    code = event.get("code")
                    raise LaneError(code if code in WORKER_ERROR_CODES else "generation_failed", str(event.get("message", "loading the model failed")))
                if kind == "exited":
                    lane.degraded = True
                    raise LaneError("generation_failed", f"while loading the model: {describe_exit(event.get('code'), worker.log_name)}")
        finally:
            lane.loading = None

    # ------------------------------------------------------------------ terminal states
    def _finish_cancelled(self, job_id: str) -> None:
        if self.store.finish_job(job_id, "cancelled", now(), error={"code": "user_cancelled", "message": "cancelled"}):
            log.info("job %s cancelled", job_id)
        self._forget(job_id)
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)

    def _fail(self, job_id: str, code: str, message: str) -> None:
        if self.store.finish_job(job_id, "failed", now(), error={"code": code, "message": message}):
            log.info("job %s failed: %s", job_id, code)

    # ------------------------------------------------------------------ janitor
    def _janitor_loop(self) -> None:
        while not self._stop.wait(20.0):
            try:
                self.sweep()
            except Exception:
                log.exception("janitor sweep failed")

    def sweep(self) -> None:
        t = now()
        for row in self.store.expired_uploads(t - self.cfg.upload_ttl_s):
            shutil.rmtree(self.upload_dir(row["id"]), ignore_errors=True)
            self.store.delete_upload(row["id"])
        for row in self.store.expirable_jobs(t - self.cfg.result_ttl_s):
            shutil.rmtree(self.job_dir(row["id"]), ignore_errors=True)
            self.store.set_job(row["id"], expired=1)
        self.store.clients.run("UPDATE enrolls SET status='expired', pending_token=NULL WHERE status='pending' AND expires_at<=?", (t,))
        self.store.clients.run("UPDATE enrolls SET pending_token=NULL WHERE status='approved' AND expires_at<=?", (t - 3600,))
