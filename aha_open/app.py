"""The /v1 HTTP surface (docs/03_api_v1.md). Gateway only: this module never imports torch.

Order of defence for every request: Host allow-list -> Origin rule -> bearer token -> ownership of the object.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import secrets
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .auth import Auth
from .catalog import CATALOG, detect_gpus, enabled_models, manifest_model, model_state, validate_job
from .config import Config
from .errors import ApiError, busy, forbidden, invalid, not_found
from .install import INSTALLABLE
from .loras import ID_RE as LORA_ID_RE
from .loras import MAX_FILE_BYTES as LORA_MAX_BYTES
from .loras import LoraError, check_meta
from .loras import add as add_lora
from .loras import remove as remove_lora
from .loras import scan as scan_loras
from .model_install import InstallManager
from .report import build_report
from .scheduler import Scheduler
from .store import TERMINAL, Store
from .util import iso, new_id, now, sha256_file

log = logging.getLogger("aha_open.api")
API_VERSION = "1.0"
BODY = Body(...)  # module-level singleton: FastAPI reads it as the JSON body marker
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
REQUEST_ID_WINDOW_S = 600
ROLE_MEDIA = {
    "source": "image/", "reference": "image/", "mask": "image/", "first_frame": "image/", "last_frame": "image/",
    "reference_video": "video/", "reference_audio": "audio/",
}
MAX_OPEN_UPLOADS_PER_CLIENT = 32


def _strip_port(host: str) -> str:
    host = host.strip().lower()
    if host.startswith("["):
        return host[1:].split("]")[0]
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


def allowed_hosts(cfg: Config) -> set[str]:
    hosts = {"127.0.0.1", "localhost", "::1"} if cfg.profile == "local" else set()
    listen = cfg.listen_host
    if listen and listen not in ("0.0.0.0", "::", "*"):
        hosts.add(listen.lower())
    if cfg.profile in ("local", "internet"):
        hosts |= {"127.0.0.1", "localhost", "::1"}
    hosts |= {h.lower() for h in cfg.hosts}
    return hosts


LOOPBACK_PEERS = {"127.0.0.1", "::1"}
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def admin_allowed(profile: str, peer: str | None, host_header: str) -> bool:
    """`/admin/*` is for the person sitting at this machine, nobody else.

    Why a loopback peer is NOT enough: in the `internet` profile cloudflared connects from 127.0.0.1 and passes the public
    request through, so every visitor of the tunnel looks like a loopback peer (conformance/test_v1_security.py even relies
    on that). The page is therefore on only in the `local` profile, and only for a loopback peer asking for a loopback name.
    """
    return profile == "local" and peer in LOOPBACK_PEERS and _strip_port(host_header) in LOOPBACK_HOSTS


def own_inputs(jobdir: Path, resolved: list[dict]) -> None:
    """Give the job its own name for every input file and point the request at it.

    An upload lives by its own clock (``Scheduler.sweep`` deletes it by age); a job may wait in the queue far longer, so it must
    not read the upload's file. A hard link costs no space; a file system without hard links (exFAT) gets a copy.
    """
    for index, item in enumerate(resolved):
        mine = jobdir / f"in_{index}"
        try:
            os.link(item["path"], mine)
        except OSError:
            shutil.copyfile(item["path"], mine)  # a source that is gone raises FileNotFoundError here
        item["path"] = str(mine)


class GuardMiddleware:
    """Host allow-list (DNS rebinding) and Origin rule (hostile web pages), plus no-store on everything (rules 6)."""

    def __init__(self, app, cfg: Config) -> None:
        self.app = app
        self.hosts = allowed_hosts(cfg)
        self.origins = {o.rstrip("/").lower() for o in cfg.allowed_origins}

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        host = headers.get("host", "")
        problem = None
        if _strip_port(host) not in self.hosts:
            problem = "this Host is not served here"
        elif "origin" in headers:
            origin = headers["origin"].rstrip("/").lower()
            same_origin = origin.split("://", 1)[-1] == host.lower()
            if origin not in self.origins and not same_origin:
                problem = "requests from this Origin are not allowed"
        if problem:
            response = JSONResponse(forbidden(problem).body(), status_code=403)
            return await response(scope, receive, send)

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                extra = [(b"cache-control", b"no-store"), (b"x-content-type-options", b"nosniff")]
                message = {**message, "headers": [*message.get("headers", []), *extra]}
            await send(message)

        return await self.app(scope, receive, send_with_headers)


def create_app(cfg: Config, installer: InstallManager | None = None) -> FastAPI:
    store = Store(cfg.data_dir)
    auth = Auth(cfg, store)
    scheduler = Scheduler(cfg, store)
    installer = installer or InstallManager(cfg)
    installer.attach_scheduler(scheduler)  # switching or deleting a variant must not cut a job off

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        (cfg.data_dir / "jobs").mkdir(parents=True, exist_ok=True)
        (cfg.data_dir / "uploads").mkdir(parents=True, exist_ok=True)
        auth.sync_local_token()
        scheduler.start()
        yield
        installer.shutdown()
        scheduler.stop()
        store.close()

    app = FastAPI(title="AHA open", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(GuardMiddleware, cfg=cfg)
    app.state.cfg, app.state.store, app.state.auth, app.state.scheduler = cfg, store, auth, scheduler

    # ------------------------------------------------------------------ errors
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, err: ApiError):
        headers = {"Retry-After": str(math.ceil(err.retry_after_s))} if err.retry_after_s else None
        return JSONResponse(err.body(), status_code=err.status, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, err: RequestValidationError):
        return JSONResponse(invalid("the request is malformed").body(), status_code=400)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, err: StarletteHTTPException):
        code = {404: "not_found", 405: "invalid_request", 413: "invalid_request"}.get(err.status_code, "invalid_request")
        return JSONResponse({"error": {"code": code, "message": str(err.detail), "retry_after_s": None}}, status_code=err.status_code)

    # ------------------------------------------------------------------ helpers
    def client_of(request: Request) -> dict:
        return auth.authenticate(request.headers.get("authorization"))

    def job_view(row: dict) -> dict:
        results = [{k: v for k, v in r.items() if k != "file"} for r in json.loads(row["results_json"])]
        total = row["total_steps"]
        view = {
            "id": row["id"], "status": row["status"],
            "progress": {"step": row["step"], "total_steps": total, "fraction": round(row["step"] / total, 4) if total else 0.0},
            "queue_position": scheduler.queue_position(row["id"]) if row["status"] == "queued" else None,
            "results": results if row["status"] == "succeeded" else [],
            "error": json.loads(row["error_json"]) if row["error_json"] and row["status"] in ("failed", "cancelled") else None,
            "created_at": iso(row["created_at"]),
        }
        return view

    # ------------------------------------------------------------------ health / manifest
    @app.get("/v1/health")
    def health():
        h = scheduler.health()
        gpus = [{**g, "active_model": h["active"].get(g["index"])} for g in detect_gpus()]
        status = "starting" if not scheduler.started else ("degraded" if h["degraded"] else "ok")
        return {
            "status": status, "api_version": API_VERSION, "gpus": gpus,
            "queue": {"queued": h["queued"], "running": h["running"]},
        }

    @app.get("/v1/manifest")
    def manifest(request: Request):
        client_of(request)
        loading = scheduler.health()["loading"]
        models = [manifest_model(cfg, mc, spec, loading=mc.id in loading) for mc, spec in enabled_models(cfg)]
        return {
            "api_version": API_VERSION,
            "provider": {"name": "aha-open", "version": __version__},
            "profile": cfg.profile,
            "limits": {
                "part_size": cfg.part_size, "max_upload_bytes": cfg.max_upload_bytes,
                "max_queue_per_client": cfg.max_queue_per_client, "result_ttl_s": cfg.result_ttl_s,
            },
            "optional": {"enroll": cfg.enroll},
            "models": models,
        }

    # ------------------------------------------------------------------ admin (local profile, loopback only)
    def admin_client(request: Request) -> dict:
        peer = request.client.host if request.client else None
        if not admin_allowed(cfg.profile, peer, request.headers.get("host", "")):
            raise not_found("not found")  # the same answer as for a path that does not exist
        return client_of(request)

    @app.get("/admin/loras")
    def admin_list_loras(request: Request):
        admin_client(request)
        found, problems = scan_loras(cfg)
        return {"loras": [lora.public() for lora in found.values()], "problems": problems}

    @app.put("/admin/loras/{lora_id}", status_code=201)
    async def admin_put_lora(lora_id: str, request: Request):
        """The body is the raw .safetensors file; details travel in the query (name, trigger, scale, license, notes)."""
        admin_client(request)
        if not LORA_ID_RE.match(lora_id):
            raise invalid("the id must be lowercase letters, digits, '-' or '_' (at most 64)")
        query = request.query_params
        meta: dict = {"display_name": query.get("name"), "trigger": query.get("trigger"),
                      "license": query.get("license"), "notes": query.get("notes")}
        if query.get("scale"):
            try:
                meta["default_scale"] = float(query["scale"])
            except ValueError as err:
                raise invalid("scale must be a number") from err
        try:
            check_meta(lora_id, {k: v for k, v in meta.items() if v not in (None, "")})  # refuse before receiving gigabytes
        except LoraError as err:
            raise invalid(str(err)) from err
        # Same disk as loras/, so the final step is a rename, never a second copy. The file carries the id the user typed, so
        # every message about it ("<id>.safetensors is not a safetensors file") speaks of something the user recognises.
        incoming = cfg.data_dir / "tmp" / secrets.token_hex(8)
        incoming.mkdir(parents=True, exist_ok=True)
        path = incoming / f"{lora_id}.safetensors"
        try:
            size = 0
            with path.open("wb") as fh:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > LORA_MAX_BYTES:
                        raise invalid(f"the file is larger than {LORA_MAX_BYTES // 2**30} GiB")
                    fh.write(chunk)
            try:
                return add_lora(cfg, path, lora_id, move=True, **meta).public()
            except LoraError as err:
                raise invalid(str(err)) from err
        finally:
            shutil.rmtree(incoming, ignore_errors=True)  # after a successful move the folder is already empty

    @app.delete("/admin/loras/{lora_id}", status_code=204)
    def admin_delete_lora(lora_id: str, request: Request):
        admin_client(request)
        if not remove_lora(cfg, lora_id):
            raise not_found("unknown LoRA")
        return Response(status_code=204)

    # The support report (aha_open/report.py): text the person copies or saves and sends to whoever supports them. Same guard as the
    # other admin routes, so the logs it carries are never offered to the network.
    @app.get("/admin/report")
    def admin_report(request: Request):
        admin_client(request)
        return Response(build_report(cfg), media_type="text/plain; charset=utf-8")

    # Install a model from the web page (aha_open/model_install.py). Same guard as the LoRA page: this machine, local profile.
    @app.get("/admin/models")
    def admin_list_models(request: Request):
        admin_client(request)
        return {"models": installer.models()}

    @app.post("/admin/models/{model_id}/install", status_code=202)
    def admin_install_model(model_id: str, request: Request, body: dict = BODY):
        admin_client(request)
        return installer.start(model_id, body.get("accept_license"), body.get("confirm_territory", False), body.get("variant"))

    @app.get("/admin/models/{model_id}/install")
    def admin_install_status(model_id: str, request: Request):
        admin_client(request)
        if model_id not in INSTALLABLE:
            raise not_found("this model cannot be installed from here")
        return installer.status(model_id)

    @app.post("/admin/models/{model_id}/use")
    def admin_use_variant(model_id: str, request: Request, body: dict = BODY):
        admin_client(request)
        return installer.use(model_id, body.get("variant"))

    @app.delete("/admin/models/{model_id}/variants/{variant}")
    def admin_remove_variant(model_id: str, variant: str, request: Request):
        admin_client(request)
        return installer.remove(model_id, variant, request.query_params.get("include_shared") == "true")

    @app.delete("/admin/models/{model_id}/install", status_code=204)
    def admin_cancel_install(model_id: str, request: Request):
        admin_client(request)
        if not installer.cancel(model_id):
            raise not_found("no installation is running for this model")
        return Response(status_code=204)

    # ------------------------------------------------------------------ uploads
    @app.post("/v1/uploads", status_code=201)
    def create_upload(request: Request, body: dict = BODY):
        client = client_of(request)
        nbytes, sha, media = body.get("bytes"), body.get("sha256"), body.get("media_type")
        if not isinstance(nbytes, int) or isinstance(nbytes, bool) or nbytes < 1:
            raise invalid("bytes must be a positive integer")
        if nbytes > cfg.max_upload_bytes:
            raise invalid(f"bytes exceeds the limit of {cfg.max_upload_bytes}")
        if not isinstance(sha, str) or not SHA_RE.match(sha):
            raise invalid("sha256 must be 64 lowercase hex characters")
        if not isinstance(media, str) or media.split("/", 1)[0] not in ("image", "video", "audio") or "/" not in media:
            raise invalid("media_type must be an image/*, video/* or audio/* type")
        if store.count_uploads(client["id"]) >= MAX_OPEN_UPLOADS_PER_CLIENT:
            raise busy("too many uploads are stored for this client; use them or wait for them to expire", 30.0)
        upload_id = new_id("up")
        scheduler.upload_dir(upload_id).mkdir(parents=True, exist_ok=True)
        store.add_upload(upload_id, client["id"], nbytes, sha, media, now())
        return {"upload_id": upload_id, "status": "open", "part_size": cfg.part_size, "bytes": nbytes}

    @app.put("/v1/uploads/{upload_id}/parts/{n}", status_code=204)
    async def put_part(upload_id: str, n: int, request: Request):
        client = client_of(request)
        row = store.upload(upload_id, client["id"])
        if row is None:
            raise not_found("unknown upload")
        if row["status"] != "open":
            raise invalid("this upload is already complete")
        parts = max(1, math.ceil(row["bytes"] / cfg.part_size))
        if not 0 <= n < parts:
            raise invalid(f"part number must be between 0 and {parts - 1}")
        expected = cfg.part_size if n < parts - 1 else row["bytes"] - cfg.part_size * (parts - 1)
        target = scheduler.upload_dir(upload_id) / f"part_{n:06d}"
        tmp = target.with_suffix(".tmp")
        size = 0
        with tmp.open("wb") as fh:
            async for chunk in request.stream():
                size += len(chunk)
                if size > expected:
                    break
                fh.write(chunk)
        if size != expected:
            tmp.unlink(missing_ok=True)
            raise invalid(f"part {n} must be exactly {expected} bytes")
        tmp.replace(target)
        return Response(status_code=204)

    @app.post("/v1/uploads/{upload_id}/complete")
    def complete_upload(upload_id: str, request: Request):
        client = client_of(request)
        row = store.upload(upload_id, client["id"])
        if row is None:
            raise not_found("unknown upload")
        updir = scheduler.upload_dir(upload_id)
        if row["status"] == "complete":
            return {"upload_id": upload_id, "status": "complete", "bytes": row["bytes"]}
        parts = max(1, math.ceil(row["bytes"] / cfg.part_size))
        files = [updir / f"part_{i:06d}" for i in range(parts)]
        if not all(f.exists() for f in files):
            raise ApiError(400, "upload_incomplete", "some parts have not been sent yet")
        data = updir / "data"
        with data.open("wb") as out:
            for f in files:
                with f.open("rb") as src:
                    shutil.copyfileobj(src, out, 1024 * 1024)
        if data.stat().st_size != row["bytes"] or sha256_file(data) != row["sha256"]:
            shutil.rmtree(updir, ignore_errors=True)
            store.delete_upload(upload_id)
            raise ApiError(400, "upload_incomplete", "the assembled file does not match the announced sha256; upload it again")
        for f in files:
            f.unlink(missing_ok=True)
        store.set_upload_status(upload_id, "complete")
        return {"upload_id": upload_id, "status": "complete", "bytes": row["bytes"]}

    # ------------------------------------------------------------------ jobs
    @app.post("/v1/jobs", status_code=202)
    def create_job(request: Request, body: dict = BODY):
        client = client_of(request)
        task, model_id, prompt = body.get("task"), body.get("model"), body.get("prompt")
        inputs, params, req_id = body.get("inputs", []), body.get("params", {}), body.get("client_request_id")
        if not isinstance(task, str) or not isinstance(model_id, str):
            raise invalid("task and model are required")
        if req_id is not None and (not isinstance(req_id, str) or not 1 <= len(req_id) <= 128):
            raise invalid("client_request_id must be a string of 1 to 128 characters")
        if req_id:
            previous = store.job_by_request_id(client["id"], req_id, now() - REQUEST_ID_WINDOW_S)
            if previous:
                return job_view(previous)
        mc = cfg.model(model_id)
        spec = CATALOG.get(model_id)
        if mc is None or spec is None or not mc.enabled:
            raise invalid(f"unknown model {model_id!r}")
        state, detail = model_state(cfg, mc, spec)
        if state == "not_installed":
            raise ApiError(409, "model_not_installed", f"model {model_id} is not installed ({detail})")
        if state == "unsupported_hardware":
            raise ApiError(409, "unsupported_hardware", detail or "this hardware cannot run the model")
        if state == "error":
            raise forbidden(f"model {model_id} is not enabled on this server ({detail})")
        if not isinstance(inputs, list) or not all(isinstance(i, dict) for i in inputs):
            raise invalid("inputs must be a list of objects")
        for item in inputs:
            if not isinstance(item.get("role"), str) or not isinstance(item.get("upload_id"), str):
                raise invalid("every input needs a role and an upload_id")
        available, _ = scan_loras(cfg)
        accepted = validate_job(spec, task, prompt, params, inputs, frozenset(available), mc.variant)  # 400 before anything is queued
        lora_files = [
            {**item, "path": str(available[item["id"]].path), "sha256": available[item["id"]].sha256()} for item in accepted.get("loras", [])
        ]
        accepted.setdefault("seed", secrets.randbelow(2**31))  # a result can always be reproduced from its job's params
        resolved = []
        for item in inputs:
            up = store.upload(item["upload_id"], client["id"])
            if up is None:
                raise not_found(f"unknown upload {item['upload_id']}")
            if up["status"] != "complete":
                raise ApiError(400, "upload_incomplete", f"upload {item['upload_id']} has not been completed")
            if not up["media_type"].startswith(ROLE_MEDIA[item["role"]]):
                raise invalid(f"role {item['role']} needs a {ROLE_MEDIA[item['role']]}* upload")
            resolved.append({"role": item["role"], "path": str(scheduler.upload_dir(up["id"]) / "data"), "media_type": up["media_type"]})

        with scheduler.admission_lock:
            if store.recent_job_count(client["id"], now() - 60) >= cfg.max_jobs_per_minute:
                raise busy("too many jobs per minute for this client", 30.0)
            if store.active_count(client["id"]) >= cfg.max_queue_per_client:
                raise busy("this client already has the maximum number of waiting jobs", 10.0)
            if store.active_count() >= cfg.max_queue_total:
                raise busy("the server queue is full", 15.0)
            job_id = new_id("job")
            jobdir = scheduler.job_dir(job_id)
            jobdir.mkdir(parents=True, exist_ok=True)
            try:
                own_inputs(jobdir, resolved)
            except OSError as err:
                shutil.rmtree(jobdir, ignore_errors=True)  # no job row yet: nothing else would ever remove it
                if isinstance(err, FileNotFoundError):  # the upload expired between the check above and here
                    raise not_found("an upload expired while the job was being created; upload it again") from err
                raise
            (jobdir / "request.json").write_text(
                json.dumps({
                    "task": task, "model": model_id, "variant": mc.variant, "prompt": prompt, "params": accepted,
                    "inputs": resolved, "loras": lora_files,
                }),
                encoding="utf-8",
            )
            store.add_job(job_id, client["id"], task, model_id, req_id, now())
            scheduler.submit(job_id, client["id"], mc)
        log.info("job %s accepted (model=%s task=%s)", job_id, model_id, task)
        return job_view(store.job(job_id))

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str, request: Request):
        client = client_of(request)
        row = store.job_for(job_id, client["id"])
        if row is None:
            raise not_found("unknown job")
        return job_view(row)

    @app.post("/v1/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, request: Request):
        client = client_of(request)
        row = store.job_for(job_id, client["id"])
        if row is None:
            raise not_found("unknown job")
        if row["status"] not in TERMINAL:
            scheduler.cancel(job_id)
            deadline = now() + min(cfg.cancel_grace_s, 5.0) + 1.0
            while now() < deadline:  # a running job stops at the next step; give the answer after that, not before
                row = store.job_for(job_id, client["id"])
                if row["status"] in TERMINAL:
                    break
                time.sleep(0.05)
        return job_view(store.job_for(job_id, client["id"]))

    @app.get("/v1/jobs/{job_id}/result")
    def get_result(job_id: str, request: Request, index: int = 0):
        client = client_of(request)
        row = store.job_for(job_id, client["id"])
        if row is None:
            raise not_found("unknown job")
        if row["status"] != "succeeded":
            raise ApiError(409, "job_not_ready", f"the job is {row['status']}; a result exists only after it succeeded")
        if row["expired"]:
            raise not_found("the result has expired")
        results = json.loads(row["results_json"])
        if not 0 <= index < len(results):
            raise not_found("no such result index")
        path = scheduler.job_dir(job_id) / results[index]["file"]
        if not path.is_file():
            raise not_found("the result has expired")
        return FileResponse(path, media_type=results[index]["media_type"])

    # ------------------------------------------------------------------ pairing
    @app.post("/v1/enroll", status_code=201)
    def create_enroll(body: dict = BODY):
        made = auth.create_enroll(body.get("client_name", ""), body.get("client_kind", ""))
        return {"id": made["id"], "enroll_code": made["enroll_code"], "expires_at": iso(made["expires_at"])}

    @app.get("/v1/enroll/{enroll_id}")
    def get_enroll(enroll_id: str):
        if not cfg.enroll:
            raise not_found("pairing is not enabled on this server")
        return auth.poll_enroll(enroll_id)

    ui_dir = Path(__file__).parent / "ui"
    if ui_dir.is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount("/ui", StaticFiles(directory=ui_dir, html=True), name="ui")  # static files only; they call /v1 like any client

    return app
