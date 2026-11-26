"""A `/v1` client for the tools. Output is built by COPYING an allowlist of fields (plan D8): a field added to the server
tomorrow is excluded until someone decides it is safe, instead of leaking until someone notices."""

from __future__ import annotations

import re

import httpx

from .targets import Target, TargetError

# GET paths a read-only probe may call. Anything else is refused, not "an error".
PROBE_PATHS = (
    (re.compile(r"^/v1/health$"), "health", False),
    (re.compile(r"^/v1/manifest$"), "manifest", True),
    (re.compile(r"^/v1/jobs/[A-Za-z0-9_-]{1,64}$"), "job", True),
)


def _pick(src: dict, keys: tuple[str, ...]) -> dict:
    return {k: src[k] for k in keys if k in src}


def project_health(d: dict) -> dict:
    return {**_pick(d, ("status", "api_version", "queue")), "gpus": [
        _pick(g, ("index", "name", "memory_total_mib", "memory_free_mib", "active_model")) for g in d.get("gpus", [])]}


def project_manifest(d: dict) -> dict:
    return {
        **_pick(d, ("api_version", "profile", "limits")),
        "provider": _pick(d.get("provider", {}), ("name", "version")),
        "models": [_pick(m, ("id", "engine", "tasks", "state", "state_detail")) for m in d.get("models", [])],
    }


def project_job(d: dict) -> dict:
    return {
        **_pick(d, ("id", "status", "progress", "queue_position", "created_at")),
        # an allowlist: only fields known to be safe to show. A video result also says how long it is, how fast, and whether it has sound.
        "results": [_pick(r, ("kind", "media_type", "width", "height", "bytes", "sha256", "duration_s", "fps", "has_audio")) for r in d.get("results", [])],
        "error": _pick(d["error"], ("code", "message")) if isinstance(d.get("error"), dict) else None,
    }


PROJECTORS = {"health": project_health, "manifest": project_manifest, "job": project_job}


def get_json(target: Target, path: str, auth: bool, timeout: float = 10.0) -> tuple[int, dict | None]:
    headers = {"Authorization": f"Bearer {target.token()}"} if auth else {}
    try:
        r = httpx.get(target.base_url + path, headers=headers, timeout=timeout, verify=target.cfg.profile != "lan")
    except httpx.HTTPError as err:
        raise TargetError(f"cannot reach {target.base_url} ({err.__class__.__name__}): the server is not listening there") from err
    try:
        body = r.json()
    except ValueError:
        body = None
    return r.status_code, body if isinstance(body, dict) else None


def probe(target: Target, path: str, with_token: bool = True) -> dict:
    """`with_token=False` asks a protected path WITHOUT credentials, to prove it is refused."""
    for pattern, kind, auth in PROBE_PATHS:
        if pattern.match(path):
            status, body = get_json(target, path, auth and with_token)
            if status != 200 or body is None:
                err = body.get("error", {}) if body else {}
                return {"http_status": status, "error": _pick(err, ("code", "message"))}
            return {"http_status": status, kind: PROJECTORS[kind](body)}
    raise TargetError(f"probe_v1 only reads {[p.pattern for p, _, _ in PROBE_PATHS]}; {path!r} is refused")
