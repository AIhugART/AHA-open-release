"""Client tokens, the local token file and pairing (docs/07_security.md rules 4, 11, 12).

A token is shown exactly once and stored as sha256. A pairing token waits in the enrolls table until the client's
first poll takes it; that read wipes it.
"""

from __future__ import annotations

import os
import secrets
import subprocess
import sys

from .config import Config
from .errors import ApiError, busy, unauthorized
from .store import Store
from .util import hash_token, new_id, now

ENROLL_TTL_S = 300
MAX_PENDING_ENROLLS = 5
_CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"  # no 0/O/1/I/L: read aloud or typed from a screen


def new_token() -> str:
    return "aho_" + secrets.token_urlsafe(32)


def new_code() -> str:
    raw = "".join(secrets.choice(_CODE_ALPHABET) for _ in range(6))
    return f"{raw[:4]}-{raw[4:]}"


def restrict_to_current_user(path) -> bool:
    """Make a secret file readable only by the current user. False means we could not prove it (the caller reports it)."""
    try:
        if sys.platform == "win32":
            user = os.environ.get("USERNAME", "")
            if not user:
                return False
            done = subprocess.run(
                ["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:(R,W,D)"],
                capture_output=True, check=False, timeout=15,
            )
            return done.returncode == 0
        os.chmod(path, 0o600)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


class Auth:
    def __init__(self, cfg: Config, store: Store) -> None:
        self.cfg = cfg
        self.store = store

    # ------------------------------------------------------------------ clients
    def create_client(self, name: str, kind: str = "", is_local: bool = False) -> tuple[str, str]:
        token = new_token()
        client_id = new_id("cli")
        self.store.add_client(client_id, name, kind, hash_token(token), now(), is_local)
        return client_id, token

    def authenticate(self, header: str | None) -> dict:
        """The client row for an Authorization header. require_token = false (local only) maps everyone to one anonymous client."""
        if not header or not header.lower().startswith("bearer "):
            if not self.cfg.require_token and self.cfg.profile == "local":
                return {"id": "cli_anonymous", "name": "anonymous", "kind": ""}
            raise unauthorized()
        token = header[7:].strip()
        client = self.store.client_by_hash(hash_token(token)) if token else None
        if client is None:
            raise unauthorized()
        return client

    # ------------------------------------------------------------------ local token file (profile local only)
    def sync_local_token(self) -> str | None:
        """local: ensure a token file the local client can read; any other profile: remove it and revoke its client."""
        path = self.cfg.local_token_path
        if self.cfg.profile != "local" or not self.cfg.require_token:
            if path.exists():
                path.unlink()
            self.store.revoke_local_clients(now())
            return None
        if path.exists():
            existing = path.read_text(encoding="utf-8").strip()
            if existing and self.store.client_by_hash(hash_token(existing)):
                return existing
        self.store.revoke_local_clients(now())
        _, token = self.create_client("local", "local", is_local=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(token + "\n", encoding="utf-8")
        restrict_to_current_user(path)
        return token

    # ------------------------------------------------------------------ pairing
    def create_enroll(self, client_name: str, client_kind: str) -> dict:
        if not self.cfg.enroll:
            raise ApiError(404, "not_found", "pairing is not enabled on this server")
        if not isinstance(client_name, str) or not client_name.strip() or len(client_name) > 80:
            raise ApiError(400, "invalid_request", "client_name is required (at most 80 characters)")
        if self.store.pending_enrolls(now()) >= MAX_PENDING_ENROLLS:
            raise busy("too many pairing requests are waiting for approval", 60.0)
        enroll_id, ts = new_id("enr"), now()
        for _ in range(5):
            code = new_code()
            if self.store.enroll_by_code(code) is None:
                break
        self.store.add_enroll(enroll_id, code, client_name.strip(), str(client_kind or "")[:40], ts + ENROLL_TTL_S, ts)
        return {"id": enroll_id, "enroll_code": code, "expires_at": ts + ENROLL_TTL_S}

    def approve_enroll(self, code: str) -> dict:
        row = self.store.enroll_by_code(code.strip().upper())
        if row is None or row["status"] != "pending" or row["expires_at"] <= now():
            raise ApiError(404, "not_found", "no pending pairing request with that code (expired or already used)")
        client_id, token = self.create_client(row["client_name"], row["client_kind"])
        self.store.set_enroll(row["id"], status="approved", pending_token=token, client_id=client_id)
        return {"client_id": client_id, "name": row["client_name"]}

    def deny_enroll(self, code: str) -> bool:
        row = self.store.enroll_by_code(code.strip().upper())
        if row is None or row["status"] != "pending":
            return False
        self.store.set_enroll(row["id"], status="denied")
        return True

    def poll_enroll(self, enroll_id: str) -> dict:
        row = self.store.enroll(enroll_id)
        if row is None:
            raise ApiError(404, "not_found", "unknown pairing request")
        if row["status"] == "pending" and row["expires_at"] <= now():
            self.store.set_enroll(enroll_id, status="expired", pending_token=None)
            return {"status": "expired"}
        if row["status"] == "approved":
            token = self.store.take_enroll_token(enroll_id)
            return {"status": "approved", "token": token} if token else {"status": "approved"}
        return {"status": row["status"]}
