"""SQLite state: clients and pairing (clients.sqlite), uploads and jobs (jobs.sqlite).

Rule: no prompt and no token ever lands in these files (docs/product/07_security.md section 3, rules 4 and 9). Tokens are
stored as sha256; prompts live only in jobs/<id>/request.json while a job is alive.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

TERMINAL = ("succeeded", "failed", "cancelled")
ACTIVE = ("queued", "loading_model", "running")


class _Db:
    def __init__(self, path: Path, schema: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(schema)

    def run(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self.lock:
            return self.conn.execute(sql, args)

    def one(self, sql: str, args: tuple = ()) -> dict | None:
        with self.lock:
            row = self.conn.execute(sql, args).fetchone()
        return dict(row) if row else None

    def all(self, sql: str, args: tuple = ()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def close(self) -> None:
        with self.lock:
            self.conn.close()


CLIENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS clients (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL DEFAULT '',
  token_hash TEXT NOT NULL UNIQUE, created_at REAL NOT NULL, revoked_at REAL, is_local INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS enrolls (
  id TEXT PRIMARY KEY, code TEXT NOT NULL UNIQUE, client_name TEXT NOT NULL, client_kind TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL, expires_at REAL NOT NULL, created_at REAL NOT NULL,
  pending_token TEXT, client_id TEXT
);
"""

JOBS_SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
  id TEXT PRIMARY KEY, client_id TEXT NOT NULL, bytes INTEGER NOT NULL, sha256 TEXT NOT NULL,
  media_type TEXT NOT NULL, status TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, client_id TEXT NOT NULL, task TEXT NOT NULL, model TEXT NOT NULL, status TEXT NOT NULL,
  step INTEGER NOT NULL DEFAULT 0, total_steps INTEGER NOT NULL DEFAULT 0,
  results_json TEXT NOT NULL DEFAULT '[]', error_json TEXT, client_request_id TEXT,
  cancel_requested INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL, finished_at REAL, expired INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS jobs_client ON jobs (client_id, status);
"""


class Store:
    def __init__(self, data_dir: Path) -> None:
        self.clients = _Db(data_dir / "clients.sqlite", CLIENTS_SCHEMA)
        self.jobs = _Db(data_dir / "jobs.sqlite", JOBS_SCHEMA)

    def close(self) -> None:
        self.clients.close()
        self.jobs.close()

    # ------------------------------------------------------------------ clients
    def add_client(self, client_id: str, name: str, kind: str, token_hash: str, ts: float, is_local: bool = False) -> None:
        self.clients.run(
            "INSERT INTO clients (id, name, kind, token_hash, created_at, is_local) VALUES (?,?,?,?,?,?)",
            (client_id, name, kind, token_hash, ts, int(is_local)),
        )

    def client_by_hash(self, token_hash: str) -> dict | None:
        return self.clients.one("SELECT * FROM clients WHERE token_hash=? AND revoked_at IS NULL", (token_hash,))

    def client(self, client_id: str) -> dict | None:
        return self.clients.one("SELECT * FROM clients WHERE id=?", (client_id,))

    def list_clients(self) -> list[dict]:
        return self.clients.all("SELECT id, name, kind, created_at, revoked_at, is_local FROM clients ORDER BY created_at")

    def revoke_client(self, client_id_or_name: str, ts: float) -> int:
        cur = self.clients.run(
            "UPDATE clients SET revoked_at=? WHERE revoked_at IS NULL AND (id=? OR name=?)",
            (ts, client_id_or_name, client_id_or_name),
        )
        return cur.rowcount

    def revoke_local_clients(self, ts: float) -> None:
        self.clients.run("UPDATE clients SET revoked_at=? WHERE is_local=1 AND revoked_at IS NULL", (ts,))

    # ------------------------------------------------------------------ enroll
    def add_enroll(self, enroll_id: str, code: str, name: str, kind: str, expires_at: float, ts: float) -> None:
        self.clients.run(
            "INSERT INTO enrolls (id, code, client_name, client_kind, status, expires_at, created_at) VALUES (?,?,?,?,?,?,?)",
            (enroll_id, code, name, kind, "pending", expires_at, ts),
        )

    def enroll(self, enroll_id: str) -> dict | None:
        return self.clients.one("SELECT * FROM enrolls WHERE id=?", (enroll_id,))

    def enroll_by_code(self, code: str) -> dict | None:
        return self.clients.one("SELECT * FROM enrolls WHERE code=?", (code,))

    def pending_enrolls(self, ts: float) -> int:
        row = self.clients.one("SELECT COUNT(*) AS n FROM enrolls WHERE status='pending' AND expires_at>?", (ts,))
        return int(row["n"]) if row else 0

    def set_enroll(self, enroll_id: str, **fields: object) -> None:
        cols = ", ".join(f"{k}=?" for k in fields)
        self.clients.run(f"UPDATE enrolls SET {cols} WHERE id=?", (*fields.values(), enroll_id))

    def take_enroll_token(self, enroll_id: str) -> str | None:
        """Return the approved token exactly once: the read and the wipe are one critical section."""
        with self.clients.lock:
            row = self.clients.one("SELECT pending_token FROM enrolls WHERE id=?", (enroll_id,))
            if not row or not row["pending_token"]:
                return None
            self.clients.run("UPDATE enrolls SET pending_token=NULL WHERE id=?", (enroll_id,))
            return row["pending_token"]

    # ------------------------------------------------------------------ uploads
    def add_upload(self, upload_id: str, client_id: str, nbytes: int, sha: str, media_type: str, ts: float) -> None:
        self.jobs.run(
            "INSERT INTO uploads (id, client_id, bytes, sha256, media_type, status, created_at) VALUES (?,?,?,?,?,'open',?)",
            (upload_id, client_id, nbytes, sha, media_type, ts),
        )

    def upload(self, upload_id: str, client_id: str) -> dict | None:
        return self.jobs.one("SELECT * FROM uploads WHERE id=? AND client_id=?", (upload_id, client_id))

    def set_upload_status(self, upload_id: str, status: str) -> None:
        self.jobs.run("UPDATE uploads SET status=? WHERE id=?", (status, upload_id))

    def delete_upload(self, upload_id: str) -> None:
        self.jobs.run("DELETE FROM uploads WHERE id=?", (upload_id,))

    def expired_uploads(self, before: float) -> list[dict]:
        return self.jobs.all("SELECT id FROM uploads WHERE created_at<?", (before,))

    def count_uploads(self, client_id: str) -> int:
        return int(self.jobs.one("SELECT COUNT(*) AS n FROM uploads WHERE client_id=?", (client_id,))["n"])

    # ------------------------------------------------------------------ jobs
    def add_job(self, job_id: str, client_id: str, task: str, model: str, request_id: str | None, ts: float) -> None:
        self.jobs.run(
            "INSERT INTO jobs (id, client_id, task, model, status, client_request_id, created_at) VALUES (?,?,?,?,'queued',?,?)",
            (job_id, client_id, task, model, request_id, ts),
        )

    def job(self, job_id: str) -> dict | None:
        return self.jobs.one("SELECT * FROM jobs WHERE id=?", (job_id,))

    def job_for(self, job_id: str, client_id: str) -> dict | None:
        return self.jobs.one("SELECT * FROM jobs WHERE id=? AND client_id=?", (job_id, client_id))

    def job_by_request_id(self, client_id: str, request_id: str, since: float) -> dict | None:
        return self.jobs.one(
            "SELECT * FROM jobs WHERE client_id=? AND client_request_id=? AND created_at>=? ORDER BY created_at DESC",
            (client_id, request_id, since),
        )

    def active_count(self, client_id: str | None = None) -> int:
        marks = ",".join("?" for _ in ACTIVE)
        if client_id is None:
            row = self.jobs.one(f"SELECT COUNT(*) AS n FROM jobs WHERE status IN ({marks})", ACTIVE)
        else:
            row = self.jobs.one(f"SELECT COUNT(*) AS n FROM jobs WHERE client_id=? AND status IN ({marks})", (client_id, *ACTIVE))
        return int(row["n"])

    def recent_job_count(self, client_id: str, since: float) -> int:
        return int(self.jobs.one("SELECT COUNT(*) AS n FROM jobs WHERE client_id=? AND created_at>=?", (client_id, since))["n"])

    def set_job(self, job_id: str, **fields: object) -> None:
        cols = ", ".join(f"{k}=?" for k in fields)
        self.jobs.run(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))

    def advance_job(self, job_id: str, status: str, **fields: object) -> bool:
        """Move to a non-terminal status only while the job is still alive: a cancelled job can never be resurrected."""
        marks = ",".join("?" for _ in TERMINAL)
        sets = ", ".join(["status=?", *(f"{k}=?" for k in fields)])
        cur = self.jobs.run(f"UPDATE jobs SET {sets} WHERE id=? AND status NOT IN ({marks})", (status, *fields.values(), job_id, *TERMINAL))
        return cur.rowcount == 1

    def finish_job(self, job_id: str, status: str, ts: float, results: list | None = None, error: dict | None = None) -> bool:
        """Move a job to a terminal state exactly once. False when it was already terminal (a cancelled job never gets a result)."""
        marks = ",".join("?" for _ in TERMINAL)
        with self.jobs.lock:
            cur = self.jobs.run(
                f"UPDATE jobs SET status=?, finished_at=?, results_json=?, error_json=? WHERE id=? AND status NOT IN ({marks})",
                (status, ts, json.dumps(results or []), json.dumps(error) if error else None, job_id, *TERMINAL),
            )
            return cur.rowcount == 1

    def expirable_jobs(self, before: float) -> list[dict]:
        return self.jobs.all("SELECT id FROM jobs WHERE finished_at IS NOT NULL AND finished_at<? AND expired=0", (before,))
