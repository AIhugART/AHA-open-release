"""Install a model from the web page, for the person at this machine (`/admin/models/*`, docs/04_architecture.md).

The work is done by a CHILD PROCESS running the same `aha-open models install` the command line uses, so there is exactly
one way to install a model. That is also what makes cancelling real: the whole process tree is killed (a download resumes
the next time). The gateway never imports `huggingface_hub`, and no request waits for the download: the page polls.

When the child exits cleanly the gateway re-reads `config.toml` and replaces `cfg.models` in place; the model's state is
computed from disk on every request (catalog.model_state), so the manifest changes with no restart.
"""

from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

from .catalog import CATALOG, detect_gpus, model_state, variant_installed, variant_public, weights_dir
from .config import Config, ModelConfig, load
from .errors import ApiError, invalid, not_found
from .hardware import detect_ram_gib, fit_by_tier, fit_for, free_disk_gib, gpu_budget_mib, recommend_variant_for
from .install import (
    INSTALLABLE,
    PROGRESS_ENV,
    PROGRESS_PREFIX,
    _missing_gib,
    remove_variant,
    use_variant,
    uv_executable,
)
from .util import iso, now

GIB = 2**30
TAIL_LINES = 40
SIZE_POLL_S = 1.0
MESSAGE_LIMIT = 300

# What the child prints (aha_open/install.py) -> the phase the page shows.
PHASES = (
    ("Installing ", "preparing"),
    ("Downloading ", "downloading"),
    ("Creating the ", "creating_environment"),
    ("Installed ", "finishing"),
)
# Output of a download or `uv` that could not reach the network. Matched on the tail of the child's output.
# The machine reached the server but does not trust what it answered with: a company proxy, an antivirus that scans HTTPS, or a
# captive portal. Checked BEFORE the network markers, because Python words it as a ConnectError / SSLError too (observed 2026-11-25).
CERTIFICATE_MARKERS = ("certificate_verify_failed", "certificate verify failed", "self-signed certificate", "unable to get local issuer")
NETWORK_MARKERS = (
    "connectionerror", "max retries exceeded", "getaddrinfo", "name resolution", "timed out", "connection reset",
    "sslerror", "proxyerror", "network is unreachable", "error sending request", "dns error", "failed to fetch",
    "failed to download", "could not resolve", "connection refused", "connecterror", "readtimeout",
)

Command = Callable[[str, str], list[str]]


def default_command(cfg: Config) -> Command:
    """`python -m aha_open.cli --data-dir D models install MODEL --accept-license KEY` with the interpreter running the gateway."""

    def build(model_id: str, license_key: str) -> list[str]:
        return [
            sys.executable, "-m", "aha_open.cli", "--data-dir", str(cfg.data_dir),
            "models", "install", model_id, "--accept-license", license_key,
        ]

    return build


def has_download_support() -> bool:
    return importlib.util.find_spec("huggingface_hub") is not None


def _kill_tree(proc: subprocess.Popen) -> None:
    """Stop the child AND what it started (`uv`, the download): a bare terminate() would leave them running."""
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, check=False)
    else:
        import signal

        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _dir_bytes(path: Path) -> int:
    total = 0
    stack = [path]
    while stack:
        try:
            with os.scandir(stack.pop()) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    else:
                        try:
                            total += entry.stat(follow_symlinks=False).st_size
                        except OSError:
                            continue
        except OSError:
            continue
    return total


def classify_failure(returncode: int, tail: list[str], data_dir: Path) -> dict:
    """One sentence the person can act on, and a code for the cause (docs/04_architecture.md)."""
    text = "\n".join(tail).lower()
    last_error = next((line for line in reversed(tail) if line.startswith("error:")), "")
    if any(marker in text for marker in CERTIFICATE_MARKERS):
        return {
            "code": "certificate_untrusted",
            "message": "This computer does not trust the certificate the download server presented. A company proxy, an antivirus that "
                       "scans secure connections, or a captive portal is usually the cause. Ask the person who runs the network for its "
                       "root certificate and point the SSL_CERT_FILE environment variable at it, or download on another network; "
                       "what was already downloaded is kept.",
        }
    if any(marker in text for marker in NETWORK_MARKERS):
        return {
            "code": "network_unreachable",
            "message": "The download could not reach the internet. Check the connection (and any proxy or firewall), then press "
                       "Download again; what was already downloaded is kept.",
        }
    line = last_error or next((ln for ln in reversed(tail) if ln.strip()), f"the installer stopped (exit {returncode})")
    line = line.replace(str(data_dir), "<data folder>")[:MESSAGE_LIMIT]
    return {"code": "install_failed", "message": f"The installation stopped: {line}"}


class _Run:
    def __init__(self, model_id: str, target: Path, expected_bytes: int) -> None:
        self.model_id = model_id
        self.state = "running"
        self.phase = "preparing"
        self.target = target
        self.expected_bytes = expected_bytes
        self.error: dict | None = None
        self.started_at = now()
        self.finished_at: float | None = None
        self.tail: deque[str] = deque(maxlen=TAIL_LINES)
        self.proc: subprocess.Popen | None = None
        self.cancelled = False
        self.size = (0.0, 0)  # (when it was measured, bytes of the weights folder): only a fallback, see `reported`
        self.reported: tuple[int, int] | None = None  # (bytes done, bytes total) as the downloader itself says


class InstallManager:
    def __init__(self, cfg: Config, command: Command | None = None) -> None:
        self.cfg = cfg
        self._command = command or default_command(cfg)
        self._lock = threading.Lock()
        self._run: _Run | None = None  # the running install, or the last one that ended

    # ------------------------------------------------------------------ what the page shows
    def models(self) -> list[dict]:
        out = []
        for model_id, plan in INSTALLABLE.items():
            if not plan.get("web", True):  # a model that can only be installed from the command line
                continue
            spec = CATALOG[model_id]
            variant, _ = self._variant(model_id)
            mc = self.cfg.model(model_id) or ModelConfig(id=model_id, worker=spec.family, variant=variant or "")
            state, detail = model_state(self.cfg, mc, spec)
            if detail:  # "weights not found at <path>": the page shows it, and it must not spell out the data folder
                detail = detail.replace(str(self.cfg.data_dir), "<data folder>")
            blocker = None if state == "ready" else self._blocker(model_id)
            lic = spec.license
            out.append({
                "id": model_id,
                "display_name": spec.display_name,
                "state": state,
                "state_detail": detail,
                "license": {
                    "key": plan["license_key"], "name": lic.get("name"), "commercial_use": lic.get("commercial_use"),
                    "notes": lic.get("notes", ""), "url": lic.get("url", ""),
                    "excluded_territories": list(lic.get("excluded_territories", [])),
                },
                # a model whose licence excludes territories needs the person to confirm they are not in one (its own checkbox)
                "requires_territory_confirmation": bool(plan.get("territory")),
                "disk_gib": self._disk_gib(model_id, plan, variant),
                "variants": self._variants(model_id, plan, mc, variant),
                "blocker": blocker,
                "install": self.status(model_id),
            })
        return out

    def status(self, model_id: str) -> dict:
        with self._lock:
            run = self._run if self._run and self._run.model_id == model_id else None
            if run is None:
                return {"state": "idle", "phase": None, "downloaded_bytes": 0, "expected_bytes": 0, "error": None,
                        "started_at": None, "finished_at": None}
            done, expected = run.size[1], run.expected_bytes
            if run.reported is not None:  # the downloader's own count: the folder under-counts (hf-xet assembles files at the end)
                done = run.reported[0]
                if run.reported[1] >= expected * 0.9:  # the total grows as files are discovered; trust it once it is nearly complete
                    expected = run.reported[1]
            elif run.state == "running" and run.phase == "downloading" and time.monotonic() - run.size[0] >= SIZE_POLL_S:
                run.size = (time.monotonic(), _dir_bytes(run.target))
                done = run.size[1]
            return {
                "state": run.state, "phase": run.phase if run.state == "running" else None,
                "downloaded_bytes": done, "expected_bytes": expected, "error": run.error,
                "started_at": iso(run.started_at), "finished_at": iso(run.finished_at) if run.finished_at else None,
            }

    # ------------------------------------------------------------------ start / cancel
    def start(self, model_id: str, accept_license: object, confirm_territory: object = False, variant: object = None) -> dict:
        plan = INSTALLABLE.get(model_id)
        if plan is None or CATALOG.get(model_id) is None or not plan.get("web", True):
            raise not_found("this model cannot be installed from here")
        if accept_license != plan["license_key"]:
            raise invalid("the licence has not been accepted: send accept_license with the licence key shown on the page")
        if plan.get("territory") and confirm_territory is not True:  # the same refusal the command line gives without --confirm-territory
            excluded = ", ".join(CATALOG[model_id].license.get("excluded_territories", []))
            raise invalid(f"the territory has not been confirmed: this licence does not allow use in {excluded}; "
                          "send confirm_territory: true only if you are not in one of them")
        if variant is not None and (not isinstance(variant, str) or variant not in plan["variants"]):
            raise invalid(f"variant must be one of: {', '.join(plan['variants'])}")
        with self._lock:
            if self._run and self._run.state == "running":
                raise ApiError(409, "install_running", "a model is being installed already; wait for it to finish or cancel it")
            spec = CATALOG[model_id]
            chosen = variant  # None = the best one for this machine
            variant = chosen or self._variant(model_id)[0]
            mc = self.cfg.model(model_id) or ModelConfig(id=model_id, worker=spec.family, variant=variant or "")
            # no variant named: "installed" is what it always was - the model works as configured. A variant named: that one is on disk.
            installed = (model_state(self.cfg, mc, spec)[0] == "ready") if (chosen is None or mc.variant == chosen) else variant_installed(self.cfg, spec, chosen)
            if installed:
                raise ApiError(409, "already_installed", f"{spec.display_name} ({variant}) is already installed")
            blocker = self._blocker(model_id, chosen)
            if blocker:
                raise ApiError(409, blocker["code"], blocker["message"])
            variant = variant or next(iter(plan["variants"]))
            target = weights_dir(self.cfg, ModelConfig(id=model_id, worker=spec.family, variant=variant))
            run = _Run(model_id, target, int(self._disk_gib(model_id, plan, variant) * GIB))
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "AHA_OPEN_UV": uv_executable() or "",
                   PROGRESS_ENV: "lines"}
            kwargs: dict = {}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True
            command = self._command(model_id, plan["license_key"])
            if plan.get("territory"):
                command = [*command, "--confirm-territory"]  # confirmed above; the command line asks for it again, so it is passed on
            if chosen:
                command = [*command, "--variant", chosen]
            run.proc = subprocess.Popen(  # cwd: see launcher.start - `python -m` must not pick up an `aha_open` from the current folder
                command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env, cwd=str(self.cfg.data_dir), **kwargs,
            )
            self._run = run
        threading.Thread(target=self._watch, args=(run,), name=f"install-{model_id}", daemon=True).start()
        return self.status(model_id)

    def cancel(self, model_id: str) -> bool:
        with self._lock:
            run = self._run
            if run is None or run.model_id != model_id or run.state != "running" or run.proc is None:
                return False
            run.cancelled = True
            proc = run.proc
        _kill_tree(proc)
        return True

    def attach_scheduler(self, scheduler) -> None:
        """The scheduler knows which jobs are waiting or running: switching or deleting a variant must not cut one off."""
        self._scheduler = scheduler

    def use(self, model_id: str, variant: object) -> dict:
        """Select an installed variant. The next job loads it (the worker that holds the old one is released when no job needs it)."""
        spec = self._variant_model(model_id, variant)
        self._guard(model_id)
        done, message = use_variant(self.cfg, model_id, variant)
        if not done:
            raise ApiError(409, "variant_not_installed", message[0].upper() + message[1:] + ".")
        if self._scheduler is not None:
            self._scheduler.release(model_id)
        return self._entry(model_id, spec)

    def remove(self, model_id: str, variant: object, include_shared: bool = False) -> dict:
        """Delete one variant's files (never while a job of this model is waiting or running, and never outside the models folder)."""
        spec = self._variant_model(model_id, variant)
        self._guard(model_id)
        if self._scheduler is not None:
            self._scheduler.release(model_id)  # a loaded worker holds the weights open: Windows would refuse to delete them
        done, message = remove_variant(self.cfg, model_id, variant, include_shared=include_shared)
        if not done:
            raise not_found(message[0].upper() + message[1:] + ".")
        offer = "--include-shared" in message  # the last variant went and its shared files are still there: the page offers to remove them too
        message = message.split(": add --include-shared")[0] if offer else message  # a command-line flag means nothing on the page
        return {**self._entry(model_id, spec), "message": message.replace(str(self.cfg.data_dir), "<data folder>"), "shared_offer": offer}

    def shutdown(self) -> None:
        """The gateway is stopping: do not leave a download running with nobody to report to."""
        with self._lock:
            run = self._run
            proc = run.proc if run and run.state == "running" else None
            if run and proc:
                run.cancelled = True
        if proc:
            _kill_tree(proc)

    # ------------------------------------------------------------------ internals
    _scheduler = None

    def _variant_model(self, model_id: str, variant: object):
        spec = CATALOG.get(model_id)
        if spec is None or model_id not in INSTALLABLE or not spec.variants:
            raise not_found("this model has no variants to choose from")
        if not isinstance(variant, str) or variant not in spec.variants:
            raise invalid(f"variant must be one of: {', '.join(spec.variants)}")
        return spec

    def _guard(self, model_id: str) -> None:
        with self._lock:
            if self._run and self._run.state == "running":
                raise ApiError(409, "install_running", "a model is being installed; wait for it to finish or cancel it first")
        if self._scheduler is not None and self._scheduler.busy(model_id):
            raise ApiError(409, "model_busy", "a job of this model is waiting or running; wait for it to finish or cancel it, then try again")

    def _entry(self, model_id: str, spec) -> dict:
        return next(m for m in self.models() if m["id"] == model_id)

    def _variants(self, model_id: str, plan: dict, mc: ModelConfig, suggestion: str | None) -> list[dict]:
        """Every variant the model offers, for the page: what it costs, whether it fits here, whether it is on disk, which one runs."""
        spec = CATALOG[model_id]
        gpus = detect_gpus()
        gpu_mib = gpu_budget_mib(self.cfg, gpus[0] if gpus else None)
        out = []
        for name, info in spec.variants.items():
            if name not in plan["variants"]:
                continue
            installed = variant_installed(self.cfg, spec, name)
            out.append({
                "name": name, **variant_public(info), "own_gib": self._own_gib(spec, plan, name),
                "download_gib": 0 if installed else self._disk_gib(model_id, plan, name),  # what is still to fetch: shared files already here do not count
                "fit": fit_for(info, gpu_mib), **({"fit_by_tier": t} if (t := fit_by_tier(info, gpu_mib)) else {}), "installed": installed,
                "selected": name == mc.variant, "recommended": name == suggestion,
            })
        return out

    @staticmethod
    def _own_gib(spec, plan: dict, name: str) -> float:
        """What deleting this variant frees: its own transformer file for a ConvRot build (the text encoder and VAE are shared), the whole folder otherwise."""
        info = plan["variants"][name]
        own = spec.variants[name].get("files", ())
        if not own or "parts" not in info:
            return float(info["disk_gib"])
        sizes = {f"{p['subdir']}/{n}" if p["subdir"] else n: size for p in info["parts"] for n, (size, _) in p["files"].items()}
        return round(sum(sizes.get(f, 0) for f in own) / GIB, 1)

    def _variant(self, model_id: str) -> tuple[str | None, str]:
        """The variant for this machine, and the sentence why not: the rule is the model's own, the one the command line uses."""
        gpus = detect_gpus()
        return recommend_variant_for(model_id, gpu_budget_mib(self.cfg, gpus[0] if gpus else None), detect_ram_gib())

    def _disk_gib(self, model_id: str, plan: dict, variant: str | None) -> int:
        """GB still to download. For a model made of several pinned files only those not on disk yet count: the reference model shares
        a folder with the text model, so a machine that has the text model needs about 14 GB, not 61."""
        info = plan["variants"][variant or next(iter(plan["variants"]))]
        if "parts" not in info:
            return info["disk_gib"]
        spec = CATALOG[model_id]
        target = weights_dir(self.cfg, ModelConfig(id=model_id, worker=spec.family, variant=variant or ""))
        return max(1, math.ceil(min(info["disk_gib"], _missing_gib(info["parts"], target))))

    def _blocker(self, model_id: str, chosen: str | None = None) -> dict | None:
        """Why an install would fail, checked BEFORE starting, with the same functions the command line uses."""
        plan, spec = INSTALLABLE[model_id], CATALOG[model_id]
        variant, reason = (chosen, "") if chosen else self._variant(model_id)
        if variant is None:
            return {"code": "unsupported_hardware", "message": reason[0].upper() + reason[1:] + "."}
        if uv_executable() is None:
            return {"code": "uv_missing", "message": "The tool that builds the model's environment (uv) was not found. "
                                                     "Reinstall AHA open, or install uv from https://docs.astral.sh/uv/."}
        if not has_download_support():
            return {"code": "download_support_missing", "message": "This installation cannot download models "
                    "(huggingface-hub is missing). Reinstall AHA open, or run: pip install \"aha-open[download]\"."}
        info = plan["variants"][variant]
        target = weights_dir(self.cfg, ModelConfig(id=model_id, worker=spec.family, variant=variant))
        free = free_disk_gib(target)
        need_gib = self._disk_gib(model_id, plan, variant)
        files_missing = "parts" in info and _missing_gib(info["parts"], target) > 0  # the text model is here, the reference file is not
        if free < need_gib * 1.1 and (files_missing or not (target / "model_index.json").exists()):
            return {"code": "not_enough_disk", "message": f"{spec.display_name} needs about {need_gib} GB of free disk "
                    f"(plus room for its environment) but only {free:.0f} GB are free on that drive. Free some space, then try again."}
        return None

    def _watch(self, run: _Run) -> None:
        proc = run.proc
        assert proc is not None and proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip()
            if not line:
                continue
            if line.startswith(PROGRESS_PREFIX + " "):  # machine-readable progress: not an output line, never part of an error message
                try:
                    _, done, total = line.split()
                    with self._lock:
                        run.reported = (int(done), int(total))
                except ValueError:
                    pass
                continue
            with self._lock:
                run.tail.append(line)
                for prefix, phase in PHASES:
                    if line.startswith(prefix):
                        run.phase = phase
        code = proc.wait()
        error = None
        state = "succeeded"
        if run.cancelled:
            state = "cancelled"
        elif code != 0:
            state, error = "failed", classify_failure(code, list(run.tail), self.cfg.data_dir)
        else:
            try:  # the installer wrote config.toml in another process; adopt it in place so the scheduler sees the model
                self.cfg.models[:] = load(self.cfg.data_dir).models
            except Exception as err:  # noqa: BLE001 - say so rather than report success with a stale configuration
                state = "failed"
                error = {"code": "install_failed", "message": f"The model was installed but its settings could not be read: {err}"}
        with self._lock:
            run.state, run.error, run.finished_at = state, error, now()
            run.size = (time.monotonic(), _dir_bytes(run.target))
