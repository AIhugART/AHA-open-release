"""The gate before a release: install the `.exe` into a clean folder, set up each engine, run ONE REAL JOB per build type.

    python windows_installer/release_smoke.py --installer dist/AHA-Open-Setup-<v>.exe --weights <folder with the model folders>
        --accept-license qwen-research --accept-license minimax-h3 --confirm-territory
    python windows_installer/release_smoke.py --from-wheel ...      developer run: the built wheel in a temp environment, no record

Root cause it guards: dev2 passed every test and an install smoke, and could not run ConvRot or MiniMax H3 at all, because nothing
ever ran a job from the installed program. Needs the GPU, the weights already on disk and about 20 minutes.

The weights are not downloaded: files are HARD-LINKED into a temporary data folder (same drive), so nothing is copied and deleting the
temporary folder never deletes an original. Everything goes through the web page's own routes (`/admin/models/*`, then `/v1`).
Only a run with `--installer` writes `dist/RELEASE-SMOKE.json`, which `build_installer.py --check-release <tag>` requires.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zlib
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "aha_open_mcp" / "src" / "aha_open_mcp"))
import build_installer as b  # noqa: E402
from mp4 import probe_mp4  # noqa: E402  (stdlib only: a reader that did not write the file)

PROMPT = "a red lighthouse on a rocky coast under a blue sky, waves, photograph"
# What each required build is asked for; which builds are required is build_installer.REQUIRED_JOBS.
REQUESTS = {
    "image.generate": {"width": 1024, "height": 1024, "seed": 42},
    "video.generate": {"short_edge": 352, "seed": 42},
}
LINK_FROM_BYTES = 2**20  # big files are linked; small ones (install.json is rewritten by the program) are copied


def say(text: str) -> None:
    print(text, flush=True)


def link_weights(source: Path, target: Path) -> None:
    """Mirror `source` into `target`: hard links for the large files, real copies for everything the program may rewrite."""
    for path in source.rglob("*"):
        out = target / path.relative_to(source)
        if path.is_dir():
            out.mkdir(parents=True, exist_ok=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        if path.stat().st_size >= LINK_FROM_BYTES:
            try:
                os.link(path, out)
            except OSError as err:
                raise b.BuildError(f"cannot hard-link {path} into {out}: the temporary folder must be on the same drive ({err})") from None
        else:
            shutil.copy2(path, out)


def call(base: str, token: str, method: str, path: str, payload: dict | None = None) -> bytes:
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(base + path, data=body, method=method, headers={"Authorization": f"Bearer {token}"})
    if body:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
    except urllib.error.HTTPError as err:
        raise b.BuildError(f"{method} {path} answered {err.code}: {err.read()[:400]!r}") from None


def check_png(data: bytes, width: int, height: int) -> None:
    """The size asked for, and a picture that is not one flat colour ("the job succeeded" is not "the picture is right")."""
    if data[:8] != b"\x89PNG\r\n\x1a\n" or struct.unpack(">II", data[16:24]) != (width, height):
        raise b.BuildError(f"the result is not a {width}x{height} PNG")
    pos, pixels = 8, b""
    while pos < len(data):
        size, kind = struct.unpack(">I4s", data[pos : pos + 8])
        if kind == b"IDAT":
            pixels += data[pos + 8 : pos + 8 + size]
        pos += size + 12
    if len(set(zlib.decompress(pixels)[::997])) < 32:
        raise b.BuildError("the picture is almost a single value: look at it before releasing")


def check_video(data: bytes, width: int, height: int) -> None:
    probe = probe_mp4(data)
    video = probe["video"] or {}
    if (video.get("width"), video.get("height")) != (width, height) or probe["audio"] is None:
        raise b.BuildError(f"the video is not {width}x{height} with sound: {probe}")


def wait_for(what: str, poll, done, timeout_s: float) -> dict:
    deadline, shown = time.monotonic() + timeout_s, None
    while True:
        state = poll()
        mark = state.get("phase") or state.get("status") or state.get("state")
        if mark != shown:
            say(f"  {what}: {mark}")
            shown = mark
        if done(state):
            return state
        if time.monotonic() > deadline:
            raise b.BuildError(f"{what} did not finish in {timeout_s:.0f} s")
        time.sleep(2)


def run_jobs(py: Path, tmp: Path, weights: Path, licenses: set[str], territory: bool, out: Path) -> list[dict]:
    """Against the program installed at `py`: set up what each required build needs and make one real result with it."""
    data = tmp / "data"
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV")}
    env.update(AHA_OPEN_DATA_DIR=str(data), HF_HUB_OFFLINE="1")  # a download here would mean the weights were not recognised
    cli = [str(py), "-m", "aha_open.cli"]

    def run(*args: str) -> None:
        done = subprocess.run([*cli, *args], env=env, cwd=tmp, capture_output=True, text=True, timeout=180, check=False)
        if done.returncode:
            raise b.BuildError(f"aha-open {' '.join(args)} failed: {done.stdout}{done.stderr}")

    if not (data / "config.toml").exists():
        run("init", "--profile", "local")
    for folder in sorted({model for model, _ in b.REQUIRED_JOBS}):
        if not (weights / folder).is_dir():
            raise b.BuildError(f"{weights / folder} is not there: the gate needs the weights already downloaded")
        link_weights(weights / folder, data / "models" / folder)
    port = b._free_port()
    cfg = data / "config.toml"
    cfg.write_text(re.sub(r"(?m)^listen\s*=.*$", f'listen = "127.0.0.1:{port}"', cfg.read_text(encoding="utf-8")), encoding="utf-8")
    run("start", "--no-open")
    results = []
    try:
        base = f"http://127.0.0.1:{port}"
        token = (data / "local_client.token").read_text(encoding="utf-8").strip()

        def get(path: str) -> dict:
            return json.loads(call(base, token, "GET", path))

        def entry(model: str) -> dict:
            return next(m for m in get("/admin/models")["models"] if m["id"] == model)

        for model, variant in b.REQUIRED_JOBS:
            say(f"== {model} / {variant}")
            listed = entry(model)
            if not next(v for v in listed["variants"] if v["name"] == variant)["installed"]:
                raise b.BuildError(f"the installed program does not see the weights of {model} / {variant}")
            call(base, token, "POST", f"/admin/models/{model}/use", {"variant": variant})
            listed = entry(model)
            if listed["state"] != "ready":
                if not listed["engine_missing"]:
                    raise b.BuildError(f"{model} is {listed['state']}: {listed['state_detail']}")
                key = listed["license"]["key"]
                if key not in licenses or (listed["requires_territory_confirmation"] and not territory):
                    raise b.BuildError(f"{model}: pass --accept-license {key} (and --confirm-territory where its licence needs it)")
                call(base, token, "POST", f"/admin/models/{model}/install",
                     {"accept_license": key, "variant": variant, "confirm_territory": territory})
                state = wait_for("engine setup", lambda model=model: get(f"/admin/models/{model}/install"),
                                 lambda s: s["state"] != "running", 3600)
                if state["state"] != "succeeded":
                    raise b.BuildError(f"engine setup of {model} {state['state']}: {state['error']}")
                listed = entry(model)
                if listed["state"] != "ready":
                    raise b.BuildError(f"after the setup {model} is {listed['state']}: {listed['state_detail']}")

            task = next(m for m in get("/v1/manifest")["models"] if m["id"] == model)["tasks"][0]
            started = time.monotonic()
            job_id = json.loads(call(base, token, "POST", "/v1/jobs",
                                     {"task": task, "model": model, "prompt": PROMPT, "params": REQUESTS[task]}))["id"]
            job = wait_for("job", lambda job_id=job_id: get(f"/v1/jobs/{job_id}"),
                           lambda s: s["status"] in ("succeeded", "failed", "cancelled"), 1800)
            if job["status"] != "succeeded":
                raise b.BuildError(f"{model} / {variant}: the job {job['status']}: {job.get('error')}")
            result = call(base, token, "GET", f"/v1/jobs/{job_id}/result?index=0")
            size = job["results"][0]["width"], job["results"][0]["height"]
            out.mkdir(parents=True, exist_ok=True)
            kept = out / f"{model}-{variant}.{'png' if task == 'image.generate' else 'mp4'}"
            kept.write_bytes(result)
            (check_png if task == "image.generate" else check_video)(result, *size)
            results.append({"model": model, "variant": variant, "seconds": round(time.monotonic() - started),
                            "bytes": len(result), "sha256": hashlib.sha256(result).hexdigest()})
            say(f"  ok: {size[0]}x{size[1]}, {results[-1]['seconds']} s, kept at {kept}")
    except b.BuildError:
        out.mkdir(parents=True, exist_ok=True)
        for log in sorted((data / "logs").glob("*.log")):  # the temporary folder goes away: keep what explains the failure
            shutil.copy2(log, out / log.name)
        say(f"logs kept in {out}")
        raise
    finally:
        subprocess.run([*cli, "stop"], env=env, cwd=tmp, capture_output=True, timeout=120, check=False)
    return results


def install_exe(installer: Path, tmp: Path) -> Path:
    b.refuse_if_installed()
    app = tmp / "app"
    done = subprocess.run([str(installer), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/SP-", f"/DIR={app}"],
                          env={**os.environ, "AHA_OPEN_DATA_DIR": str(tmp / "data")}, timeout=900, check=False)
    if done.returncode:
        raise b.BuildError(f"the installer exited {done.returncode}: {b._install_log_tail(app)}")
    return app / "venv" / "Scripts" / "python.exe"


def install_wheel(tmp: Path) -> Path:
    """The same two steps the installer takes (aha-open.iss), without registering anything in Windows."""
    uv = str(b.find_uv())
    pins = tmp / "pins.txt"
    b.export_pins(pins)
    for cmd in ([uv, "build", "--wheel", "--out-dir", str(tmp / "wheel")], [uv, "venv", "--python", "3.12", str(tmp / "app")]):
        if subprocess.run(cmd, cwd=ROOT, check=False).returncode:
            raise b.BuildError(f"uv {cmd[1]} failed")
    py = tmp / "app" / "Scripts" / "python.exe"
    wheel = next((tmp / "wheel").glob("*.whl"))
    install = [uv, "pip", "install", "--python", str(py), "--constraint", str(pins), f"{wheel}[download]"]
    if subprocess.run(install, cwd=tmp, check=False).returncode:
        raise b.BuildError("installing the wheel failed")
    shutil.copy2(uv, py.with_name("uv.exe"))
    return py


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--installer", type=Path, help="the .exe that is about to be released")
    what.add_argument("--from-wheel", action="store_true", help="developer run from the built wheel; does not count for a release")
    ap.add_argument("--weights", type=Path, required=True, help="a data folder's `models` directory that holds the weights")
    ap.add_argument("--accept-license", action="append", default=[], help="repeat for each licence you accept")
    ap.add_argument("--confirm-territory", action="store_true", help="you are not in a territory a licence excludes")
    ap.add_argument("--keep", action="store_true", help="keep the temporary folder")
    args = ap.parse_args(argv)

    tmp = Path(tempfile.mkdtemp(prefix="aha-open-release-smoke-"))
    out = b.DIST / "release-smoke"
    installed = False
    try:
        if args.installer:
            installer = args.installer.resolve()
            (b.DIST / b.SMOKE_RECORD).unlink(missing_ok=True)  # an older pass must not vouch for this run
            py = install_exe(installer, tmp)
            installed = True
        else:
            py = install_wheel(tmp)
        results = run_jobs(py, tmp, args.weights.resolve(), set(args.accept_license), args.confirm_territory, out)
        if args.installer:
            record = {"version": b.read_version(), "installer": installer.name,
                      "installer_sha256": hashlib.sha256(installer.read_bytes()).hexdigest(),
                      "passed_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "jobs": results}
            b.DIST.mkdir(exist_ok=True)
            (b.DIST / b.SMOKE_RECORD).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        say(f"RELEASE SMOKE OK ({len(results)} real jobs). Look at the results in {out} before releasing.")
        if not args.installer:
            say("This was a --from-wheel run: it does not count for a release.")
        return 0
    except b.BuildError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    finally:
        if installed:
            uninstaller = next((tmp / "app").glob("unins*.exe"), None)
            if uninstaller:
                subprocess.run([str(uninstaller), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"],
                               env={**os.environ, "AHA_OPEN_DATA_DIR": str(tmp / "data")}, timeout=300, check=False)
        if args.keep:
            say(f"left for inspection: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
