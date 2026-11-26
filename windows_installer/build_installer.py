"""Build `dist/AHA-Open-Setup-<version>.exe` (docs/06_deployment.md, plan/2026-11-24_windows_installer_plan.md).

    python windows_installer/build_installer.py                  build (runs the tests first)
    python windows_installer/build_installer.py --smoke          build, then install into a temp folder, run, uninstall
    python windows_installer/build_installer.py --check-tag v0.1.0-dev0     only check that a tag matches this tree

Why this is more than one `ISCC` call: "the build is green" has meant "the script ran to the end" before (AHA Studio's release
playbook), not "the installer is right". So: the tree must be clean (a build packs the working tree, not a commit), the tests
must pass, the wheel must be torch-free and carry the web page, the version must be ONE number (wheel = file name = tag), and
`--smoke` installs the real file and starts the real gateway.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
STAGE = ROOT / "build" / "installer"
DIST = ROOT / "dist"
SKIP_GATE_ENV = "AHA_OPEN_BUILD_SKIP_QUALITY_GATE"

# A gateway wheel that needs any of these is no longer the small installer's gateway (CLAUDE.md: the gateway never imports torch).
FORBIDDEN_REQUIRES = ("torch", "torchvision", "diffusers", "transformers", "bitsandbytes", "accelerate", "safetensors")
WEIGHT_SUFFIXES = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx")
WEIGHT_MIN_BYTES = 10 * 2**20  # `_virtualenv.pth` is a Python path file, not weights: a suffix alone says nothing, size does


def looks_like_weights(name: str, size: int) -> bool:
    return name.lower().endswith(WEIGHT_SUFFIXES) and size >= WEIGHT_MIN_BYTES
EXPECTED_SIZE_MB = (10, 80)  # uv.exe is ~65 MiB; measured 2026-11-24 (plan section 3e)


class BuildError(Exception):
    pass


# ---------------------------------------------------------------------------------------------- one version, everywhere


def read_version() -> str:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]


def wheel_name(version: str) -> str:
    return f"aha_open-{version}-py3-none-any.whl"


def installer_name(version: str) -> str:
    return f"AHA-Open-Setup-{version}.exe"


def tag_for(version: str) -> str:
    """0.1.0.dev0 -> v0.1.0-dev0 (the pre-release tag already published); 0.1.0 -> v0.1.0."""
    return "v" + re.sub(r"\.dev(\d+)$", r"-dev\1", version)


def check_release_tag(tag: str, version: str | None = None) -> str:
    """Refuse a tag that does not match what this tree builds: a tag once named an artifact that reported another version."""
    version = version or read_version()
    if tag != tag_for(version):
        raise BuildError(f"REFUSED: tag {tag} does not match this tree, which builds {version} (tag {tag_for(version)})")
    return f"OK: {tag} matches version {version}"


# ---------------------------------------------------------------------------------------------- gates


def find_iscc() -> Path:
    """`ISCC.exe` is NOT on PATH after `winget install`: it lives in the per-user Programs folder."""
    candidates = [shutil.which("iscc")]
    local = os.environ.get("LOCALAPPDATA", "")
    candidates += [
        str(Path(local) / "Programs" / "Inno Setup 6" / "ISCC.exe") if local else None,
        r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        r"C:\Program Files\Inno Setup 6\ISCC.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise BuildError("Inno Setup 6 was not found (checked PATH and the usual folders). Install it: winget install JRSoftware.InnoSetup")


def assert_clean_tree() -> None:
    """A build packs the WORKING TREE: anything uncommitted under these paths would ship without being in any commit."""
    out = subprocess.run(["git", "status", "--porcelain", "--", "pyproject.toml", "uv.lock", "aha_open", "workers", "windows_installer",
                          "LICENSE", "NOTICE", "THIRD_PARTY_LICENSES.md"], cwd=ROOT, capture_output=True, text=True, check=False).stdout
    if out.strip():
        raise BuildError("uncommitted changes would ship in the installer (commit them, or pass --allow-dirty to try a build):\n" + out)


def quality_gate() -> None:
    if os.environ.get(SKIP_GATE_ENV) == "1":
        print(f"!! {SKIP_GATE_ENV}=1: the test gate is skipped on request", flush=True)
        return
    print("running the tests before building (set " + SKIP_GATE_ENV + "=1 to skip) ...", flush=True)
    done = subprocess.run([sys.executable, "-m", "pytest", "tests", "conformance", "-q", "-x", "-p", "no:cacheprovider"], cwd=ROOT, check=False)
    if done.returncode:
        raise BuildError("the tests are red, so no installer is built")


def check_wheel(wheel: Path, version: str) -> list[str]:
    """What the gateway wheel must and must not be. Returns the problems (empty = fine)."""
    problems: list[str] = []
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
        sizes = {i.filename: i.file_size for i in zf.infolist()}
        meta_name = next((n for n in names if n.endswith(".dist-info/METADATA")), None)
        if meta_name is None:
            return ["the wheel has no METADATA"]
        meta = zf.read(meta_name).decode("utf-8")
    declared = re.search(r"^Version: (.+)$", meta, re.MULTILINE)
    if not declared or declared.group(1).strip() != version:
        problems.append(f"the wheel says version {declared.group(1) if declared else '?'} but pyproject says {version}")
    for line in re.findall(r"^Requires-Dist: (.+)$", meta, re.MULTILINE):
        base = re.split(r"[ ;<>=!\[(~]", line, maxsplit=1)[0].lower().replace("_", "-")
        if base in FORBIDDEN_REQUIRES:
            problems.append(f"the wheel requires {base}: the gateway must stay torch-free")
    for needed in ("aha_open/ui/index.html", "aha_open/launcher.py", "aha_open/workers/qwen_image/worker.py", "aha_open/workers/qwen_image/uv.lock"):
        if needed not in names:
            problems.append(f"the wheel does not contain {needed}")
    for name in names:
        if looks_like_weights(name, sizes[name]):
            problems.append(f"the wheel contains what looks like model weights: {name}")
    return problems


REQUIRED_PINS = ("fastapi", "uvicorn", "cryptography", "huggingface-hub", "pydantic", "starlette")


def parse_pins(text: str) -> dict[str, str]:
    """`name==version ; marker` lines -> {normalised name: version}."""
    pins = {}
    for line in text.splitlines():
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)", line.strip())
        if match:
            pins[match.group(1).lower().replace("_", "-")] = match.group(2)
    return pins


def check_pins(text: str) -> list[str]:
    """The versions the installer asks for. Without them `uv pip install` resolves anew on the user's machine and picks whatever is
    newest that day, which may not be what the tests ran against (pyproject only says `fastapi>=0.115`, `huggingface-hub>=1.0`)."""
    pins = parse_pins(text)
    problems = [f"{name} is not pinned" for name in REQUIRED_PINS if name not in pins]
    problems += [f"the pins contain {name}: the gateway must stay torch-free" for name in pins if name in FORBIDDEN_REQUIRES]
    return problems


def export_pins(dest: Path) -> None:
    """Pin the gateway's dependencies to uv.lock (the versions the tests ran against)."""
    done = subprocess.run(["uv", "export", "--frozen", "--no-dev", "--extra", "download", "--no-emit-project", "--no-hashes",
                           "--format", "requirements-txt", "-o", str(dest)], cwd=ROOT, capture_output=True, text=True, check=False)
    if done.returncode:
        raise BuildError("uv export failed (is uv.lock out of date? run `uv lock`):\n" + done.stdout + done.stderr)


def check_stage(stage: Path) -> list[str]:
    """The folder ISCC packs, checked (a build that 'ran' is not a build that is right)."""
    problems = []
    for path in stage.rglob("*"):
        if path.is_file() and looks_like_weights(path.name, path.stat().st_size):
            problems.append(f"model weights in the installer's input: {path}")
    uv = stage / "uv.exe"
    if not uv.is_file() or uv.stat().st_size < 1_000_000:
        problems.append("uv.exe is missing or too small to be a real uv")
    return problems


# ---------------------------------------------------------------------------------------------- build


def find_uv() -> Path:
    found = shutil.which("uv")
    if not found:
        raise BuildError("uv was not found on PATH; it is what the installer ships and what builds the wheel")
    path = Path(found)
    if path.suffix.lower() != ".exe":
        raise BuildError(f"{path} is not a .exe")
    return path


def build(allow_dirty: bool) -> Path:
    version = read_version()
    if not allow_dirty:
        assert_clean_tree()
    quality_gate()
    iscc = find_iscc()
    uv = find_uv()
    if STAGE.exists():
        shutil.rmtree(STAGE)  # our own output folder, recreated from scratch every time
    (STAGE / "wheel").mkdir(parents=True)
    built = subprocess.run([str(uv), "build", "--wheel", "--out-dir", str(STAGE / "wheel")], cwd=ROOT, capture_output=True, text=True, check=False)
    if built.returncode:
        raise BuildError("uv build failed:\n" + built.stdout + built.stderr)
    wheel = STAGE / "wheel" / wheel_name(version)
    if not wheel.is_file():
        raise BuildError(f"expected {wheel.name} but the build produced: {[p.name for p in (STAGE / 'wheel').iterdir()]}")
    problems = check_wheel(wheel, version)
    shutil.copy2(uv, STAGE / "uv.exe")
    export_pins(STAGE / "pins.txt")
    problems += check_pins((STAGE / "pins.txt").read_text(encoding="utf-8"))
    problems += check_stage(STAGE)
    if problems:
        raise BuildError("the installer's input is wrong:\n  " + "\n  ".join(problems))
    uv_version = subprocess.run([str(uv), "--version"], capture_output=True, text=True, check=False).stdout.strip()
    print(f"input ok: {wheel.name} ({wheel.stat().st_size / 1024:.0f} KiB), {uv_version}", flush=True)

    DIST.mkdir(exist_ok=True)
    target = DIST / installer_name(version)
    target.unlink(missing_ok=True)
    done = subprocess.run([str(iscc), "/Qp", f"/DAppVersion={version}", f"/DWheelName={wheel.name}", f"/DStage={STAGE}", f"/DRoot={ROOT}",
                           f"/DOutDir={DIST}", str(HERE / "aha-open.iss")], capture_output=True, text=True, check=False)
    if done.returncode or not target.is_file():
        raise BuildError("ISCC failed:\n" + done.stdout + done.stderr)
    size_mb = target.stat().st_size / 2**20
    low, high = EXPECTED_SIZE_MB
    if not low <= size_mb <= high:
        raise BuildError(f"{target.name} is {size_mb:.1f} MiB; expected {low}-{high} MiB (uv.exe plus a small wheel). Investigate before shipping.")
    published_wheel = DIST / wheel.name
    shutil.copy2(wheel, published_wheel)  # the release carries the wheel too, byte for byte the one inside the installer
    write_sums(DIST / "SHA256SUMS", [target, published_wheel])
    print(f"built {target} ({size_mb:.1f} MiB)\n" + (DIST / "SHA256SUMS").read_text(encoding="utf-8"), end="", flush=True)
    return target


def write_sums(dest: Path, files: list[Path]) -> None:
    """`<sha256>  <file name>` lines in the format `sha256sum -c` reads. Exactly the files of THIS release, no leftovers of an older one."""
    lines = [f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}" for f in sorted(files, key=lambda p: p.name)]
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------------------------- smoke: install the real file


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(url: str, token: str | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


# What a fresh install must offer on the Download panel. Written out rather than read from the installed package, so a model that
# silently disappears fails the smoke; a test holds it equal to aha_open.install so that adding a model is noticed before a build.
WEB_MODELS = ["qwen-image-2.1", "minimax-h3-fl2va", "minimax-h3-ref2va"]

APP_ID = "{56B3FE1F-7C37-4C1B-AF6E-9C1F28DF42B4}"  # must equal AppId in aha-open.iss (a test checks it)


def _uninstall_key_present() -> bool:
    import winreg

    try:
        winreg.CloseKey(winreg.OpenKey(winreg.HKEY_CURRENT_USER, rf"Software\Microsoft\Windows\CurrentVersion\Uninstall\{APP_ID}_is1"))
        return True
    except FileNotFoundError:
        return False


def _install_log_tail(app: Path) -> str:
    path = app / "install.log"
    return path.read_text(encoding="utf-8", errors="replace")[-1500:] if path.exists() else ""


def smoke(installer: Path, extra_env: dict[str, str] | None = None, expect_failure: bool = False, keep: bool = False) -> str:
    """Install into a temp folder with a temp data folder, start the gateway, look at it, stop it, uninstall. Never touches the real ones.

    With expect_failure the install is supposed to fail (the environment makes the internet unreachable): it returns the install
    log so the caller can check the cause was named, and checks that nothing was left running or half-installed.
    """
    version = read_version()
    tmp = Path(tempfile.mkdtemp(prefix="aha-open-smoke-"))
    app, data = tmp / "app", tmp / "data"
    env = {**os.environ, "AHA_OPEN_DATA_DIR": str(data), **(extra_env or {})}
    py = app / "venv" / "Scripts" / "python.exe"
    log = tmp / "setup.log"
    ok = False
    try:
        t0 = time.monotonic()
        done = subprocess.run([str(installer), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/SP-", f"/DIR={app}", f"/LOG={log}"],
                              env=env, capture_output=True, text=True, timeout=900, check=False)
        print(f"install finished in {time.monotonic() - t0:.0f} s, exit {done.returncode}", flush=True)
        if expect_failure:
            if done.returncode == 0:
                raise BuildError("the install was supposed to fail but exited 0 (a failed install must never report success)")
            if (app / "venv").exists():
                raise BuildError("a failed install left a half-built environment behind")
            if _uninstall_key_present():
                raise BuildError("a failed install left an entry in Add/Remove Programs")
            tail = _install_log_tail(app)
            print(f"install failed as expected (exit {done.returncode}), nothing registered, no environment left; install.log tail:\n{tail[-400:]}", flush=True)
            ok = True
            return tail
        if done.returncode:
            raise BuildError(f"the installer exited {done.returncode}; see {log}\n{_install_log_tail(app)}")
        for need in (py, app / "venv" / "Scripts" / "uv.exe", app / "venv" / "Scripts" / "pythonw.exe",
                     app / "licenses" / "uv-LICENSE-MIT.txt", data / "config.toml"):
            if not need.exists():
                raise BuildError(f"after the install, {need} is missing")
        # Every command below runs from the temp folder: `python -m` puts the CURRENT folder first on sys.path, so run from a source
        # checkout it would exercise the checkout instead of what was installed (an earlier version of this smoke did exactly that).
        said = subprocess.run([str(py), "-m", "aha_open.cli", "--version"], env=env, cwd=tmp, capture_output=True, text=True, check=False).stdout.strip()
        if said != version:
            raise BuildError(f"the installed AHA open says {said!r}, expected {version!r}")
        origin = subprocess.run([str(py), "-c", "import aha_open; print(aha_open.__file__)"], env=env, cwd=tmp, capture_output=True, text=True, check=False).stdout.strip()
        if not Path(origin).resolve().is_relative_to((app / "venv").resolve()):
            raise BuildError(f"aha_open is imported from {origin!r}, not from the installed environment")
        site = app / "venv" / "Lib" / "site-packages"
        torchy = [p.name for p in site.iterdir() if p.name.lower().startswith(("torch", "diffusers", "transformers"))]
        if torchy:
            raise BuildError(f"the gateway environment holds {torchy}: it must be torch-free")
        weights = [p for p in app.rglob("*") if p.is_file() and looks_like_weights(p.name, p.stat().st_size)]
        if weights:
            raise BuildError(f"model weights were installed: {weights[:3]}")
        wanted = parse_pins((STAGE / "pins.txt").read_text(encoding="utf-8"))
        listed = subprocess.run([str(find_uv()), "pip", "list", "--python", str(py), "--format", "json"], capture_output=True, text=True, check=False)
        got = {p["name"].lower().replace("_", "-"): p["version"] for p in json.loads(listed.stdout)} if listed.returncode == 0 else {}
        wrong = {n: (got.get(n), v) for n, v in wanted.items() if n in got and got[n] != v}
        strangers = sorted(n for n in got if n not in wanted and n != "aha-open")
        if listed.returncode or wrong or strangers:
            raise BuildError(f"the installed packages are not the pinned ones: different {wrong}, unpinned {strangers}, list exit {listed.returncode}")
        venv_mib = sum(f.stat().st_size for f in (app / "venv").rglob("*") if f.is_file()) / 2**20
        print(f"installed {said}; the gateway environment has no torch ({venv_mib:.0f} MiB)", flush=True)

        port = _free_port()
        cfg = data / "config.toml"
        cfg.write_text(re.sub(r"(?m)^listen\s*=.*$", f'listen = "127.0.0.1:{port}"', cfg.read_text(encoding="utf-8")), encoding="utf-8")
        started = subprocess.run([str(py), "-m", "aha_open.cli", "start", "--no-open"], env=env, cwd=tmp, capture_output=True, text=True, timeout=120, check=False)
        if started.returncode:
            raise BuildError("start failed: " + started.stdout + started.stderr)
        # The running gateway must be the installed code: its own record of the digest of the source it loaded equals the digest of
        # the installed files (a gateway that loaded some other copy of aha_open would differ).
        running = json.loads((data / "run" / "gateway.json").read_text(encoding="utf-8"))["source_digest"]
        on_disk = subprocess.run([str(py), "-c", "from aha_open.runtime import source_digest; print(source_digest())"], env=env, cwd=tmp,
                                 capture_output=True, text=True, check=False).stdout.strip()
        if running != on_disk:
            raise BuildError("the running gateway is not the installed code (source digests differ)")
        base = f"http://127.0.0.1:{port}"
        status, _ = _get(base + "/v1/health")
        if status != 200:
            raise BuildError(f"/v1/health answered {status}")
        status, page = _get(base + "/ui/")
        if status != 200 or b"Download a Model" not in page:
            raise BuildError(f"/ui/ answered {status} or lacks the download panel")
        token = (data / "local_client.token").read_text(encoding="utf-8").strip()
        status, body = _get(base + "/admin/models", token)
        listed = json.loads(body) if status == 200 else {}
        if status != 200 or [m["id"] for m in listed.get("models", [])] != WEB_MODELS:
            raise BuildError(f"/admin/models answered {status}: {body[:200]!r}")
        offered = ", ".join(f"{m['id']} (state {m['state']}, blocker {m['blocker']['code'] if m['blocker'] else 'none'})" for m in listed["models"])
        print(f"gateway up on {port}: health 200, /ui/ has the download panel, /admin/models offers {offered}", flush=True)
        stopped = subprocess.run([str(py), "-m", "aha_open.cli", "stop"], env=env, cwd=tmp, capture_output=True, text=True, timeout=60, check=False)
        if stopped.returncode or "stopped" not in stopped.stdout:
            raise BuildError("stop failed: " + stopped.stdout + stopped.stderr)
        try:
            _get(base + "/v1/health")
        except urllib.error.URLError:
            pass
        else:
            raise BuildError("the gateway still answers after stop")

        uninstaller = next(app.glob("unins*.exe"), None)
        if uninstaller is None:
            raise BuildError("no uninstaller was installed")
        subprocess.run([str(uninstaller), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"], env=env, timeout=300, check=False)
        time.sleep(1)
        if (app / "venv").exists() or (app / "licenses").exists():
            raise BuildError(f"the uninstaller left program files behind in {app}")
        if _uninstall_key_present():
            raise BuildError("the uninstaller left its entry in Add/Remove Programs")
        if not (data / "config.toml").exists():
            raise BuildError("a silent uninstall deleted the user's data; it must keep it")
        print("uninstalled: program files gone, the data folder kept", flush=True)
        ok = True
        print("SMOKE OK", flush=True)
        return ""
    finally:
        if ok or not keep:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f"left for inspection: {tmp}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="after building, install it into a temp folder and run it")
    ap.add_argument("--smoke-only", metavar="EXE", help="smoke-test an existing installer, build nothing")
    ap.add_argument("--smoke-offline", metavar="EXE", help="install with the internet made unreachable: it must fail cleanly and say why")
    ap.add_argument("--allow-dirty", action="store_true", help="build although files are uncommitted (never for a release)")
    ap.add_argument("--check-tag", metavar="TAG", help="only check that TAG matches this tree's version")
    ap.add_argument("--keep", action="store_true", help="keep the smoke folder when the smoke fails")
    args = ap.parse_args(argv)
    try:
        if args.check_tag:
            print(check_release_tag(args.check_tag))
            return 0
        if args.smoke_only:
            smoke(Path(args.smoke_only), keep=args.keep)
            return 0
        if args.smoke_offline:
            cache = Path(tempfile.mkdtemp(prefix="aha-open-emptycache-"))
            try:
                tail = smoke(Path(args.smoke_offline), {"UV_OFFLINE": "1", "UV_CACHE_DIR": str(cache)}, expect_failure=True, keep=args.keep)
            finally:
                shutil.rmtree(cache, ignore_errors=True)
            print("SMOKE OFFLINE OK" if "network" in tail.lower() else "install.log does not mention the network")
            return 0
        exe = build(args.allow_dirty)
        if args.smoke:
            smoke(exe, keep=args.keep)
    except BuildError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
