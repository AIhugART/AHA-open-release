"""The Windows installer's guards (windows_installer/). They run without Inno Setup and without the internet.

What each one is for is a thing that went wrong, or nearly: a tag that named an artifact reporting another version (AHA Studio),
a `.pth` Python path file taken for model weights, a failed install that exited 0 and stayed registered (observed here on
2026-11-24), and a network failure explained with a sentence about something else. `tests/fixtures/installer/*.log` are the real
`install.log` files of two real failed installs.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ISS_RAW = (ROOT / "windows_installer" / "aha-open.iss").read_text(encoding="utf-8")
# No parser exists for .iss, so comments are removed before scanning: a sentence in a comment must not satisfy or break a check.
ISS = "\n".join(ln for ln in re.sub(r"\{\s[^{}]*\}", "", ISS_RAW, flags=re.DOTALL).splitlines() if not ln.lstrip().startswith(";"))
FIXTURES = Path(__file__).parent / "fixtures" / "installer"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_installer", ROOT / "windows_installer" / "build_installer.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_installer"] = module
    spec.loader.exec_module(module)
    return module


b = load_builder()


# ---------------------------------------------------------------------------------------------- one version, everywhere


@pytest.mark.parametrize(("version", "tag"), [("0.1.0.dev0", "v0.1.0-dev0"), ("0.1.0", "v0.1.0"), ("1.2.3.dev12", "v1.2.3-dev12")])
def test_the_tag_is_derived_from_the_one_version(version, tag):
    assert b.tag_for(version) == tag


def test_a_tag_that_does_not_match_the_tree_is_refused():
    version = b.read_version()
    assert b.check_release_tag(b.tag_for(version)).startswith("OK")
    for wrong in ("v9.9.9", version, "v" + version, "v0.1.0-dev1"):
        if wrong == b.tag_for(version):
            continue
        with pytest.raises(b.BuildError, match="REFUSED"):
            b.check_release_tag(wrong)


def test_the_installer_file_name_carries_the_version_it_installs():
    assert b.installer_name("0.1.0") == "AHA-Open-Setup-0.1.0.exe"
    assert b.wheel_name("0.1.0.dev0") == "aha_open-0.1.0.dev0-py3-none-any.whl"


# ---------------------------------------------------------------------------------------------- the wheel


def make_wheel(path: Path, version="0.1.0.dev0", requires=(), extra: dict[str, int] | None = None, drop=()) -> Path:
    members = {
        "aha_open/ui/index.html": b"<html>", "aha_open/launcher.py": b"x",
        **{f"aha_open/workers/{name}": b"x" for name in b.worker_files()},
    }
    meta = f"Metadata-Version: 2.1\nName: aha-open\nVersion: {version}\n" + "".join(f"Requires-Dist: {r}\n" for r in requires)
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            if name not in drop:
                zf.writestr(name, data)
        zf.writestr("aha_open-0.1.0.dist-info/METADATA", meta)
        for name, size in (extra or {}).items():
            zf.writestr(name, b"\0" * size)
    return path


def test_a_clean_wheel_has_no_problems(tmp_path):
    assert b.check_wheel(make_wheel(tmp_path / "a.whl", requires=("fastapi>=0.115", "uvicorn>=0.30")), "0.1.0.dev0") == []


def test_a_wheel_of_another_version_is_a_problem(tmp_path):
    assert any("version" in p for p in b.check_wheel(make_wheel(tmp_path / "a.whl", version="0.0.9"), "0.1.0.dev0"))


@pytest.mark.parametrize("dependency", ["torch>=2.10", "torch", "diffusers==0.35", "transformers>=5", "bitsandbytes", "safetensors>=0.4"])
def test_a_gateway_wheel_that_needs_torch_or_its_family_is_a_problem(tmp_path, dependency):
    problems = b.check_wheel(make_wheel(tmp_path / "a.whl", requires=(dependency,)), "0.1.0.dev0")
    assert any("torch-free" in p for p in problems), problems


def test_a_wheel_without_the_web_page_or_the_launcher_is_a_problem(tmp_path):
    problems = b.check_wheel(make_wheel(tmp_path / "a.whl", drop=("aha_open/ui/index.html", "aha_open/launcher.py")), "0.1.0.dev0")
    assert any("index.html" in p for p in problems) and any("launcher.py" in p for p in problems)


def test_a_wheel_without_a_module_a_worker_imports_is_a_problem(tmp_path):
    missing = ("aha_open/workers/minimax_h3/h3_runner.py", "aha_open/workers/qwen_image/qwen_models.py")
    problems = b.check_wheel(make_wheel(tmp_path / "a.whl", drop=missing), "0.1.0.dev0")
    assert all(any(name in p for p in problems) for name in missing), problems


def test_nothing_from_an_earlier_build_can_reach_the_next_wheel(tmp_path):
    stale = tmp_path / "build" / "lib" / "aha_open" / "workers" / "qwen_image" / "deleted_module.py"
    stale.parent.mkdir(parents=True)
    stale.write_text("")
    (tmp_path / "build" / "installer").mkdir()
    b.remove_build_leftovers(tmp_path)
    assert not stale.exists() and (tmp_path / "build" / "installer").exists()


def test_weights_are_recognised_by_size_not_by_suffix_alone():
    assert not b.looks_like_weights("_virtualenv.pth", 31)  # a Python path file: the false positive the first smoke run hit
    assert not b.looks_like_weights("small.bin", 1024)
    assert b.looks_like_weights("model.safetensors", 11 * 2**20)
    assert b.looks_like_weights("Weights.PTH", 11 * 2**20)
    assert not b.looks_like_weights("model.txt", 11 * 2**20)


def test_a_wheel_that_carries_model_weights_is_a_problem(tmp_path):
    problems = b.check_wheel(make_wheel(tmp_path / "a.whl", extra={"aha_open/x/model.safetensors": 11 * 2**20}), "0.1.0.dev0")
    assert any("weights" in p for p in problems)


def test_the_real_wheel_built_from_this_tree_passes_every_check(tmp_path):
    uv = shutil.which("uv")
    if not uv:
        pytest.skip("uv is not installed")
    out = subprocess.run([uv, "build", "--wheel", "--out-dir", str(tmp_path)], cwd=ROOT, capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr
    wheel = tmp_path / b.wheel_name(b.read_version())
    assert wheel.is_file() and b.check_wheel(wheel, b.read_version()) == []


# ---------------------------------------------------------------------------------------------- the pins


PINS = "fastapi==0.165.0\nuvicorn==0.54.0\ncryptography==50.0.2\nhuggingface-hub==1.32.0\npydantic==2.14.4\nstarlette==1.10.0\n"


def test_pins_are_read_as_name_and_version_whatever_the_marker_or_spelling():
    got = b.parse_pins("Typing_Extensions==4.16.0 ; python_version < '3.13'\ncolorama==0.4.6 ; sys_platform == 'win32'\n# a comment\n-e .\n")
    assert got == {"typing-extensions": "4.16.0", "colorama": "0.4.6"}


def test_complete_pins_have_no_problems():
    assert b.check_pins(PINS) == []


@pytest.mark.parametrize("missing", ["fastapi", "uvicorn", "cryptography", "huggingface-hub", "pydantic", "starlette"])  # fixed: not b.REQUIRED_PINS
def test_a_dependency_without_a_pin_is_a_problem(missing):
    text = "\n".join(ln for ln in PINS.splitlines() if not ln.startswith(missing + "=="))
    assert any(missing in p for p in b.check_pins(text))


def test_pins_that_drag_torch_in_are_a_problem():
    assert any("torch-free" in p for p in b.check_pins(PINS + "torch==2.10.0\n"))


def test_the_pins_exported_from_the_lock_file_are_complete():
    if not shutil.which("uv"):
        pytest.skip("uv is not installed")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "pins.txt"
        b.export_pins(out)
        text = out.read_text(encoding="utf-8")
    assert b.check_pins(text) == []
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    hub = b.parse_pins(text)["huggingface-hub"]
    assert f'name = "huggingface-hub"\nversion = "{hub}"' in lock  # the version in uv.lock (what the tests ran against), not a fresh resolution


def test_the_installer_installs_under_the_pins_not_a_fresh_resolution():
    body = ISS.split("function PrepareToInstall", 1)[1].split("\nend;", 1)[0]
    assert "--constraint" in body and "Pins" in body
    line = next(ln for ln in ISS.splitlines() if ln.startswith("Source:") and "pins.txt" in ln)
    assert "dontcopy" in line


# ---------------------------------------------------------------------------------------------- the checksums


def test_the_sums_file_lists_exactly_this_releases_files_in_sha256sum_format(tmp_path):
    import hashlib

    exe, whl = tmp_path / "AHA-Open-Setup-1.exe", tmp_path / "aha_open-1-py3-none-any.whl"
    exe.write_bytes(b"installer")
    whl.write_bytes(b"wheel")
    sums = tmp_path / "SHA256SUMS"
    sums.write_text("0" * 64 + "  AHA-Open-Setup-0.exe\n", encoding="utf-8")  # a leftover of an older release
    b.write_sums(sums, [exe, whl])
    lines = sums.read_text(encoding="utf-8").splitlines()
    assert lines == [f"{hashlib.sha256(b'installer').hexdigest()}  {exe.name}", f"{hashlib.sha256(b'wheel').hexdigest()}  {whl.name}"]
    assert b"\r" not in sums.read_bytes()  # `sha256sum -c` is picky about line endings


# ---------------------------------------------------------------------------------------------- finding Inno Setup


def test_iscc_is_found_in_the_per_user_folder_even_when_it_is_not_on_the_path(tmp_path, monkeypatch):
    """winget installs Inno Setup per user and does not put it on PATH: 'iscc not found' does not mean 'not installed'."""
    home = tmp_path / "Programs" / "Inno Setup 6"
    home.mkdir(parents=True)
    (home / "ISCC.exe").write_bytes(b"x")
    monkeypatch.setattr(b.shutil, "which", lambda name: None)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert b.find_iscc() == home / "ISCC.exe"


def test_a_missing_inno_setup_says_how_to_install_it(tmp_path, monkeypatch):
    monkeypatch.setattr(b.shutil, "which", lambda name: None)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(b.Path, "is_file", lambda self: False)
    with pytest.raises(b.BuildError, match="winget install"):
        b.find_iscc()


# ---------------------------------------------------------------------------------------------- the Inno script


def test_the_app_id_never_changes():
    """AppId is Windows' upgrade key: another value gives every user two entries in Add/Remove Programs."""
    declared = re.search(r"^AppId=\{(\{[0-9A-F-]+\})$", ISS, re.MULTILINE)  # Inno writes "{{" for a literal "{"
    assert declared and declared.group(1) == b.APP_ID == "{56B3FE1F-7C37-4C1B-AF6E-9C1F28DF42B4}"


def test_the_install_is_per_user_and_asks_for_no_administrator():
    assert re.search(r"^PrivilegesRequired=lowest$", ISS, re.MULTILINE)
    assert "{localappdata}" in re.search(r"^DefaultDirName=(.+)$", ISS, re.MULTILINE).group(1)


def test_setup_does_not_hand_redirection_guard_to_what_it_starts():
    """Inno 6.7 turns Windows' RedirectionGuard on by default and children inherit it. uv's venvs reach their interpreter through a
    junction the user created, which a guarded process may not follow: the gateway started from [Run] could not spawn a worker
    ("uv trampoline failed to spawn Python child process", observed 2026-11-28), and on a machine without Python the setup's own
    `uv venv` gives such a venv."""
    setup = ISS.split("[Setup]", 1)[1].split("[Files]", 1)[0]
    assert re.search(r"^RedirectionGuard=no$", setup, re.MULTILINE)
    assert re.search(r"^PrivilegesRequired=lowest$", setup, re.MULTILINE)  # why turning it off costs little: Setup never elevates


def test_the_environment_is_built_before_anything_is_committed_not_after():
    """A failure raised after the files are copied is only logged by Setup, which exits 0 and stays registered."""
    assert "function PrepareToInstall" in ISS
    assert "ssPostInstall" not in ISS and "CurStepChanged" not in ISS
    body = ISS.split("function PrepareToInstall", 1)[1].split("\nend;", 1)[0]
    for step in ("python -I -m venv", "pip install", "init --profile local"):
        assert step in body, step
    assert "DelTree(Venv" in body.split("if Result <> ''", 1)[1]  # a failed install removes its half-built environment


def test_the_apps_environment_is_made_by_cpython_so_its_pythonw_has_no_console():
    """Where uv had to download Python (a machine without it), `uv venv` wrote a CONSOLE program named `pythonw.exe`, so every
    shortcut opened a terminal window (observed 2026-11-28; the machines checked before already had a Python, which hid it).
    The release gate checks the built program; this keeps the script from going back."""
    body = ISS.split("function PrepareToInstall", 1)[1].split("\nend;", 1)[0]
    assert "run --no-project --python 3.12 python -I -m venv --without-pip " in body
    assert "' venv --python" not in body


def test_uv_and_the_wheel_are_only_extracted_while_installing_not_copied_into_the_app():
    for source in ("uv.exe", "{#WheelName}"):
        line = next(ln for ln in ISS.splitlines() if ln.startswith("Source:") and source in ln)
        assert "dontcopy" in line and "DestDir" not in line, line


def test_the_installer_carries_no_model_weights():
    sources = "\n".join(ln for ln in ISS.splitlines() if ln.startswith("Source:"))
    assert not re.search(r"\.(safetensors|ckpt|gguf|onnx|pth?)\b", sources)


def test_the_uninstaller_stops_the_gateway_and_keeps_the_users_data_by_default():
    assert "cli stop --quiet" in ISS and "RunOnceId" in ISS
    box = ISS.split("SuppressibleMsgBox", 1)[1]
    assert "MB_DEFBUTTON2" in box and "IDNO" in box  # No is the default, and a silent uninstall answers No


def test_the_licence_texts_of_what_is_shipped_are_in_the_installer():
    assert "licenses" in ISS
    names = {p.name for p in (ROOT / "windows_installer" / "licenses").iterdir()}
    assert {"uv-LICENSE-MIT.txt", "uv-LICENSE-APACHE.txt"} <= names
    assert (ROOT / "windows_installer" / "licenses" / "uv-LICENSE-MIT.txt").read_text(encoding="utf-8").startswith("MIT License")


# ---------------------------------------------------------------------------------------------- explaining a failure


def network_markers() -> list[str]:
    """The markers Explain() in the .iss tests for the 'no internet' sentence: read from the script itself, not copied."""
    explain = ISS.split("function Explain", 1)[1].split("else if Has(Log, 'os error 112')", 1)[0]
    return re.findall(r"Has\(Log, '([^']+)'\)", explain)


@pytest.mark.parametrize("fixture", ["uv_offline.log", "uv_dead_proxy.log"])
def test_a_real_network_failure_is_explained_as_a_network_failure(fixture):
    log = (FIXTURES / fixture).read_text(encoding="utf-8").lower()
    assert any(marker in log for marker in network_markers()), fixture


def test_other_failures_are_not_explained_as_the_network():
    disk = "error: failed to write to cache: There is not enough space on the disk. (os error 112)"
    assert not any(marker in disk.lower() for marker in network_markers())
    assert not any(marker in "error: no matching distribution found for aha-open[download]==9.9.9".lower() for marker in network_markers())


def test_the_smoke_expects_exactly_the_models_the_web_page_can_install():
    """The smoke once still expected only Qwen after MiniMax H3 was added, and failed at the end of a build."""
    from aha_open.install import INSTALLABLE

    assert b.WEB_MODELS == [model_id for model_id, plan in INSTALLABLE.items() if plan.get("web", True)]
