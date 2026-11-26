"""The examples in docs/aha_open_api/examples are RUN against a real gateway (synthetic model, real sockets); the markdown
under docs/aha_open_api may only name /v1 routes that exist in the OpenAPI file.

An example that stops working when the API changes turns this file red; that is the whole point (plan 2026-11-24_api_docs_plan, R3).
Negative search: the client token is the needle - it must appear 0 times in anything an example prints.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import yaml
from conftest import Api, image_job

from aha_open.catalog import CATALOG, validate_job

ROOT = Path(__file__).resolve().parents[1]
API_DOCS = ROOT / "docs" / "aha_open_api"
EXAMPLES = API_DOCS / "examples"
PNG_SIG = b"\x89PNG\r\n\x1a\n"


def run(cmd: list[str], gw, *, token: str | None = "local", via_file: bool = False, cwd: Path, extra_env: dict | None = None,
        timeout: float = 120) -> subprocess.CompletedProcess:
    """Run an example the way a user would: only environment variables and a working directory, nothing from the test process."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("AHA_")}
    env["AHA_OPEN_URL"] = gw.base
    if via_file:
        env["AHA_OPEN_DATA_DIR"] = str(gw.cfg.data_dir)  # the client finds local_client.token by itself
    else:
        env["AHA_OPEN_TOKEN"] = gw.tokens["local"] if token == "local" else (token or "")
        env["AHA_OPEN_DATA_DIR"] = str(cwd / "no-such-data-dir")
    env.update(extra_env or {})
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == PNG_SIG, "the file is not a PNG"
    return struct.unpack(">II", data[16:24])


def py(script: str, *args: str) -> list[str]:
    return [sys.executable, str(EXAMPLES / "python" / script), *args]


def no_secret(done: subprocess.CompletedProcess, *secrets: str) -> None:
    for secret in secrets:
        assert secret and secret not in done.stdout + done.stderr, "an example printed the token"


# ---------------------------------------------------------------------------------------------------------- Python
def test_python_generate_finds_the_local_token_file_and_uses_the_first_declared_size(gw, tmp_path):
    done = run(py("generate.py", "a red square", "--out", "a.png"), gw, via_file=True, cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    assert png_size(tmp_path / "a.png") == (256, 256)  # the first size example-image declares: never stretched
    assert "license:" in done.stderr and "saved a.png: 256x256" in done.stderr
    no_secret(done, gw.tokens["local"])


def test_python_generate_with_explicit_size_seed_and_steps(gw, tmp_path):
    done = run(py("generate.py", "a blue circle", "--out", "b.png", "--width", "320", "--height", "180", "--seed", "5", "--steps", "6"),
               gw, cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    assert png_size(tmp_path / "b.png") == (320, 180) and "seed 5" in done.stderr
    no_secret(done, gw.tokens["local"])


def test_python_edit_with_a_reference_and_a_mask(gw, tmp_path):
    assert run(py("generate.py", "source", "--out", "src.png"), gw, cwd=tmp_path).returncode == 0
    done = run(py("edit.py", "make it blue", "--source", "src.png", "--reference", "src.png", "--mask", "src.png", "--out", "e.png"),
               gw, cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    assert png_size(tmp_path / "e.png") == (256, 256)
    no_secret(done, gw.tokens["local"])


def test_python_edit_refuses_a_non_image_before_uploading(gw, tmp_path):
    (tmp_path / "notes.txt").write_text("hello")
    done = run(py("edit.py", "x", "--source", "notes.txt"), gw, cwd=tmp_path)
    assert done.returncode != 0 and "does not look like an image" in (done.stdout + done.stderr)


def test_a_wrong_token_gets_a_sentence_a_person_can_act_on_and_never_echoes_it(gw, tmp_path):
    needle = "needle-token-" + "x" * 24
    done = run(py("generate.py", "x"), gw, token=needle, cwd=tmp_path)
    assert done.returncode == 1
    assert "unauthorized" in done.stderr and "did not accept the token" in done.stderr
    no_secret(done, needle, gw.tokens["local"])


def test_a_server_that_is_not_there_is_reported_as_unreachable(gw, tmp_path):
    done = run(py("generate.py", "x"), gw, cwd=tmp_path, extra_env={"AHA_OPEN_URL": "http://127.0.0.1:1"})
    assert done.returncode == 1 and "cannot reach" in done.stderr
    no_secret(done, gw.tokens["local"])


def test_a_busy_server_is_waited_out_not_hammered(make_gateway, tmp_path):
    """Two slow jobs fill this client's queue (limit 2); the example's own job gets `busy` and must wait, then succeed."""
    gw = make_gateway(max_queue_per_client=2)
    api = Api(gw.base, gw.tokens["local"])
    blockers = [api.job(image_job("p __slow")).json()["id"] for _ in range(2)]
    api.wait(blockers[0], states=("running",))
    threading.Timer(1.5, lambda: [api.post(f"/v1/jobs/{b}/cancel") for b in blockers]).start()
    done = run(py("generate.py", "after the queue clears", "--out", "w.png"), gw, cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    assert png_size(tmp_path / "w.png") == (256, 256)


# ------------------------------------------------------------------------------------------------------------ recipes
def test_recipe_transparent_says_why_when_no_model_can(gw, tmp_path):
    done = run(py("recipes.py", "transparent", "a red apple"), gw, cwd=tmp_path)
    assert done.returncode == 1 and "no ready model can produce transparency" in done.stderr and "example-image" in done.stderr


def test_recipe_loras_lists_what_the_operator_added_and_the_lora_recipe_uses_it(gw, tmp_path):
    from test_v1_loras import _install

    none = run(py("recipes.py", "loras"), gw, cwd=tmp_path)
    assert none.returncode == 0 and "offers no LoRA files" in none.stderr
    unknown = run(py("recipes.py", "lora", "a fox", "--lora", "ink"), gw, cwd=tmp_path)
    assert unknown.returncode == 1 and "does not offer a LoRA called 'ink'" in unknown.stderr
    _install(gw, "ink", display_name="Ink", default_scale=0.8, trigger="inkpaint")  # no restart: the manifest re-reads the folder
    listed = run(py("recipes.py", "loras"), gw, cwd=tmp_path)
    assert "example-image: ink" in listed.stdout and "inkpaint" in listed.stdout
    plain = run(py("generate.py", "a fox", "--seed", "1", "--out", "plain.png"), gw, cwd=tmp_path)
    styled = run(py("recipes.py", "lora", "a fox", "--lora", "ink", "--scale", "0.7", "--seed", "1", "--out", "styled.png"), gw, cwd=tmp_path)
    assert plain.returncode == styled.returncode == 0, styled.stderr
    assert (tmp_path / "plain.png").read_bytes() != (tmp_path / "styled.png").read_bytes()  # the LoRA reached the worker


def test_recipe_video_and_its_duration_check(gw, tmp_path):
    done = run(py("recipes.py", "video", "waves on a beach", "--duration", "2", "--out", "v.mp4"), gw, cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "v.mp4").stat().st_size > 0 and "2.0 s, 8 fps, audio: False" in done.stderr
    long = run(py("recipes.py", "video", "waves", "--duration", "99"), gw, cwd=tmp_path)
    assert long.returncode == 1 and "makes videos of 1 to 4 seconds" in long.stderr


def test_recipe_video_with_exact_sizes_picks_a_tier_by_name_defaults_to_the_smallest_and_names_what_is_on_offer(make_gateway, tmp_path):
    from aha_open.config import ModelConfig

    gw = make_gateway(models=[ModelConfig(id="example-video-sized", worker="fake")])  # the shape of what MiniMax H3 declares
    small = run(py("recipes.py", "video", "waves", "--out", "small.mp4"), gw, cwd=tmp_path)  # no --tier, no --duration
    assert small.returncode == 0, small.stderr
    assert "size: preview (64x32)" in small.stderr and "64x32, 2 s, 8 fps" in small.stderr  # the smallest, with the model's own duration
    big = run(py("recipes.py", "video", "waves", "--tier", "native", "--out", "big.mp4"), gw, cwd=tmp_path)
    assert big.returncode == 0 and "128x64, 2 s" in big.stderr, big.stderr
    wrong = run(py("recipes.py", "video", "waves", "--tier", "huge"), gw, cwd=tmp_path)
    assert wrong.returncode == 1 and "offers these sizes: native (128x64), preview (64x32)" in wrong.stderr
    fixed = run(py("recipes.py", "video", "waves", "--duration", "3"), gw, cwd=tmp_path)
    assert fixed.returncode == 1 and "makes videos of exactly 2 seconds" in fixed.stderr  # a model with one length says so


def test_recipe_cancel_ends_cancelled_and_never_has_a_result(gw, tmp_path):
    done = run(py("recipes.py", "cancel", "p __slow", "--after", "1"), gw, cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    assert "final status: cancelled; results: 0" in done.stderr and "job_not_ready" in done.stderr


# ----------------------------------------------------------------------------------------------------------------- Node
def _node_ready() -> bool:
    if shutil.which("node") is None:
        return False
    major = subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip().lstrip("v").split(".")[0]
    return major.isdigit() and int(major) >= 18


@pytest.mark.skipif(not _node_ready(), reason="needs Node 18 or newer")
def test_node_generate(gw, tmp_path):
    script = str(EXAMPLES / "node" / "generate.mjs")
    done = run(["node", script, "a red square", "n.png", "--seed", "3"], gw, via_file=True, cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    assert png_size(tmp_path / "n.png") == (256, 256) and "license:" in done.stderr and "seed 3" in done.stderr
    no_secret(done, gw.tokens["local"])
    needle = "needle-" + "n" * 20
    wrong = run(["node", script, "x"], gw, token=needle, cwd=tmp_path)
    assert wrong.returncode == 1 and "unauthorized" in wrong.stderr and "did not accept the token" in wrong.stderr
    no_secret(wrong, needle, gw.tokens["local"])
    away = run(["node", script, "x"], gw, cwd=tmp_path, extra_env={"AHA_OPEN_URL": "http://127.0.0.1:1"})
    assert away.returncode == 1 and "cannot reach" in away.stderr


# ------------------------------------------------------------------------------------------------------- PowerShell
@pytest.mark.skipif(shutil.which("powershell") is None and shutil.which("pwsh") is None, reason="PowerShell is not installed")
def test_powershell_generate(gw, tmp_path):
    shell = shutil.which("powershell") or shutil.which("pwsh")
    script = str(EXAMPLES / "powershell" / "generate.ps1")
    base = [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script]
    done = run([*base, "a red square", "-Out", "p.png"], gw, via_file=True, cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    assert png_size(tmp_path / "p.png") == (256, 256) and "license:" in done.stderr
    no_secret(done, gw.tokens["local"])
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    absolute = run([*base, "again", "-Out", str(elsewhere / "abs.png")], gw, via_file=True, cwd=tmp_path)  # an absolute -Out was the real bug
    assert absolute.returncode == 0, absolute.stderr
    assert png_size(elsewhere / "abs.png") == (256, 256)
    missing = run([*base, "again", "-Out", str(tmp_path / "no-folder" / "x.png")], gw, via_file=True, cwd=tmp_path)
    assert missing.returncode == 1 and "folder for -Out does not exist" in missing.stderr  # not mislabelled as a network error
    needle = "needle-" + "y" * 20
    wrong = run([*base, "x"], gw, token=needle, cwd=tmp_path)
    assert wrong.returncode == 1 and "unauthorized" in wrong.stderr
    no_secret(wrong, needle, gw.tokens["local"])


# ------------------------------------------------------------------------------------------------------------- curl
def _curl_ready() -> bool:
    if shutil.which("sh") is None or shutil.which("curl") is None:
        return False
    words = subprocess.run(["curl", "--version"], capture_output=True, text=True).stdout.split()
    parts = [int(p) for p in re.findall(r"\d+", words[1])[:2]] if len(words) > 1 else [0, 0]
    return parts >= [7, 76]  # --fail-with-body


@pytest.mark.skipif(not _curl_ready(), reason="needs sh and curl 7.76 or newer")
def test_curl_generate(gw, tmp_path):
    script = EXAMPLES / "curl" / "generate.sh"
    assert b"\r" not in script.read_bytes(), "a shell script with CRLF line endings does not run"
    env = {"AHA_OPEN_MODEL": "example-image", "AHA_OPEN_WIDTH": "320", "AHA_OPEN_HEIGHT": "180"}
    done = run(["sh", str(script), 'a "quoted" red square', "c.png"], gw, cwd=tmp_path, extra_env=env)
    assert done.returncode == 0, done.stderr
    assert png_size(tmp_path / "c.png") == (320, 180)
    no_secret(done, gw.tokens["local"])
    needle = "needle-" + "z" * 20
    wrong = run(["sh", str(script), "x"], gw, token=needle, cwd=tmp_path, extra_env=env)
    assert wrong.returncode == 1 and "unauthorized" in wrong.stderr
    no_secret(wrong, needle, gw.tokens["local"])


# --------------------------------------------------------------------------------------------- the markdown names real routes
def _norm(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "{id}", path.rstrip("/"))


def _openapi_routes() -> dict[str, set[str]]:
    spec = yaml.safe_load((ROOT / "schema" / "aha_provider_v1.openapi.yaml").read_text(encoding="utf-8"))
    return {_norm(p): {m.upper() for m in item if m in ("get", "post", "put", "delete")} for p, item in spec["paths"].items()}


def _api_docs() -> list[Path]:
    return sorted(API_DOCS.glob("*.md"))


def test_the_api_docs_exist() -> None:
    assert (API_DOCS / "README.md").is_file()


@pytest.mark.parametrize("doc", _api_docs(), ids=lambda p: p.name)
def test_every_v1_route_named_in_the_docs_exists_in_the_openapi_file(doc: Path) -> None:
    routes = _openapi_routes()
    mentioned = re.findall(r"`(?:(GET|POST|PUT|DELETE) )?(/v1/[A-Za-z0-9_{}/.-]*)(?:\?[^`]*)?`", doc.read_text(encoding="utf-8"))
    for method, path in mentioned:
        path = _norm(path)
        assert path in routes, f"{doc.name}: `{path}` is not a /v1 route in schema/aha_provider_v1.openapi.yaml"
        if method:
            assert method in routes[path], f"{doc.name}: {method} {path} is not allowed by the OpenAPI file ({sorted(routes[path])})"


def test_the_example_scripts_are_listed_in_the_readme() -> None:
    readme = (API_DOCS / "README.md").read_text(encoding="utf-8")
    scripts = sorted(p.relative_to(EXAMPLES).as_posix() for p in EXAMPLES.rglob("*") if p.suffix in (".py", ".ps1", ".sh", ".mjs"))
    assert scripts, "no example scripts found"
    for script in scripts:
        assert f"examples/{script}" in readme, f"examples/{script} is not mentioned in docs/aha_open_api/README.md"


def _codes(text: str, heading: str, stop: str) -> set[str]:
    section = text.split(heading, 1)[1].split(stop, 1)[0]
    return set(re.findall(r"^\| `([a-z_]+)` \|", section, flags=re.MULTILINE))


def test_the_troubleshooting_page_covers_exactly_the_error_codes_of_the_spec() -> None:
    """05's first table and docs/03_api_v1.md section 8 must list the same codes: a new code is not left undocumented, and none is invented."""
    spec = _codes((ROOT / "docs" / "03_api_v1.md").read_text(encoding="utf-8"), "## 8. Error codes", "\n## ")
    guide = _codes((API_DOCS / "05_errors_and_troubleshooting.md").read_text(encoding="utf-8"), "## 1. The API answered with an error", "\n## 2.")
    assert spec and guide
    assert guide == spec, f"only in the spec: {sorted(spec - guide)}; only in the guide: {sorted(guide - spec)}"


def _model_sections() -> dict[str, str]:
    """The sections of 07_models.md whose heading names a model id: {id: text of the section}."""
    text = (API_DOCS / "07_models.md").read_text(encoding="utf-8")
    found = {}
    for section in re.split(r"^## ", text, flags=re.MULTILINE)[1:]:
        for model_id in re.findall(r"`([a-z0-9.-]+)`", section.split("\n", 1)[0]):
            found[model_id] = section
    return found


def test_the_job_examples_in_the_model_reference_are_accepted_by_the_catalog() -> None:
    text = (API_DOCS / "07_models.md").read_text(encoding="utf-8")
    jobs = [json.loads(block) for block in re.findall(r"^```json\n(.*?)^```", text, flags=re.MULTILINE | re.DOTALL)]
    for job in jobs:
        params = job.get("params", {})
        loras = frozenset(item["id"] for item in params.get("loras", []))
        validate_job(CATALOG[job["model"]], job["task"], job.get("prompt"), params, job.get("inputs", []), loras)  # raises on a bad one
    assert {job["model"] for job in jobs} == {i for i, spec in CATALOG.items() if not spec.internal}


def test_the_sizes_and_variant_numbers_of_the_model_reference_match_the_catalog() -> None:
    sections = _model_sections()
    published = {i: spec for i, spec in CATALOG.items() if not spec.internal}
    assert set(sections) == set(published)

    def cell(value: object) -> str:
        return "—" if value is None else str(value)

    for model_id, spec in published.items():
        section = sections[model_id]
        declared = {f"{s['width']}×{s['height']}" for s in spec.capabilities["sizes"]}
        assert declared <= set(re.findall(r"\d+×\d+", section)), f"{model_id}: a declared size is missing from the guide"
        rows = {}
        for line in section.splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if line.startswith("| `") and cells[0].strip("`") in spec.variants and cells[1].isdigit():
                rows[cells[0].strip("`")] = cells[1:]
        assert set(rows) == set(spec.variants), f"{model_id}: the variant table does not list exactly the offered builds"
        for name, info in spec.variants.items():
            expected = [cell(info["disk_gib"]), cell(spec.min_vram_mib[name])]
            if "1k" in info["vram_gib"]:
                expected += [cell(info["vram_gib"]["1k"]), cell(info["seconds"]["1k"]), cell(info["ram_gib"])]
            else:
                expected += [cell(info["vram_gib"]["preview"]), cell(info["vram_gib"].get("native"))]
            assert rows[name] == expected, f"{model_id} {name}: the guide says {rows[name]}, the catalog {expected}"


def test_every_markdown_link_between_the_api_docs_points_at_a_file_that_exists() -> None:
    for doc in _api_docs():
        for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", doc.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            assert (doc.parent / target).exists(), f"{doc.name} links to {target}, which does not exist"

