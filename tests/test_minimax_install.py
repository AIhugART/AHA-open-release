"""MiniMax H3 install (plan 2026-11-24_minimax_h3_plan.md, M1): pinned parts, territory, sha256, the worker's own lock. Offline."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path

import pytest

from aha_open import install as install_mod
from aha_open.cli import main
from aha_open.config import Config, ModelConfig, load
from aha_open.errors import ApiError
from aha_open.hardware import recommend_variant, recommend_variant_for
from aha_open.catalog import CATALOG, variant_installed, weights_dir
from aha_open.install import INSTALLABLE, _progress_lines_class, _verify_parts
from aha_open.model_install import InstallManager

ROOT = Path(__file__).resolve().parents[1]
H3 = "minimax-h3-fl2va"


def _init(tmp_path: Path) -> None:
    assert main(["--data-dir", str(tmp_path), "init"]) == 0


# ---------------------------------------------------------------------------------------------- the plan itself


@pytest.mark.parametrize("model", ["minimax-h3-fl2va", "minimax-h3-ref2va"])
def test_every_part_is_pinned_to_a_revision_and_every_large_file_to_a_size_and_sha256(model):
    plan = INSTALLABLE[model]
    assert plan["license_key"] == "minimax-h3" and plan["territory"] is True
    assert list(plan["variants"]) == list(CATALOG[model].variants)  # what the catalog offers is exactly what can be installed
    for name, variant in plan["variants"].items():
        parts = variant["parts"]
        repos = [p["repo"] for p in parts]
        assert repos == ["MiniMaxAI/MiniMax-H3", "Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot", "Comfy-Org/MiniMax-H3"]
        files = []
        for part in parts:
            assert re.fullmatch(r"[0-9a-f]{40}", part["revision"]), part
            for file, (size, digest) in part["files"].items():
                assert size > 0 and re.fullmatch(r"[0-9a-f]{64}", digest), file
                files.append(f"{part['subdir']}/{file}")
        assert len(files) == 4  # transformer + 2 VAEs (Abiray) + text encoder (Comfy-Org)
        assert CATALOG[model].variants[name]["files"][0] in files  # the worker loads the very file the install fetches
        total_gib = sum(s for p in parts for s, _ in p["files"].values()) / 2**30
        assert total_gib - 1 < variant["disk_gib"] < total_gib + 2, name


def test_the_official_part_brings_model_index_json_which_marks_the_weights_as_present():
    official = INSTALLABLE[H3]["variants"]["mixed"]["parts"][0]
    assert official["subdir"] == "" and "model_index.json" in official["patterns"] and not official["files"]


# ---------------------------------------------------------------------------------------------- territory and licence


def test_install_refuses_without_the_territory_confirmation_and_says_which_territories(tmp_path, capsys):
    _init(tmp_path)
    before = (tmp_path / "config.toml").read_text(encoding="utf-8")
    capsys.readouterr()
    code = main(["--data-dir", str(tmp_path), "models", "install", H3, "--accept-license", "minimax-h3"])
    out = capsys.readouterr().out
    assert code == 2
    assert "EU, GB, KR, US" in out and "--confirm-territory" in out
    assert (tmp_path / "config.toml").read_text(encoding="utf-8") == before  # nothing installed or written


# ---------------------------------------------------------------------------------------------- sha256 check


def _fake_part(tmp_path: Path, payload: bytes) -> list[dict]:
    return [{"repo": "x/y", "revision": "0" * 40, "subdir": "part", "patterns": [],
             "files": {"w.safetensors": (len(payload), hashlib.sha256(payload).hexdigest())}}]


def test_verify_accepts_the_pinned_file(tmp_path):
    payload = b"pinned bytes"
    parts = _fake_part(tmp_path, payload)
    (tmp_path / "part").mkdir()
    (tmp_path / "part" / "w.safetensors").write_bytes(payload)
    assert _verify_parts(parts, tmp_path) == 0


@pytest.mark.parametrize("found", [b"pinned bytez", b"short"])  # same size, other bytes / other size
def test_verify_removes_a_file_that_is_not_the_pinned_one(tmp_path, capsys, found):
    parts = _fake_part(tmp_path, b"pinned bytes")
    (tmp_path / "part").mkdir()
    bad = tmp_path / "part" / "w.safetensors"
    bad.write_bytes(found)
    assert _verify_parts(parts, tmp_path) == 2
    assert not bad.exists()  # removed, so the next run downloads it again instead of trusting it
    assert "run the install again" in capsys.readouterr().out


def test_verify_reports_a_missing_file(tmp_path, capsys):
    assert _verify_parts(_fake_part(tmp_path, b"x"), tmp_path) == 2
    assert "missing" in capsys.readouterr().out


def test_install_from_a_folder_checks_the_hashes_and_records_every_part(tmp_path, monkeypatch):
    _init(tmp_path)
    weights = tmp_path / "h3"
    (weights / "abiray").mkdir(parents=True)
    (weights / "model_index.json").write_text("{}", encoding="utf-8")
    payload = b"a tiny stand-in for 15 GB"
    (weights / "abiray" / "w.safetensors").write_bytes(payload)
    fake = {**INSTALLABLE[H3], "variants": {"mixed": {"disk_gib": 1, "parts": [
        {"repo": "MiniMaxAI/MiniMax-H3", "revision": "1" * 40, "subdir": "", "patterns": ["model_index.json"], "files": {}},
        {"repo": "Abiray/x", "revision": "2" * 40, "subdir": "abiray", "patterns": [],
         "files": {"w.safetensors": (len(payload), hashlib.sha256(payload).hexdigest())}},
    ]}}}
    monkeypatch.setitem(install_mod.INSTALLABLE, H3, fake)
    argv = ["--data-dir", str(tmp_path), "models", "install", H3, "--accept-license", "minimax-h3", "--confirm-territory",
            "--from-dir", str(weights), "--skip-env"]
    assert main(argv) == 0
    record = json.loads((weights / "install.json").read_text(encoding="utf-8"))
    assert record["variant"] == "mixed" and [p["repo"] for p in record["parts"]] == ["MiniMaxAI/MiniMax-H3", "Abiray/x"]
    mc = load(tmp_path).model(H3)
    assert mc is not None and mc.variant == "mixed" and mc.enabled

    (weights / "abiray" / "w.safetensors").write_bytes(b"a tiny stand-in for 15 GX")  # same size, other bytes
    (weights / "install.json").unlink()
    assert main(argv) == 2 and not (weights / "install.json").exists()  # never vouches for a file that is not the pinned one


# ---------------------------------------------------------------------------------------------- variant and web page


def test_variant_recommendation_for_minimax_and_unchanged_for_qwen():
    assert recommend_variant_for(H3, 24564, 32.0)[0] == "mixed"  # the reference RTX 3090 Ti
    assert recommend_variant_for(H3, 16384, 32.0)[0] == "nvfp4"  # 16 GB: only the FP4 build ran under that cap
    assert recommend_variant_for(H3, 12288, 32.0)[0] is None  # below 16 GB MiniMax H3 is not offered
    assert recommend_variant_for(H3, None, 32.0)[0] is None
    # Qwen picks among its four variants by measured fit: the best-quality one that fits (plan D8), nothing below the smallest peak
    assert recommend_variant_for("qwen-image-2.1", 24564, 32.0)[0] == "int8-convrot"
    assert recommend_variant_for("qwen-image-2.1", 8192, 32.0)[0] is None and recommend_variant_for("qwen-image-2.1", None, 32.0)[0] is None
    assert recommend_variant(24564, 32.0)[0] == "int8"  # the older single-variant rule still answers as it did


def test_the_web_download_panel_offers_minimax_but_never_without_the_territory_confirmation(tmp_path):
    """It used to be kept off the page because its territory question was not built; now the page asks it, and the server enforces it."""
    _init(tmp_path)
    manager = InstallManager(load(tmp_path))
    assert H3 in [m["id"] for m in manager.models()]
    for confirm in (False, None, "yes", 1):  # only the boolean true counts
        with pytest.raises(ApiError) as err:
            manager.start(H3, "minimax-h3", confirm)
        assert err.value.status == 400 and "territory has not been confirmed" in err.value.message
    with pytest.raises(ApiError) as err:  # a model that is not installable from here at all stays a 404
        manager.start("no-such-model", "x", True)
    assert err.value.status == 404


def test_progress_of_a_multi_part_install_is_reported_over_the_whole(capsys):
    from aha_open.install import PROGRESS_PREFIX

    cls = _progress_lines_class(offset=1000, grand_total=5000)
    cls._last = 0.0
    bar = cls(total=200, unit="B", desc="Downloading bytes")
    bar.update(50)
    bar.close()
    assert f"{PROGRESS_PREFIX} 1050 5000" in capsys.readouterr().out


def test_the_progress_total_counts_only_what_is_still_to_download(tmp_path, monkeypatch):
    """Seen live: fetching a second build showed "0.1 of about 42.4 GB" while only its 12.5 GB transformer was missing."""
    (tmp_path / "here.bin").write_bytes(b"x" * 5)
    parts = [{"repo": "a/b", "revision": "r", "subdir": "", "patterns": [], "files": {"here.bin": (5, "0" * 64), "new.bin": (7, "0" * 64)}},
             {"repo": "c/d", "revision": "r", "subdir": "", "patterns": [], "files": {"other.bin": (3, "0" * 64)}}]
    seen = []
    monkeypatch.setenv("AHA_OPEN_PROGRESS", "lines")
    monkeypatch.setattr(install_mod, "_progress_lines_class", lambda offset=0, grand_total=None: seen.append((offset, grand_total)))
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: None)
    assert install_mod._download_parts(parts, tmp_path) == 0
    assert seen == [(0, 10), (7, 10)]


# ---------------------------------------------------------------------------------------------- the worker's own environment


def _lock_packages() -> dict[str, str]:
    lock = tomllib.loads((ROOT / "workers" / "minimax_h3" / "uv.lock").read_text(encoding="utf-8"))
    return {p["name"]: p.get("version", "") for p in lock["package"]}


def test_the_worker_lock_has_comfy_kitchen_and_never_comfyui():
    packages = _lock_packages()
    assert packages.get("comfy-kitchen") == "0.2.42"
    assert "av" in packages and "torch" in packages and "diffusers" in packages
    assert not [name for name in packages if "comfyui" in name.lower()]  # the GPL application must never enter this environment


def test_every_dependency_of_the_worker_is_pinned_exactly():
    project = tomllib.loads((ROOT / "workers" / "minimax_h3" / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    for dep in project["dependencies"]:
        assert "==" in dep or re.search(r"@ git\+https://\S+@[0-9a-f]{40}$", dep), f"not pinned: {dep}"


# ---------------------------------------------------------------------------------------------- no test may download


def test_the_guard_makes_a_download_inside_a_test_fail_at_once():
    """A sabotage of the territory gate once let a test download 33 GB of real weights; the plugin turns that into a failure."""
    import huggingface_hub

    with pytest.raises(AssertionError, match="tried to download from the Hub"):
        huggingface_hub.snapshot_download(repo_id="x/y")
    with pytest.raises(AssertionError, match="tried to download from the Hub"):
        huggingface_hub.hf_hub_download(repo_id="x/y", filename="z")


def test_the_plugin_is_not_a_second_conftest():
    assert not (ROOT / "tests" / "conftest.py").exists()  # conformance/ imports `conftest` by name: there must be only one


# ---------------------------------------------------------------------------------------------- several builds, two models, one folder


def _install_on_disk(tmp_path, *builds):
    """The folder `models install` leaves, with empty stand-ins: ``builds`` = (model, variant) pairs."""
    cfg = Config(data_dir=tmp_path)
    for model, variant in builds:
        wdir = weights_dir(cfg, ModelConfig(id=model, worker="minimax_h3", variant=variant))
        for rel in ("model_index.json", "install.json", "transformer_ref/config.json", "comfyorg/text_encoders/encoder.safetensors",
                    *CATALOG[model].variants[variant]["files"]):
            (wdir / rel).parent.mkdir(parents=True, exist_ok=True)
            (wdir / rel).write_text("{}")
    return cfg


def test_deleting_the_reference_model_keeps_the_text_model_that_shares_its_folder(tmp_path):
    text, ref = ("minimax-h3-fl2va", "mixed"), ("minimax-h3-ref2va", "mixed")
    cfg = _install_on_disk(tmp_path, text, ref)
    done, message = install_mod.remove_variant(cfg, *ref, include_shared=True)  # "delete everything of this one"
    assert done, message
    assert variant_installed(cfg, CATALOG[text[0]], text[1]), "removing Ref2VA took the text model with it"
    assert not variant_installed(cfg, CATALOG[ref[0]], ref[1])


def test_deleting_one_build_keeps_the_others_and_the_shared_files(tmp_path):
    cfg = _install_on_disk(tmp_path, ("minimax-h3-fl2va", "mixed"), ("minimax-h3-fl2va", "nvfp4"))
    done, message = install_mod.remove_variant(cfg, "minimax-h3-fl2va", "nvfp4", include_shared=True)
    assert done, message
    assert variant_installed(cfg, CATALOG["minimax-h3-fl2va"], "mixed")
    encoder = weights_dir(cfg, ModelConfig(id="minimax-h3-fl2va", worker="minimax_h3", variant="mixed")) / "comfyorg"
    assert encoder.is_dir()


def test_the_last_build_with_include_shared_removes_the_folder(tmp_path):
    cfg = _install_on_disk(tmp_path, ("minimax-h3-fl2va", "mixed"))
    done, message = install_mod.remove_variant(cfg, "minimax-h3-fl2va", "mixed", include_shared=True)
    assert done, message
    assert not (tmp_path / "models" / "minimax-h3-fl2va" / "mixed").exists()


def test_a_second_build_only_needs_its_own_transformer(tmp_path):
    # the real part lists with sizes of a few bytes: NTFS allocates a "sparse" 27 GB file for real
    real = INSTALLABLE["minimax-h3-fl2va"]["variants"]["nvfp4"]["parts"]
    parts = [{**p, "files": {name: (7 if "pruned" in name else 3, "0" * 64) for name in p["files"]}} for p in real]
    for part in parts:  # the shared files of an earlier install
        for name, (size, _) in part["files"].items():
            if "pruned" not in name:
                (tmp_path / part["subdir"] / name).parent.mkdir(parents=True, exist_ok=True)
                (tmp_path / part["subdir"] / name).write_bytes(b"x" * size)
    assert install_mod._missing_gib(parts, tmp_path) * 2**30 == 7  # only the transformer of this build


def test_a_build_that_was_not_measured_is_never_suggested(monkeypatch):
    spec = CATALOG["minimax-h3-fl2va"]
    monkeypatch.setitem(spec.min_vram_mib, "mixed", 18432)
    for name in ("int8", "nvfp4"):
        monkeypatch.delitem(spec.min_vram_mib, name, raising=False)
    assert recommend_variant_for(spec.id, 24564, 32.0)[0] == "mixed"
