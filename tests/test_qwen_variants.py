"""The four Qwen-Image-2.1 variants: what fits, what is installed, which one is used, what may be deleted (plan Q1, D2/D8/D9).

Offline: downloads are replaced by files made on the fly, shaped like the real layout (a shared `convrot` folder with one
transformer file per variant, and a folder of its own for the bitsandbytes build).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from aha_open import catalog, install
from aha_open.catalog import CATALOG, QWEN_VARIANTS, variant_installed, variant_public, weights_dir
from aha_open.cli import main
from aha_open.config import Config, ModelConfig, load
from aha_open.hardware import QWEN_PREFERENCE, fit_for, recommend_variant_for
from aha_open.install import INSTALLABLE, remove_variant, use_variant

QWEN = CATALOG["qwen-image-2.1"]
MODEL = "qwen-image-2.1"
CONVROT = ("int8-convrot", "mixed-p010", "mixed-p015")


def _cfg(tmp_path: Path, variant: str = "") -> Config:
    return Config(data_dir=tmp_path, models=[ModelConfig(MODEL, "qwen_image", variant=variant)])


def _place(cfg: Config, variant: str) -> Path:
    """Make the files that make ``variant`` installed, the way a finished download leaves them."""
    folder = weights_dir(cfg, ModelConfig(MODEL, "qwen_image", variant=variant))
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model_index.json").write_text("{}", encoding="utf-8")
    for name in QWEN.variants[variant]["files"]:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * 1024)
        path.with_name(path.name.removesuffix(".safetensors") + ".quant.json").write_text("{}", encoding="utf-8")
    (folder / "comfyorg").mkdir(exist_ok=True)
    (folder / "comfyorg" / "te.safetensors").write_bytes(b"y" * 2048)  # stands for the shared text encoder
    (folder / "install.json").write_text(json.dumps({"variants": [variant]}), encoding="utf-8")
    return folder


# ---------------------------------------------------------------------------------------------- the table agrees with the plan


@pytest.mark.parametrize("variant", CONVROT)
def test_the_catalog_and_the_install_plan_name_the_same_transformer_file(variant):
    """A drift here means a variant that downloads one file and then looks for another (and is reported "not installed")."""
    parts = INSTALLABLE[MODEL]["variants"][variant]["parts"]
    pinned = {f"{p['subdir']}/{name}" for p in parts for name in p["files"] if p["subdir"] == "transformers"}
    assert pinned == set(QWEN.variants[variant]["files"])


@pytest.mark.parametrize("variant", CONVROT)
def test_disk_gib_is_the_pinned_bytes_rounded_up(variant):
    parts = INSTALLABLE[MODEL]["variants"][variant]["parts"]
    total = sum(size for p in parts for size, _ in p["files"].values())
    assert QWEN.variants[variant]["disk_gib"] == INSTALLABLE[MODEL]["variants"][variant]["disk_gib"] >= math.ceil(total / 2**30)


def test_every_pinned_file_has_a_size_and_a_full_sha256_and_every_revision_is_a_commit():
    for variant in CONVROT:
        for part in INSTALLABLE[MODEL]["variants"][variant]["parts"]:
            assert len(part["revision"]) == 40 and set(part["revision"]) <= set("0123456789abcdef")
            for size, digest in part["files"].values():
                assert size > 0 and len(digest) == 64


def test_the_three_convrot_variants_share_one_folder_and_int8_has_its_own(tmp_path):
    cfg = Config(data_dir=tmp_path)
    folders = {v: weights_dir(cfg, ModelConfig(MODEL, "qwen_image", variant=v)) for v in QWEN.variants}
    assert len({folders[v] for v in CONVROT}) == 1 and folders["int8"] not in {folders[v] for v in CONVROT}
    assert folders["int8"].parent == folders["int8-convrot"].parent


def test_the_internal_fields_are_not_in_the_public_view():
    for info in QWEN_VARIANTS.values():
        assert not {"folder", "files", "runtime"} & set(variant_public(info))


# ---------------------------------------------------------------------------------------------- fit and recommendation (D8)


@pytest.mark.parametrize(
    ("gpu_mib", "expected"),
    [(24564, "fits"), (10240, "fits"), (9216, "tight"), (8192, "too_small"), (None, "too_small"), (0, "too_small")],
)
def test_fit_follows_the_measured_peak(gpu_mib, expected):
    assert fit_for({"vram_gib": {"1k": 8.2}}, gpu_mib) == expected


def test_a_number_that_was_not_measured_never_refuses():
    assert fit_for({"vram_gib": {"1k": None}}, 4096) == "unknown"
    assert fit_for({"vram_gib": {}}, None) == "unknown"


@pytest.mark.parametrize(("gpu_mib", "expected"), [(24564, "int8-convrot"), (12288, "int8-convrot"), (9216, "int8-convrot"), (8192, None), (None, None)])
def test_the_recommendation_is_the_best_quality_variant_that_fits(gpu_mib, expected):
    variant, reason = recommend_variant_for(MODEL, gpu_mib, 32.0)
    assert variant == expected and reason and reason[0].islower()


def test_the_preference_order_names_every_variant_once():
    assert sorted(QWEN_PREFERENCE) == sorted(QWEN.variants)


# ---------------------------------------------------------------------------------------------- installed or not


def test_a_variant_is_installed_only_when_its_own_file_is_there(tmp_path):
    cfg = _cfg(tmp_path)
    _place(cfg, "mixed-p015")
    assert variant_installed(cfg, QWEN, "mixed-p015")
    assert not variant_installed(cfg, QWEN, "int8-convrot")  # same folder, shared files present, but not ITS transformer
    assert not variant_installed(cfg, QWEN, "int8")


def test_an_unfinished_download_is_not_installed(tmp_path):
    cfg = _cfg(tmp_path)
    folder = _place(cfg, "int8-convrot")
    (folder / "install.json").unlink()
    marker = folder / catalog.HF_DOWNLOAD_STATE
    marker.mkdir(parents=True)
    (marker / "x.incomplete").write_bytes(b"")
    assert not variant_installed(cfg, QWEN, "int8-convrot")


def test_the_state_of_the_selected_variant_names_its_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "detect_gpus", lambda ttl_s=30.0: [{"index": 0, "name": "RTX", "memory_total_mib": 24564}])
    monkeypatch.setattr(catalog, "worker_python", lambda cfg, mc: Path("python"))  # the environment is not what this test is about
    monkeypatch.setattr(catalog, "WORKERS_DIR", Path(catalog.WORKERS_DIR))
    cfg = _cfg(tmp_path, "mixed-p010")
    _place(cfg, "mixed-p015")  # the shared files are there, the transformer of p010 is not
    state, detail = catalog.model_state(cfg, cfg.models[0], QWEN)
    assert state == "not_installed" and "mixed_p010" in detail


# ---------------------------------------------------------------------------------------------- use (D9)


def test_use_refuses_a_variant_that_is_not_installed_and_changes_nothing(tmp_path):
    cfg = _cfg(tmp_path, "int8")
    (tmp_path / "config.toml").write_text("before", encoding="utf-8")
    done, message = use_variant(cfg, MODEL, "mixed-p015")
    assert not done and "not installed" in message and "--variant mixed-p015" in message
    assert (tmp_path / "config.toml").read_text(encoding="utf-8") == "before" and cfg.models[0].variant == "int8"


def test_use_refuses_an_unknown_variant_and_names_the_choices(tmp_path):
    done, message = use_variant(_cfg(tmp_path), MODEL, "bf16")
    assert not done and "int8-convrot" in message


def test_use_selects_an_installed_variant_and_writes_the_configuration(tmp_path):
    cfg = _cfg(tmp_path, "int8")
    _place(cfg, "mixed-p015")
    done, message = use_variant(cfg, MODEL, "mixed-p015")
    assert done and "restart" in message
    assert load(tmp_path).model(MODEL).variant == "mixed-p015"


# ---------------------------------------------------------------------------------------------- remove (D9)


def test_remove_a_convrot_variant_keeps_the_files_other_variants_need(tmp_path):
    cfg = _cfg(tmp_path)
    folder = _place(cfg, "int8-convrot")
    _place(cfg, "mixed-p015")
    done, message = remove_variant(cfg, MODEL, "mixed-p015")
    assert done and "kept" in message
    assert not (folder / QWEN.variants["mixed-p015"]["files"][0]).exists()
    assert (folder / QWEN.variants["int8-convrot"]["files"][0]).exists() and (folder / "comfyorg" / "te.safetensors").exists()
    assert not variant_installed(cfg, QWEN, "mixed-p015") and variant_installed(cfg, QWEN, "int8-convrot")
    assert "mixed-p015" not in json.loads((folder / "install.json").read_text(encoding="utf-8"))["variants"]


def test_the_shared_files_go_only_with_the_last_variant_and_only_when_asked(tmp_path):
    cfg = _cfg(tmp_path)
    folder = _place(cfg, "mixed-p015")
    done, message = remove_variant(cfg, MODEL, "mixed-p015")  # last one, but the shared files were not asked for
    assert done and "--include-shared" in message and (folder / "comfyorg" / "te.safetensors").exists()
    _place(cfg, "mixed-p015")
    done, _ = remove_variant(cfg, MODEL, "mixed-p015", include_shared=True)
    assert done and not folder.exists()


def test_include_shared_does_not_delete_while_another_variant_is_installed(tmp_path):
    cfg = _cfg(tmp_path)
    folder = _place(cfg, "int8-convrot")
    _place(cfg, "mixed-p015")
    remove_variant(cfg, MODEL, "mixed-p015", include_shared=True)
    assert folder.exists() and variant_installed(cfg, QWEN, "int8-convrot")


def test_remove_int8_deletes_its_own_folder_and_nothing_else(tmp_path):
    cfg = _cfg(tmp_path)
    convrot = _place(cfg, "int8-convrot")
    own = _place(cfg, "int8")
    done, _ = remove_variant(cfg, MODEL, "int8")
    assert done and not own.exists() and convrot.exists()


def test_a_dry_run_deletes_nothing_and_says_how_much_it_would_free(tmp_path):
    cfg = _cfg(tmp_path)
    folder = _place(cfg, "mixed-p015")
    done, message = remove_variant(cfg, MODEL, "mixed-p015", dry_run=True)
    assert done and "would free" in message and variant_installed(cfg, QWEN, "mixed-p015") and folder.exists()


def test_remove_says_so_when_there_is_nothing_to_remove(tmp_path):
    done, message = remove_variant(_cfg(tmp_path), MODEL, "mixed-p015")
    assert not done and "not installed" in message


def test_remove_never_deletes_outside_the_models_folder(tmp_path, monkeypatch):
    """Negative security test: a folder that resolves outside <data>/models/<model> (a mistake in the table, a link) is refused."""
    cfg = _cfg(tmp_path / "data")
    outside = tmp_path / "precious"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep", encoding="utf-8")
    monkeypatch.setattr(install, "weights_dir", lambda *_a, **_k: outside)
    done, message = remove_variant(cfg, MODEL, "int8")
    assert not done and "outside" in message and (outside / "keep.txt").exists()


# ---------------------------------------------------------------------------------------------- install: a second variant never switches the first


def _fake_downloads(monkeypatch):
    """Replace the network with files of the right names; the checksum of a 16-byte stand-in cannot match, so it is skipped."""

    def download(parts, target):
        for part in parts:
            dest = target / part["subdir"] if part["subdir"] else target
            dest.mkdir(parents=True, exist_ok=True)
            for name in part["files"]:
                (dest / name).parent.mkdir(parents=True, exist_ok=True)
                (dest / name).write_bytes(b"z" * 16)
            if not part["subdir"]:
                (dest / "model_index.json").write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.setattr(install, "_download_parts", download)
    monkeypatch.setattr(install, "_verify_parts", lambda parts, target: 0)
    monkeypatch.setattr(install, "free_disk_gib", lambda path: 1000.0)
    monkeypatch.setattr(install, "detect_gpus", lambda ttl_s=30.0: [{"index": 0, "name": "RTX", "memory_total_mib": 24564}])


def _install(tmp_path, variant, capsys) -> str:
    capsys.readouterr()
    code = main(["--data-dir", str(tmp_path), "models", "install", MODEL, "--variant", variant, "--accept-license", "qwen-research", "--skip-env"])
    out = capsys.readouterr().out
    assert code == 0, out
    return out


def test_installing_a_second_variant_adds_it_and_does_not_switch(tmp_path, monkeypatch, capsys):
    assert main(["--data-dir", str(tmp_path), "init"]) == 0
    _fake_downloads(monkeypatch)
    _install(tmp_path, "int8-convrot", capsys)
    assert load(tmp_path).model(MODEL).variant == "int8-convrot"  # the first one is selected: nothing else was in use
    out = _install(tmp_path, "mixed-p015", capsys)
    assert load(tmp_path).model(MODEL).variant == "int8-convrot"  # never switched behind the operator's back
    assert "still uses int8-convrot" in out and f"models use {MODEL} mixed-p015" in out
    folder = weights_dir(load(tmp_path), ModelConfig(MODEL, "qwen_image", variant="mixed-p015"))
    assert json.loads((folder / "install.json").read_text(encoding="utf-8"))["variants"] == ["int8-convrot", "mixed-p015"]


def test_models_list_shows_every_variant_and_which_one_is_in_use(tmp_path, monkeypatch, capsys):
    assert main(["--data-dir", str(tmp_path), "init"]) == 0
    _fake_downloads(monkeypatch)
    _install(tmp_path, "int8-convrot", capsys)
    capsys.readouterr()
    assert main(["--data-dir", str(tmp_path), "models", "list"]) == 0
    out = capsys.readouterr().out
    for name in QWEN.variants:
        assert name in out
    assert "in use" in out and "not installed" in out


def test_models_remove_asks_for_yes_before_deleting(tmp_path, monkeypatch, capsys):
    assert main(["--data-dir", str(tmp_path), "init"]) == 0
    _fake_downloads(monkeypatch)
    _install(tmp_path, "int8-convrot", capsys)
    cfg = load(tmp_path)
    target = weights_dir(cfg, ModelConfig(MODEL, "qwen_image", variant="int8-convrot")) / QWEN.variants["int8-convrot"]["files"][0]
    assert main(["--data-dir", str(tmp_path), "models", "remove", MODEL, "int8-convrot"]) == 0
    assert target.exists() and "Nothing was deleted" in capsys.readouterr().out
    assert main(["--data-dir", str(tmp_path), "models", "remove", MODEL, "int8-convrot", "--yes"]) == 0
    assert not target.exists()


# ---------------------------------------------------------------------------------------------- the worker's own environment


def _lock_packages() -> dict[str, str]:
    import tomllib

    lock = tomllib.loads((Path(__file__).resolve().parents[1] / "workers" / "qwen_image" / "uv.lock").read_text(encoding="utf-8"))
    return {p["name"]: p.get("version", "") for p in lock["package"]}


def test_the_qwen_lock_has_comfy_kitchen_and_still_bitsandbytes_and_never_comfyui():
    packages = _lock_packages()
    assert packages.get("comfy-kitchen") == "0.2.42"  # the ConvRot variants
    assert "bitsandbytes" in packages  # the `int8` variant is the fourth choice, not a leftover (plan D10)
    assert not [name for name in packages if "comfyui" in name.lower()]  # the GPL application must never enter this environment


def test_every_dependency_of_the_qwen_worker_is_pinned_exactly():
    import re
    import tomllib

    project = tomllib.loads((Path(__file__).resolve().parents[1] / "workers" / "qwen_image" / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    for dep in project["dependencies"]:
        assert "==" in dep or re.search(r"@ git\+https://\S+@[0-9a-f]{40}$", dep), f"not pinned: {dep}"
