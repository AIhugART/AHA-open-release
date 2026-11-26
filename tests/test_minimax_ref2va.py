"""MiniMax H3 reference-to-video (Ref2VA): what the gateway declares, accepts and refuses (plan 2026-11-24_minimax_h3_plan.md, M7)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aha_open import catalog
from aha_open import install as install_module
from aha_open.catalog import CATALOG, manifest_model, model_state, validate_job, weights_dir
from aha_open.config import Config, ModelConfig
from aha_open.errors import ApiError

REF = CATALOG["minimax-h3-ref2va"]
TEXT = CATALOG["minimax-h3-fl2va"]
FILE = "abiray/MiniMax_H3_Ref2VA_pruned_mixed_int4_int8_convrot.safetensors"


def ok(inputs, params=None, spec=REF):
    return validate_job(spec, "video.generate", "a person waves", params or {}, inputs, variant="mixed")


def refused(inputs, needle, params=None, spec=REF):
    with pytest.raises(ApiError) as err:
        ok(inputs, params, spec)
    assert err.value.status == 400 and needle in err.value.message, err.value.message


# ---------------------------------------------------------------------------------------------- what is declared


def test_it_declares_the_same_licence_as_the_text_model_and_only_the_size_and_references_that_were_run():
    assert [s["tier"] for s in REF.capabilities["sizes"]] == ["preview"]  # native was not run with a reference
    assert REF.capabilities["short_edges"] == [352] and REF.capabilities["sizes"][0] in TEXT.capabilities["sizes"]
    assert REF.variants["mixed"]["vram_gib"] == {"preview": 17.1, "native": None}  # measured at preview; native not claimed
    assert REF.license == TEXT.license
    assert REF.capabilities["reference_required"] is True and "reference_required" not in TEXT.capabilities
    assert REF.capabilities["max_reference"] == {"image": 1, "video": 0, "audio": 0, "total": 1}  # widened only after a real run
    assert TEXT.capabilities["max_reference"]["total"] == 0  # the text model is not changed by the new one


def test_the_manifest_of_the_reference_model_says_what_the_client_may_send():
    cfg = Config(data_dir=Path("."), listen="127.0.0.1:1", models=[ModelConfig(id=REF.id, worker="minimax_h3", variant="mixed")])
    entry = manifest_model(cfg, cfg.models[0], REF)
    assert entry["id"] == "minimax-h3-ref2va" and entry["tasks"] == ["video.generate"]
    assert entry["capabilities"]["max_reference"]["image"] == 1


# ---------------------------------------------------------------------------------------------- what is accepted and refused


def test_the_size_that_was_not_run_with_a_reference_is_refused_and_the_offered_one_named():
    with pytest.raises(ApiError) as err:
        ok([{"role": "reference"}], {"short_edge": 768})
    assert err.value.status == 400 and "short_edge must be one of [352]" in err.value.message


def test_one_reference_image_is_accepted_and_the_params_are_complete_like_the_text_model():
    accepted = ok([{"role": "reference"}], {"short_edge": 352})
    assert (accepted["width"], accepted["height"], accepted["num_frames"], accepted["steps"]) == (608, 352, 124, 20)


def test_a_request_without_any_reference_is_refused_and_names_the_text_model():
    refused([], "needs at least one reference image or video")
    refused([], "minimax-h3-fl2va")


@pytest.mark.parametrize(
    ("inputs", "needle"),
    [
        ([{"role": "reference"}, {"role": "reference"}], "too many reference inputs"),  # one image was measured; the per-kind limit answers first
        ([{"role": "reference_video"}], "too many reference_video"),
        ([{"role": "reference_audio"}], "too many reference_audio"),
        ([{"role": "first_frame"}], "does not accept a first frame"),
        ([{"role": "last_frame"}], "does not accept a last frame"),
    ],
)
def test_what_was_not_run_is_refused_before_a_job_exists(inputs, needle):
    refused(inputs, needle)


def test_an_audio_reference_cannot_stand_alone_even_when_audio_is_allowed():
    wide = {**REF.capabilities, "max_reference": {"image": 9, "video": 3, "audio": 3, "total": 12}}
    spec = catalog.replace(REF, capabilities=wide)
    refused([{"role": "reference_audio"}], "audio reference cannot stand alone", spec=spec)
    assert ok([{"role": "reference_audio"}, {"role": "reference"}], spec=spec)  # beside an image it is fine


def test_the_total_limit_holds_even_when_every_kind_is_below_its_own_limit():
    wide = {**REF.capabilities, "max_reference": {"image": 9, "video": 3, "audio": 3, "total": 2}}
    spec = catalog.replace(REF, capabilities=wide)
    refused([{"role": "reference"}] * 3, "at most 2 references", spec=spec)


def test_the_text_model_still_refuses_every_reference():
    refused([{"role": "reference"}], "too many reference inputs", spec=TEXT)


# ---------------------------------------------------------------------------------------------- where the files live


def test_the_reference_model_lives_in_the_folder_of_the_text_model(tmp_path):
    cfg = Config(data_dir=tmp_path, listen="127.0.0.1:1")
    ref = weights_dir(cfg, ModelConfig(id=REF.id, worker="minimax_h3", variant="mixed"))
    text = weights_dir(cfg, ModelConfig(id=TEXT.id, worker="minimax_h3", variant="mixed"))
    assert ref == text == tmp_path / "models" / "minimax-h3-fl2va" / "mixed"


def test_text_model_installed_does_not_make_the_reference_model_ready(tmp_path, monkeypatch):
    cfg = Config(data_dir=tmp_path, listen="127.0.0.1:1")
    mc = ModelConfig(id=REF.id, worker="minimax_h3", variant="mixed")
    folder = weights_dir(cfg, mc)
    folder.mkdir(parents=True)
    (folder / "model_index.json").write_text("{}", encoding="utf-8")  # what the text model's install leaves
    monkeypatch.setattr(catalog, "worker_python", lambda *_: Path("python"))
    monkeypatch.setattr(catalog, "WORKERS_DIR", tmp_path)
    (tmp_path / "minimax_h3").mkdir()
    (tmp_path / "minimax_h3" / "worker.py").write_text("", encoding="utf-8")
    state, detail = model_state(cfg, mc, REF)
    assert state == "not_installed" and FILE in detail and "aha-open models install minimax-h3-ref2va" in detail


# ---------------------------------------------------------------------------------------------- the install plan


def parts(model="minimax-h3-ref2va"):
    return install_module.INSTALLABLE[model]["variants"]["mixed"]["parts"]


def test_missing_gib_counts_only_files_that_are_not_complete_on_disk(tmp_path):
    """Tiny synthetic parts: nothing the size of a real model is ever created by a test."""
    shared = {"vae/v.bin": (10, "x"), "enc.bin": (20, "x")}
    synthetic = [{"subdir": "abiray", "files": {"ref.bin": (2**30, "x"), **shared}}]
    (tmp_path / "abiray" / "vae").mkdir(parents=True)
    for name, (size, _) in shared.items():
        (tmp_path / "abiray" / name).write_bytes(b"\x00" * size)
    assert install_module._missing_gib(synthetic, tmp_path) == pytest.approx(1.0)  # only ref.bin is missing
    (tmp_path / "abiray" / "ref.bin").write_bytes(b"\x00" * 5)  # present but the wrong size = incomplete = counted
    assert install_module._missing_gib(synthetic, tmp_path) == pytest.approx(1.0)


def test_the_two_install_plans_differ_only_in_the_transformer_file():
    """So a machine with the text model downloads exactly the Ref2VA file and nothing else."""
    def files(model):
        return {n: v for p in parts(model) for n, v in p["files"].items()}

    text, ref = files("minimax-h3-fl2va"), files("minimax-h3-ref2va")
    assert set(text) - set(ref) == {"MiniMax_H3_FL2VA_pruned_mixed_int4_int8_convrot.safetensors"}
    assert set(ref) - set(text) == {"MiniMax_H3_Ref2VA_pruned_mixed_int4_int8_convrot.safetensors"}
    assert all(text[n] == ref[n] for n in set(text) & set(ref))  # same pinned size and sha256 for the shared files


def test_the_plan_pins_the_file_that_was_downloaded_and_checked():
    files = {n: v for p in parts() for n, v in p["files"].items()}
    assert files["MiniMax_H3_Ref2VA_pruned_mixed_int4_int8_convrot.safetensors"] == (
        15093774276, "434c372cbb68ff055ffb5aa2c35de42211d19ebacc2ae8e9ae9202b00ee48afb")
    assert "transformer_ref/config.json" in parts()[0]["patterns"]
    assert install_module.INSTALLABLE["minimax-h3-ref2va"]["territory"] is True  # the licence question is asked here too
