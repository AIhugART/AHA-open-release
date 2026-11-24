"""The error-code guide is a second copy of knowledge that lives in the gateway: it is guarded in both directions."""

from __future__ import annotations

from pathlib import Path

import aha_open
from aha_open.worker_proc import describe_exit

from aha_open_mcp import guide


def test_every_code_the_gateway_emits_or_the_spec_lists_has_an_entry():
    known = set(guide.ERROR_CODE_GUIDE)
    assert guide.codes_in_gateway_source() - known == set(), "the gateway emits a code the guide cannot explain"
    assert guide.codes_in_specification() - known == set(), "the specification lists a code the guide cannot explain"


def test_no_entry_describes_a_code_nobody_emits_or_specifies():
    live = guide.codes_in_gateway_source() | guide.codes_in_specification()
    assert set(guide.ERROR_CODE_GUIDE) - live == set()


def test_the_source_scanner_finds_the_codes_it_is_meant_to():
    found = guide.codes_in_gateway_source()
    assert {"invalid_request", "unauthorized", "busy", "generation_failed", "gpu_out_of_memory", "upload_incomplete"} <= found


def test_every_generation_failed_pattern_is_text_the_gateway_still_writes():
    """A reworded message in the gateway would silently turn its pattern into dead text: the pattern must still occur in source."""
    source = "\n".join(p.read_text(encoding="utf-8") for p in Path(aha_open.__file__).parent.rglob("*.py") if "workers" not in p.parts)
    dead = [needle for needle, _, _ in guide.GENERATION_FAILED_PATTERNS if needle not in source]
    assert dead == [], dead


def test_worker_exit_sentences_are_explained_by_the_pattern_meant_for_them():
    memory = describe_exit(0xC000012D, "worker-qwen_image.log")
    crash = describe_exit(0xC0000005, "worker-qwen_image.log")
    no_start = describe_exit(0xC0000142, "worker-qwen_image.log")
    assert "RAM" in guide.explain("generation_failed", f"while loading the model: {memory}")[0]
    assert "native code" in guide.explain("generation_failed", f"while generating: {crash}")[0]
    assert "could not even start" in guide.explain("generation_failed", f"while loading the model: {no_start}")[0]


def test_different_causes_get_different_actions():
    """'the token is wrong', 'the model is loading' and 'the queue is full' must not collapse into one advice."""
    actions = {c: guide.ERROR_CODE_GUIDE[c][1] for c in ("unauthorized", "model_loading", "busy", "model_not_installed")}
    assert len(set(actions.values())) == len(actions)
    assert "NOT 'the server is off'" in guide.ERROR_CODE_GUIDE["unauthorized"][0]


def test_an_unknown_code_is_said_to_be_unknown_not_guessed():
    cause, action = guide.explain("brand_new_code", "x")
    assert "Unknown error code" in cause and "guide.py" in action
