"""The support report (aha_open/report.py): what it carries, what it must never carry, and how it keeps within a size.

Plan 2026-11-25_ui_menu_report_about_plan.md, U2. Real files in a temporary data folder; the one thing replaced is the GPU probe (it would
run `nvidia-smi`), so the result does not depend on the machine the tests run on.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from aha_open import cli, report, runtime
from aha_open.config import Config, ModelConfig, render

TOKEN = "aho_SENTINEL_TOKEN_0123456789abcdef"
CONFIG_WORD = "SENTINEL-CONFIG-WORD-91c2"


@pytest.fixture(autouse=True)
def no_gpu_probe(monkeypatch):
    monkeypatch.setattr(report, "detect_gpus", lambda ttl_s=0: [{"index": 0, "name": "Test GPU", "memory_total_mib": 24576}])


@pytest.fixture
def cfg(tmp_path):
    data = tmp_path / "AHA open data"
    (data / "logs").mkdir(parents=True)
    cfg = Config(data_dir=data, models=[ModelConfig(id="example-image", worker="fake")])
    (data / "config.toml").write_text(render(cfg) + f"\n# {CONFIG_WORD}\n", encoding="utf-8")
    cfg.local_token_path.write_text(TOKEN, encoding="utf-8")
    return cfg


def write_log(cfg: Config, name: str, text: str) -> Path:
    path = cfg.data_dir / "logs" / name
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------------------------- redact


@pytest.mark.parametrize("secret", [
    "aho_AbCdEf123456_-xyz",
    "hf_" + "a" * 30,
])
def test_a_token_shaped_value_is_removed_wherever_it_stands(secret):
    out = report.redact(f"x {secret} y")
    assert secret not in out and "[redacted]" in out


def test_an_authorization_header_and_a_key_value_pair_keep_their_words_and_lose_the_secret():
    out = report.redact("Authorization: Bearer abc.DEF-123456 and api_key=sk-live-99999 and token: zzzzzz1")
    assert "abc.DEF-123456" not in out and "sk-live-99999" not in out and "zzzzzz1" not in out
    assert "api_key=[redacted]" in out


def test_a_sentence_that_only_mentions_a_token_is_left_alone():
    sentence = "The token is required on this profile and the token file was created."
    assert report.redact(sentence) == sentence


def test_the_home_and_data_folders_become_short_names_in_every_spelling(cfg, monkeypatch, tmp_path):
    home = tmp_path / "Users" / "alice"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    data = str(cfg.data_dir)
    text = chr(10).join([data + "\\logs\\x.log", data.replace("\\", "/") + "/y", json.dumps(data), str(home / "docs" / "p.png")])
    out = report.redact(text, cfg)
    assert "AHA open data" not in out and "alice" not in out
    assert "<data>" in out and "~" in out


def test_another_persons_user_folder_keeps_its_shape_and_loses_the_name():
    out = report.redact(r"C:\Users\bob\AppData\x and /home/carol/.cache and /Users/dave/Library")
    assert "bob" not in out and "carol" not in out and "dave" not in out
    assert r"C:\Users\<user>" in out and "/home/<user>" in out and "/Users/<user>" in out


# ---------------------------------------------------------------------------------------------- budget


def test_a_small_file_keeps_all_of_itself_and_the_large_ones_share_the_rest():
    shares = report.split_budget(1000, {"small": 100, "big_a": 5000, "big_b": 7000})
    assert shares["small"] == 100 and shares["big_a"] == shares["big_b"] == 450
    assert sum(shares.values()) <= 1000


def test_a_budget_that_covers_everything_gives_every_file_all_of_itself_and_none_gets_more_than_its_size():
    sizes = {"a": 10, "b": 20, "c": 30}
    assert report.split_budget(10_000, sizes) == sizes
    assert sum(report.split_budget(5, {"a": 100, "b": 100, "c": 100}).values()) <= 5


def test_the_logs_are_cut_to_the_budget_from_a_whole_line_and_the_end_is_kept(cfg):
    write_log(cfg, "gateway.log", "".join(f"line {i:05d} {'x' * 40}" + chr(10) for i in range(2000)))
    text = report.build_report(cfg, budget=2000)
    body = text.split("== Log: gateway.log ==")[1].split(chr(10), 2)[2]
    assert len(body.encode("utf-8")) <= 2000
    assert "line 01999" in body and "line 00000" not in body
    assert body.splitlines()[0].startswith("line ")  # no half line at the cut


def test_the_gateway_log_is_read_across_its_rotated_backups_oldest_first(cfg):
    write_log(cfg, "gateway.log.2", "oldest" + chr(10))
    write_log(cfg, "gateway.log.1", "middle" + chr(10))
    write_log(cfg, "gateway.log", "newest" + chr(10))
    body = report.build_report(cfg).split("== Log: gateway.log ==")[1]
    assert body.index("oldest") < body.index("middle") < body.index("newest")


# ---------------------------------------------------------------------------------------------- what is in it


def test_the_report_names_the_machine_the_gateway_the_models_and_the_age_of_every_log(cfg):
    runtime.write_run_record(cfg)
    log = write_log(cfg, "gateway.log", "something happened" + chr(10))
    old = time.time() - 3 * 3600
    os.utime(log, (old, old))
    write_log(cfg, "worker-fake.log", "worker line" + chr(10))
    text = report.build_report(cfg)
    for needle in ("== Machine ==", "Test GPU, 24576 MiB", "== Gateway ==", "Profile: local", "== Models ==", "example-image", "== Log: gateway.log ==",
                   "== Log: worker-fake.log ==", "something happened", "worker line"):
        assert needle in text, needle
    assert "3.0 h ago" in text  # an old log says it is old: it is a hypothesis about the failure, not evidence of it
    assert "the running gateway matches the files on disk" in text


def test_a_gateway_started_from_other_files_than_the_ones_on_disk_is_said_so(cfg):
    runtime.write_run_record(cfg)
    record = json.loads(runtime.record_path(cfg).read_text(encoding="utf-8"))
    record["source_digest"] = "0" * 64
    runtime.record_path(cfg).write_text(json.dumps(record), encoding="utf-8")
    text = report.build_report(cfg)
    assert "Code: DIFFERENT" in text and "restart it" in text


def test_the_persons_note_comes_first_and_is_cut_to_its_limit(cfg):
    text = report.build_report(cfg, note="  The picture came out grey.  " + "x" * 10_000)
    assert text.index("The picture came out grey.") < text.index("== Machine ==")
    note = text.split("== What happened (the person's words) ==" + chr(10))[1].split(chr(10) * 2)[0]
    assert len(note) <= report.NOTE_MAX


def test_a_data_folder_without_logs_still_gives_a_report_that_says_so(cfg):
    assert "(no log file in the data folder yet)" in report.build_report(cfg)


# ---------------------------------------------------------------------------------------------- what must never be in it


def test_the_token_file_and_the_config_file_are_never_read_into_the_report(cfg):
    runtime.write_run_record(cfg)
    write_log(cfg, "gateway.log", "ok" + chr(10))
    text = report.build_report(cfg)
    assert TOKEN not in text and CONFIG_WORD not in text


def test_a_secret_that_reached_a_log_does_not_reach_the_report(cfg):
    write_log(cfg, "gateway.log", f"2026 INFO x: client sent Authorization: Bearer {TOKEN} and {TOKEN} in {cfg.data_dir}\\tmp" + chr(10))
    text = report.build_report(cfg)
    assert TOKEN not in text and "SENTINEL_TOKEN" not in text
    assert str(cfg.data_dir) not in text and "<data>" in text


def test_building_the_report_changes_nothing_on_disk(cfg):
    write_log(cfg, "gateway.log", "a" + chr(10))
    before = sorted((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in cfg.data_dir.rglob("*") if p.is_file())
    report.build_report(cfg)
    assert sorted((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in cfg.data_dir.rglob("*") if p.is_file()) == before


# ---------------------------------------------------------------------------------------------- the command


def test_the_command_writes_a_file_in_utf8_and_says_where(cfg, tmp_path, capsys):
    write_log(cfg, "gateway.log", "dòng tiếng Việt" + chr(10))
    out = tmp_path / "report.txt"
    assert cli.main(["--data-dir", str(cfg.data_dir), "report", "--out", str(out), "--note", "nó bị lỗi"]) == 0
    text = out.read_text(encoding="utf-8")
    assert "dòng tiếng Việt" in text and "nó bị lỗi" in text and TOKEN not in text
    assert str(out) in capsys.readouterr().out


def test_the_command_prints_utf8_bytes_whatever_the_console_code_page_is(cfg, capsysbinary):
    write_log(cfg, "gateway.log", "dòng tiếng Việt" + chr(10))
    assert cli.main(["--data-dir", str(cfg.data_dir), "report"]) == 0
    assert "dòng tiếng Việt" in capsysbinary.readouterr().out.decode("utf-8")


# ---------------------------------------------------------------------------------------------- repeated lines and the time zone (measured on a real log)


def test_a_client_that_polls_every_few_milliseconds_does_not_push_the_failure_out_of_the_budget(cfg):
    polls = "".join(f"2026-11-25 15:50:{i % 60:02d},{i % 1000:03d} INFO uvicorn.access: 127.0.0.1:{50000 + i} - \"GET /v1/jobs/job_1 HTTP/1.1\" 200" + chr(10)
                    for i in range(3000))
    write_log(cfg, "gateway.log", "2026-11-25 15:49:00,000 ERROR aha_open.scheduler: worker crashed: out of memory" + chr(10) + polls)
    text = report.build_report(cfg, budget=3000)
    assert "worker crashed: out of memory" in text  # the cause survives: it was before 3000 polls
    assert "the same line 2999 more times" in text and text.count("GET /v1/jobs/job_1") == 1


def test_different_lines_and_short_runs_are_never_collapsed():
    lines = ["2026-11-25 10:00:00,001 INFO a: x", "2026-11-25 10:00:00,002 INFO a: x", "2026-11-25 10:00:00,003 INFO a: y",
             "2026-11-25 10:00:00,004 INFO a: x", "", "", "", ""]
    assert report.collapse_repeats(chr(10).join(lines)) == chr(10).join(lines)  # two in a row stay; blank lines are not "the same line"


def test_the_report_says_which_time_zone_the_log_times_are_in(cfg):
    text = report.build_report(cfg)
    assert "Time zone: UTC" in text and "local time" in text
