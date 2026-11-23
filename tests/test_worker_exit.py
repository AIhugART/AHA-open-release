"""What the person running the server reads when a worker process dies (no GPU, no torch).

Root cause this guards: a native crash gave the client only "exit code 3221225477". That number is 0xC0000005
(access violation) and says nothing the operator can act on; and memory exhaustion needs different advice than a crash.
"""

from __future__ import annotations

import pytest

from aha_open.worker_proc import describe_exit

LOG = "worker-qwen_image.log"


def test_an_access_violation_says_crash_not_the_request_and_points_at_the_log():
    text = describe_exit(3221225477, LOG)
    assert "access violation" in text and "0xC0000005" in text and "3221225477" in text
    assert "not the cause" in text and LOG in text
    assert "Free memory" not in text  # a crash must not be diagnosed as a memory problem we have not shown


def test_the_same_status_as_a_signed_number_reads_the_same():
    assert describe_exit(-1073741819, LOG) == describe_exit(3221225477, LOG)  # 0xC0000005 as a signed 32-bit integer


@pytest.mark.parametrize("code", [0xC000012D, 0xC0000017, -9])
def test_running_out_of_memory_gets_memory_advice(code):
    text = describe_exit(code, LOG)
    assert "Free memory" in text and "not the cause" not in text


@pytest.mark.parametrize(("code", "word"), [(-11, "segmentation fault"), (-6, "aborted"), (0xC0000409, "fail-fast"), (0xC0000374, "heap corruption")])
def test_other_crashes_are_named(code, word):
    assert word in describe_exit(code, LOG)


def test_a_plain_exit_code_is_not_called_a_crash():
    text = describe_exit(3, LOG)
    assert "exit code 3" in text and "crashed" not in text


def test_unknown_statuses_still_give_a_number_to_search_for():
    assert "0xC0001234" in describe_exit(0xC0001234, LOG)
    assert "signal 99" in describe_exit(-99, LOG)
    assert LOG in describe_exit(None, LOG)


def test_the_text_that_travels_to_the_client_names_the_log_file_never_a_path():
    for code in (3221225477, -9, 3, None):
        text = describe_exit(code, LOG)
        assert "\\" not in text and "/" not in text  # no server directory is revealed to a client
