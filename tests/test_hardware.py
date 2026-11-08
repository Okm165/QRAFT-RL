from __future__ import annotations

from tools.hardware import _hip_unique_id, _status, _unique


def test_unique_id_normalizes_rocm_prefix() -> None:
    assert _unique(" 0xABC ") == "abc"


def test_hip_uuid_decodes_ascii_bytes() -> None:
    assert _hip_unique_id("41424344-4546-4748-494a-4b4c4d4e4f50") == "abcdefghijklmnop"


def test_status_reports_link_and_aer_hops() -> None:
    status, issues = _status(
        8,
        16,
        "0000:40:01.1",
        16,
        77500,
        "0000:40:03.1",
        "0000:46:00.0",
        0.0,
        False,
        40.0,
    )
    assert status == "warn"
    assert issues == (
        "link x8 at 0000:40:01.1 (max x16; endpoint x16)",
        "77500 correctable AER at 0000:40:03.1; endpoint 0000:46:00.0 clean",
    )
