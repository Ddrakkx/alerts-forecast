import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import download_data  # noqa: E402


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_downloads_missing_file_and_checks_size(tmp_path):
    seen = []

    def opener(url):
        seen.append(url)
        return FakeResponse(b"12345")

    download_data.ensure(tmp_path, {"a.csv": 5}, sha="abc", opener=opener)
    assert (tmp_path / "a.csv").read_bytes() == b"12345"
    assert "/abc/datasets/a.csv" in seen[0]
    assert not list(tmp_path.glob("*.part"))


def test_existing_file_is_not_overwritten_or_refetched(tmp_path):
    (tmp_path / "a.csv").write_bytes(b"12345")

    def opener(url):
        raise AssertionError("must not download")

    download_data.ensure(tmp_path, {"a.csv": 5}, sha="abc", opener=opener)
    (tmp_path / "a.csv").write_bytes(b"1")
    with pytest.raises(SystemExit, match="not overwriting"):
        download_data.ensure(tmp_path, {"a.csv": 5}, sha="abc", opener=opener)
    assert (tmp_path / "a.csv").read_bytes() == b"1"


def test_wrong_downloaded_size_is_rejected_and_nothing_saved(tmp_path):
    with pytest.raises(SystemExit, match="expected"):
        download_data.ensure(tmp_path, {"a.csv": 5}, sha="abc", opener=lambda url: FakeResponse(b"123"))
    assert not (tmp_path / "a.csv").exists()
