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


def test_holdout_verdict_direction_depends_on_the_metric():
    import run_holdout

    assert run_holdout.fmt((0.02, 0.01, 0.03), "pr_auc").endswith("better")
    assert run_holdout.fmt((0.02, 0.01, 0.03), "brier").endswith("worse")
    assert run_holdout.fmt((-0.02, -0.03, -0.01), "brier").endswith("better")
    assert run_holdout.fmt((0.0, -0.01, 0.01), "pr_auc").endswith("inconclusive")


def test_new_snapshot_records_sha_and_never_overwrites(tmp_path):
    def opener(url):
        if "api.github.com" in url:
            return FakeResponse(b'{"sha": "abc123"}')
        assert "/abc123/datasets/volunteer_data_en.csv" in url
        return FakeResponse(b"region,started_at\n")

    assert download_data.fetch_new_snapshot(tmp_path, opener=opener) == "abc123"
    info = (tmp_path / "SNAPSHOT.txt").read_text()
    assert "sha abc123" in info and "bytes 18" in info and "sha256 " in info
    with pytest.raises(SystemExit, match="not overwriting"):
        download_data.fetch_new_snapshot(tmp_path, opener=opener)
