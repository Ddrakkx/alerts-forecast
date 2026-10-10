"""Download the pinned snapshot of the volunteer and official CSV files into data/raw/.

Existing files are never overwritten: if one is present its size is checked and the script
stops with an error on a mismatch.

    python scripts/download_data.py                 # the pinned snapshot used for all results
    python scripts/download_data.py --new-snapshot  # the newest volunteer file into data/holdout/ (holdout test, decision 9)
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

SHA = "2f115548a8bd0816c901dbf422082091b3e5c34c"
REPO = "Vadimkin/ukrainian-air-raid-sirens-dataset"
URL = "https://raw.githubusercontent.com/" + REPO + "/{sha}/datasets/{name}"
API_HEAD = "https://api.github.com/repos/" + REPO + "/commits/HEAD"
FILES = {  # name -> expected size in bytes
    "volunteer_data_en.csv": 8876306,
    "official_data_en.csv": 31025633,
}
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw"
HOLDOUT = ROOT / "data" / "holdout"


def ensure(dest: Path = DEST, files: dict = FILES, sha: str = SHA, opener=urllib.request.urlopen) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name, size in files.items():
        path = dest / name
        if path.exists():
            if path.stat().st_size != size:
                raise SystemExit(f"{path}: size {path.stat().st_size} != expected {size}, not overwriting")
            print(f"ok      {name} ({size} bytes)")
            continue
        print(f"fetch   {name}")
        with opener(URL.format(sha=sha, name=name)) as resp:
            data = resp.read()
        if len(data) != size:
            raise SystemExit(f"{name}: downloaded {len(data)} bytes, expected {size}")
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(path)
        print(f"saved   {path}")


def fetch_new_snapshot(dest: Path = HOLDOUT, opener=urllib.request.urlopen) -> str:
    """Newest volunteer file at the current HEAD commit; its SHA, size and sha256 go to SNAPSHOT.txt. Never overwrites."""
    path = dest / "volunteer_data_en.csv"
    if path.exists():
        raise SystemExit(f"{path} exists, not overwriting")
    with opener(API_HEAD) as resp:
        sha = json.load(resp)["sha"]
    with opener(URL.format(sha=sha, name="volunteer_data_en.csv")) as resp:
        data = resp.read()
    dest.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    (dest / "SNAPSHOT.txt").write_text(
        f"sha {sha}" + chr(10) + f"bytes {len(data)}" + chr(10) + f"sha256 {hashlib.sha256(data).hexdigest()}" + chr(10),
        encoding="utf-8",
    )
    print(f"saved   {path} ({len(data)} bytes) at commit {sha}")
    return sha


if __name__ == "__main__":
    sys.exit(fetch_new_snapshot() and None if "--new-snapshot" in sys.argv else ensure())
