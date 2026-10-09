"""Download the pinned snapshot of the volunteer and official CSV files into data/raw/.

Existing files are never overwritten: if one is present its size is checked and the script
stops with an error on a mismatch.

    python scripts/download_data.py
"""
import sys
import urllib.request
from pathlib import Path

SHA = "2f115548a8bd0816c901dbf422082091b3e5c34c"
URL = "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/{sha}/datasets/{name}"
FILES = {  # name -> expected size in bytes
    "volunteer_data_en.csv": 8876306,
    "official_data_en.csv": 31025633,
}
DEST = Path(__file__).resolve().parents[1] / "data" / "raw"


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


if __name__ == "__main__":
    sys.exit(ensure())
