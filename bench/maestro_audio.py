"""Fetch the MAESTRO audio of ASAP performances, without downloading the 100 GB zip.

    uv run python bench/maestro_audio.py           # one performance per piece beat_this held out
    uv run python bench/maestro_audio.py --all     # every ASAP performance with audio

Reads single members of the remote zip with HTTP range requests, cuts the
performance out (ASAP's start/end) and stores it as 16 kHz mono FLAC in
bench/asap_audio/<performance id>.flac, plus the beat_this fold that held
it out (bench/asap_audio/folds.json), so its beats can be tracked by a model
that never saw the piece.
"""

import csv
import io
import json
import subprocess
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent
ASAP = ROOT / "asap"
AUDIO = ROOT / "asap_audio"
URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"
FOLDS = "https://raw.githubusercontent.com/CPJKU/beat_this_annotations/main/asap/8-folds.split"


class RangeFile(io.RawIOBase):
    def __init__(self, url: str):
        self.url, self.pos = url, 0
        self.size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD")).headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        data = urllib.request.urlopen(req).read()
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def perf_id(row: dict) -> str:
    return row["midi_performance"].removesuffix(".mid").replace("/", "_")


def main() -> None:
    AUDIO.mkdir(exist_ok=True)
    folds = dict(line.split("\t") for line in urllib.request.urlopen(FOLDS).read().decode().splitlines() if line)
    (AUDIO / "folds.json").write_text(json.dumps(folds))
    rows = [r for r in csv.DictReader((ASAP / "metadata.csv").open()) if r["maestro_audio_performance"] and perf_id(r) in folds]
    if "--all" not in sys.argv:
        seen, keep = set(), []
        for r in rows:
            if r["folder"] not in seen:
                seen.add(r["folder"])
                keep.append(r)
        rows = keep
    rows = [r for r in rows if not (AUDIO / f"{perf_id(r)}.flac").exists()]
    print(f"fetching {len(rows)} performances", flush=True)

    def fetch(r: dict) -> None:
        zf = zipfile.ZipFile(io.BufferedReader(RangeFile(URL), buffer_size=1 << 20))
        member = r["maestro_audio_performance"].replace("{maestro}", "maestro-v3.0.0")
        data = zf.read(member)
        cut = ["-ss", r["start"] or "0"] + (["-to", r["end"]] if r["end"] else [])
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", "-", *cut, "-ac", "1", "-ar", "16000", str(AUDIO / f"{perf_id(r)}.flac")],
            input=data, check=True,
        )
        print("  ", perf_id(r), f"{len(data) / 1e6:.0f} MB", flush=True)

    with ThreadPoolExecutor(8) as ex:
        for f in [ex.submit(fetch, r) for r in rows]:
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                print("   failed:", e, flush=True)


if __name__ == "__main__":
    main()
