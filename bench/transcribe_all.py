"""Transcribe every downloaded benchmark recording once (cached as notes.json)."""

import sys
from pathlib import Path

from sheetmusicgen.pipeline import transcribe_file

cache = Path(__file__).parent / "cache"
for d in sorted(p for p in cache.iterdir() if p.is_dir()):
    if (d / "audio.notes.json").exists():
        continue
    audio = next(d.glob("audio.*"))
    print(f"transcribing {d.name}", flush=True)
    transcribe_file(audio, d, progress=lambda m: print("  ", m, flush=True))
print("done", flush=True)
