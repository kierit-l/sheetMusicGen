"""Where our barlines fall relative to the annotated beats of one ASAP performance.

    uv run python bench/diag.py Bach/Prelude/bwv_864/SunD01M
"""
import collections
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "bench")
sys.path.insert(0, ".")
from asap_bench import ASAP, OUT, entries, performance  # noqa: E402

from sheetmusicgen.pipeline import Options, Transcription, notate  # noqa: E402

pid = sys.argv[1]
e = next(e for e in entries(["--all"]) if e["midi_performance"].removesuffix(".mid") == pid)
notes, pedals, end = performance(ASAP / e["midi_performance"])
res = notate(Transcription(notes, end, ASAP / e["midi_performance"], pedals), OUT / "diag", "score", Options(pdf=False))
beats = np.array(e["ann"]["performance_beats"])
downs = set(np.round(e["ann"]["performance_downbeats"], 4))
db_idx = [i for i, b in enumerate(beats) if round(b, 4) in downs]
per_bar = int(np.median(np.diff(db_idx)))
print(res.time_sig, f"{res.bpm:.0f} bpm", res.key, "| ref", next(iter(e["ann"]["midi_score_time_signatures"].values())),
      f"{60 / np.median(np.diff(beats)):.0f} bpm, {per_bar} beats/bar")
pos = np.interp(res.downbeat_times, beats, np.arange(len(beats)))
ph = collections.Counter(np.round((pos - db_idx[0]) % per_bar, 1))
print("our barlines at annotated beat position (mod bar):", ph.most_common(6))
pos = np.interp(res.beat_times, beats, np.arange(len(beats)))
print("our beats, fraction of annotated beat:", collections.Counter(np.round(pos % 1, 1)).most_common(5))
