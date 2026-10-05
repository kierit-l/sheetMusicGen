"""Meter and downbeat choice given the annotated beats (isolates them from beat tracking).

    uv run python bench/oracle.py [folder filters]
"""
import collections
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "bench")
sys.path.insert(0, ".")
from asap_bench import ASAP, entries, f_measure, performance  # noqa: E402

from sheetmusicgen.rhythm import extend_beats, drop_detached, quantize  # noqa: E402


def run(e):
    notes, pedals, end = performance(ASAP / e["midi_performance"])
    notes = drop_detached(notes)
    ann = e["ann"]
    beats = np.array(ann["performance_beats"])
    ts = next(iter(ann["midi_score_time_signatures"].values()))[0]
    num, den = map(int, ts.split("/"))
    compound = den >= 8 and num % 3 == 0
    if den == 2:  # annotated halves -> quarters
        beats = np.sort(np.concatenate([beats, (beats[:-1] + beats[1:]) / 2]))
    if den == 8 and not compound or den == 16 and not compound:
        pass
    beats = extend_beats(beats, max(n.offset for n in notes) + 1)
    r = quantize(notes, beats, compound=compound, pedals=pedals)
    lo, hi = ann["performance_beats"][0], ann["performance_beats"][-1]
    return e["midi_performance"], ts, r.time_sig, f_measure(r.downbeat_times, ann["performance_downbeats"], lo, hi)


if __name__ == "__main__":
    todo = entries(sys.argv[1:])
    with Pool(os.cpu_count() - 1) as p:
        res = p.map(run, todo, chunksize=1)
    by = collections.defaultdict(list)
    for pid, ts, got, f in res:
        by[ts].append((got, f))
    print(f"all n={len(res)} downF {np.mean([r[3] for r in res]):.3f}")
    for ts, xs in sorted(by.items(), key=lambda x: -len(x[1])):
        print(f"  {ts:6} n={len(xs):3} downF {np.mean([f for _, f in xs]):.3f} got {collections.Counter(g for g, _ in xs).most_common()}")
