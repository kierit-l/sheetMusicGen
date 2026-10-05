"""Beat tracker variants scored at their best metrical level against ASAP beats."""
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

from sheetmusicgen import rhythm  # noqa: E402
from sheetmusicgen.rhythm import _onset_envelope, drop_detached  # noqa: E402


def levels(ref):
    ref = np.asarray(ref)
    out = {1: ref}
    for k in (2, 3):
        sub = [ref]
        for j in range(1, k):
            sub.append(ref[:-1] + (ref[1:] - ref[:-1]) * j / k)
        out[1 / k] = np.sort(np.concatenate(sub))
        out[k] = max((ref[p::k] for p in range(k)), key=len)  # any phase; scored as max below
    return out


def best_level(est, ref):
    lo, hi = ref[0], ref[-1]
    best = (0, None)
    R = np.asarray(ref)
    for lvl, r in levels(ref).items():
        cands = [r] if lvl <= 1 else [R[p::int(lvl)] for p in range(int(lvl))]
        for c in cands:
            f = f_measure(est, list(c), lo, hi)
            if f > best[0]:
                best = (f, lvl)
    return best


def variant(name, notes, last):
    import librosa
    fps = 100
    hop = 160
    env = _onset_envelope(notes, last, fps)
    if name == "current":
        return rhythm.track_beats(last - 1, notes, None)
    if name.startswith("tight"):
        t = float(name[5:])
        _, b = librosa.beat.beat_track(onset_envelope=env, sr=16000, hop_length=hop, units="time", tightness=t)
        return b
    if name.startswith("local"):
        t = float(name[5:])
        tg = librosa.feature.tempo(onset_envelope=env, sr=16000, hop_length=hop, aggregate=None, ac_size=8.0)
        tg = np.exp(np.convolve(np.log(tg), np.ones(301) / 301, mode="same"))
        _, b = librosa.beat.beat_track(onset_envelope=env, sr=16000, hop_length=hop, units="time", bpm=tg, tightness=t)
        return b
    raise ValueError(name)


VARIANTS = sys.argv[1].split(",")


def run(e):
    notes, pedals, end = performance(ASAP / e["midi_performance"])
    notes = drop_detached(notes)
    last = max(n.offset for n in notes) + 1
    ref = e["ann"]["performance_beats"]
    return {v: best_level(variant(v, notes, last), ref) for v in VARIANTS}


if __name__ == "__main__":
    todo = entries(sys.argv[2:])
    with Pool(os.cpu_count() - 1) as p:
        res = p.map(run, todo, chunksize=1)
    for v in VARIANTS:
        fs = [r[v][0] for r in res]
        print(f"{v:10} bestF {np.mean(fs):.3f}  >0.8: {np.mean(np.array(fs) > 0.8):.2f}  levels {collections.Counter(round(r[v][1], 2) for r in res if r[v][1]).most_common()}")
