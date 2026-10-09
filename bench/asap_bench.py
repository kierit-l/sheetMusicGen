"""Benchmark beat tracking, meter, key and quantization on human performances (ASAP).

    git clone --depth 1 https://github.com/fosfrancesco/asap-dataset bench/asap
    uv run python bench/asap_bench.py                 # one performance per score
    uv run python bench/asap_bench.py chopin bach     # folders containing any of these
    uv run python bench/asap_bench.py --all           # every performance (~1000)
    uv run python bench/asap_bench.py --save base     # also write bench/out/asap_base.json
    uv run python bench/asap_bench.py --diff base     # per-piece change against a saved run
    uv run python bench/asap_bench.py --audio         # beats tracked on the MAESTRO audio
                                                      # (fetch it first: bench/maestro_audio.py)
    uv run python bench/asap_bench.py --audio --onsets  # same performances, beats from the onsets
    uv run python bench/asap_bench.py --tune          # only the pieces used to tune constants
    uv run python bench/asap_bench.py --heldout       # only the pieces never used for tuning
    uv run python bench/asap_bench.py --pm2s-unseen   # only pieces the PM2S models never trained on
    uv run python bench/asap_bench.py --lookup        # beats and bars from a matched library score, where one
                                                      # fits (titled by the folder; see sheetmusicgen/lookup.py)
    SMG_HANDS=split uv run python bench/asap_bench.py # hands split at middle C, not by the PM2S model
    SMG_ORACLE_BEATS=1 uv run python bench/asap_bench.py --audio  # annotated beats and downbeats (a ceiling)

Constants in rhythm.py may only be tuned on --tune; --heldout is the honest
score. The split is by piece (every performance of a piece on the same side).

ASAP (Foscarin et al., ISMIR 2020) pairs MAESTRO performance MIDI with the
scores they play, annotated with beats, downbeats, time and key signatures.
With --audio, beats come from the beat model on the recording (as in the
app), using for each piece the cross-validation model that never saw it;
otherwise from the transcribed onsets. The performance MIDI stands in for a perfect transcription (sustain pedal
folded into the note lengths, as the transcription model reports them), so
this measures what happens after transcription, on real human timing:

  downF      downbeat F-measure: our barlines vs the annotated ones, +-70 ms
  beatF      beat F-measure at the notated beat level, +-70 ms
  meter      time signature, ok / equiv (same bar length) / wrong
  keysig     key signature (sharps or flats) matches the score's
  scoreF1    generated notes vs the score's, in quarter notes (see compare.py)
  rhythm     share of inter-onset intervals written with the score's values
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
ASAP = ROOT / "asap"
AUDIO = ROOT / "asap_audio"
OUT = ROOT / "out"
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

TOL = 0.07  # seconds, the usual beat-tracking tolerance


@dataclass
class Row:
    id: str
    meter: str
    meter_ref: str
    meter_ok: str
    keysig: int
    keysig_ref: int
    down_f: float
    beat_f: float
    tempo_ratio: float  # our beats per annotated beat (1 = same level)
    score_f1: float
    rhythm: float
    hands: float  # matched notes written on the same staff as in the score
    durations: float  # matched notes written with the score's length (times beat scale)
    short: float  # median written length / score length (< 1: notes cut short)
    meter_suggested: bool  # the score's meter is ours or one of the alternatives offered
    n_suggested: int  # how many alternatives were offered
    notes: int
    matched: str = ""  # with --lookup: the library score used, if any


def performance(path: Path):
    """Notes and sustain pedal of a performance MIDI, notes held on through the pedal."""
    import pretty_midi

    from sheetmusicgen.transcribe import Note

    pm = pretty_midi.PrettyMIDI(str(path))
    pedals, down = [], None
    for cc in sorted(pm.instruments[0].control_changes, key=lambda c: c.time):
        if cc.number != 64:
            continue
        if cc.value >= 64 and down is None:
            down = cc.time
        elif cc.value < 64 and down is not None:
            pedals.append((down, cc.time))
            down = None
    if down is not None:
        pedals.append((down, pm.get_end_time()))

    raw = sorted((n for i in pm.instruments for n in i.notes), key=lambda n: n.start)
    next_same: dict[int, float] = {}
    notes = []
    for n in reversed(raw):
        end = n.end
        for a, b in pedals:
            if a <= end < b:
                end = b
                break
        end = min(end, next_same.get(n.pitch, np.inf))
        next_same[n.pitch] = n.start
        notes.append(Note(n.start, max(end, n.start + 0.01), n.pitch, n.velocity))
    notes.reverse()
    return notes, pedals, pm.get_end_time()


def f_measure(est: list[float], ref: list[float], lo: float, hi: float) -> float:
    est = [t for t in est if lo - TOL <= t <= hi + TOL]
    if not est or not ref:
        return 0.0
    ref = np.asarray(ref)
    used = np.zeros(len(ref), bool)
    hit = 0
    for t in est:
        d = np.abs(ref - t)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= TOL:
            used[j] = True
            hit += 1
    p, r = hit / len(est), hit / len(ref)
    return 2 * p * r / (p + r) if p + r else 0.0


def score_reference(midi: Path, downbeats_sec: list[float]):
    """Notes of the score MIDI in quarter notes, flagged where a bar starts."""
    import pretty_midi

    from compare import N

    pm = pretty_midi.PrettyMIDI(str(midi))
    q = lambda t: pm.time_to_tick(t) / pm.resolution  # noqa: E731
    bars = np.array([q(t) for t in downbeats_sec])
    out = []
    for staff, inst in enumerate(pm.instruments):  # ASAP writes the treble staff first
        for n in inst.notes:
            on = q(n.start)
            end = round(q(n.end) * 48) / 48  # ASAP ends each note a tick early
            out.append(N(n.pitch, on, end, bool(len(bars)) and np.min(np.abs(bars - on)) < 1e-3, min(staff, 1)))
    return out


def patch_constants() -> None:
    """SMG_PATCH="TEMPO_CHANGE_COST=100,BEAT_THRESHOLD=0.3" overrides rhythm.py constants (for tuning)."""
    from sheetmusicgen import lookup, notation, rhythm

    for kv in filter(None, os.environ.get("SMG_PATCH", "").split(",")):
        k, v = kv.split("=")
        mod = next(m for m in (rhythm, lookup, notation) if hasattr(m, k))
        setattr(mod, k, type(getattr(mod, k))(float(v)))


def evaluate(entry: dict) -> Row | str:
    patch_constants()
    from compare import N, duration_scores, match, meter_verdict, prf, score_notes, snap_scale, warp

    from sheetmusicgen.pipeline import Options, Transcription, notate

    perf_path = ASAP / entry["midi_performance"]
    ann = entry["ann"]
    try:
        notes, pedals, end = performance(perf_path)
        t = Transcription(notes, end, perf_path, pedals)
        if entry.get("act"):
            from sheetmusicgen.rhythm import decode_beats
            from sheetmusicgen.transcribe import BEAT_FPS

            a = np.load(entry["act"])
            beats, probs = decode_beats(a["beat"], a["down"], BEAT_FPS)
            # Audio cut out of a longer MAESTRO recording starts at the first
            # note, but ASAP's MIDI has a lead-in before it; uncut audio lines up.
            lead = min(n.onset for n in notes) if entry["start"] else 0.0
            t.beats, t.downbeat_probs = [b + lead for b in beats], probs
        if os.environ.get("SMG_ORACLE_BEATS"):  # the annotated beats and downbeats, as if tracked perfectly
            downs = set(ann["performance_downbeats"])
            t.beats = list(ann["performance_beats"])
            t.downbeat_probs = [0.95 if b in downs else 0.05 for b in t.beats]
        slug = entry["midi_performance"].removesuffix(".mid").replace("/", "_")
        reference = None
        if entry.get("lookup"):
            from sheetmusicgen.lookup import find

            reference = find(notes, entry["folder"].replace("/", " ").replace("_", " "))
        opts = Options(title=slug, pdf=False, hands=os.environ.get("SMG_HANDS", "model"))
        res = notate(t, OUT / "asap" / slug, "score", opts, reference=reference)
    except Exception as e:  # noqa: BLE001
        return f"{entry['midi_performance']}: {type(e).__name__}: {e}"

    beats, downs = ann["performance_beats"], ann["performance_downbeats"]
    lo, hi = beats[0], beats[-1]
    down_f = f_measure(res.downbeat_times, downs, lo, hi)
    beat_f = f_measure(res.beat_times, beats, lo, hi)
    ours = [b for b in res.beat_times if lo <= b <= hi]
    tempo_ratio = (len(ours) / len(beats)) if beats else 0.0

    meter_ref = next(iter(ann["midi_score_time_signatures"].values()))[0]
    keysig_ref = next(iter(ann["perf_key_signatures"].values()))[1]
    from music21 import key as m21key

    tonic, mode = res.key.split()
    keysig = m21key.Key(tonic.replace("b", "-") if len(tonic) > 1 else tonic, mode).sharps

    ref = score_reference(ASAP / entry["midi_score"], ann["midi_score_downbeats"])
    gen = score_notes(res.musicxml)
    to_gen, (xs, ys) = warp(ref, gen, fps=8)
    a, b = len(ys) // 10, -max(1, len(ys) // 10)
    scale = snap_scale(float((ys[b] - ys[a]) / max(xs[b] - xs[a], 1e-6)))
    pairs = match(ref, gen, to_gen, tol=0.25 * scale)
    _, _, sf = prf(len(pairs), len(ref), len(gen))
    hands = float(np.mean([ref[i].staff == gen[j].staff for i, j in pairs])) if pairs else 0.0
    durations, short = duration_scores(ref, gen, pairs, scale)
    onsets = sorted({(round(ref[i].onset, 4), round(gen[j].onset, 4)) for i, j in pairs})
    good = total = 0
    for (r0, g0), (r1, g1) in zip(onsets, onsets[1:]):
        if r1 - r0 > 1e-6:
            total += 1
            good += abs((g1 - g0) - scale * (r1 - r0)) < 1e-3
    return Row(
        id=entry["midi_performance"].removesuffix(".mid"),
        meter=res.time_sig,
        meter_ref=meter_ref,
        meter_ok=meter_verdict(res.time_sig, meter_ref),
        keysig=keysig,
        keysig_ref=keysig_ref,
        down_f=round(down_f, 4),
        beat_f=round(beat_f, 4),
        tempo_ratio=round(tempo_ratio, 3),
        score_f1=round(sf, 4),
        rhythm=round(good / total if total else 0.0, 4),
        hands=round(hands, 4),
        durations=round(durations, 4),
        short=round(short, 3),
        meter_suggested=any(meter_verdict(m, meter_ref) in ("ok", "equiv") for m in [res.time_sig, *res.other_meters]),
        n_suggested=len(res.other_meters),
        notes=len(notes),
        matched=reference.id if reference else "",
    )


def entries(args: list[str]) -> list[dict]:
    ann = json.loads((ASAP / "asap_annotations.json").read_text())
    rows = list(csv.DictReader((ASAP / "metadata.csv").open()))
    values = {args[i + 1] for i, a in enumerate(args[:-1]) if a in ("--save", "--diff")}
    want = [a.lower() for a in args if not a.startswith("--") and a not in values]
    seen, out = set(), []
    for r in rows:
        a = ann.get(r["midi_performance"])
        if not a or a.get("score_and_performance_aligned") is False:
            continue
        if want and not any(w in r["folder"].lower() for w in want):
            continue
        if "--all" not in args and r["folder"] in seen:
            continue
        seen.add(r["folder"])
        out.append({**r, "ann": a})
    return out


def held_out(entry: dict) -> bool:
    """About a third of the pieces, chosen by a stable hash of the piece's folder."""
    return int(hashlib.md5(entry["folder"].encode()).hexdigest(), 16) % 3 == 0


def perf_id(entry: dict) -> str:
    return entry["midi_performance"].removesuffix(".mid").replace("/", "_")


def attach_activations(todo: list[dict]) -> list[dict]:
    """Keep the performances with audio and cache their beat activations, from the held-out fold's model."""
    from sheetmusicgen.transcribe import beat_activations, load_audio

    folds = json.loads((AUDIO / "folds.json").read_text())
    out = []
    for e in todo:
        flac = AUDIO / f"{perf_id(e)}.flac"
        if not flac.exists():
            continue
        act = AUDIO / f"{perf_id(e)}.act.npz"
        if not act.exists():
            beat, down = beat_activations(load_audio(flac), f"fold{folds[perf_id(e)]}")
            np.savez(act, beat=beat, down=down)
        out.append({**e, "act": str(act)})
    return out


def main() -> None:
    args = sys.argv[1:]
    if "--audio" in args:
        # one performance per piece among those with audio
        todo = [e for e in entries(args + ["--all"]) if (AUDIO / f"{perf_id(e)}.flac").exists()]
        if "--all" not in args:
            todo = list({e["folder"]: e for e in reversed(todo)}.values())[::-1]
        todo = attach_activations(todo)
        if "--onsets" in args:  # same performances, beats tracked on the onsets
            todo = [{k: v for k, v in e.items() if k != "act"} for e in todo]
    else:
        todo = entries(args)
    if "--pm2s-unseen" in args:  # pieces PM2S did not train on (its validation split)
        unseen = set((ROOT / "pm2s_unseen.txt").read_text().split())
        todo = [e for e in todo if e["folder"] in unseen]
    if "--tune" in args:
        todo = [e for e in todo if not held_out(e)]
    elif "--heldout" in args:
        todo = [e for e in todo if held_out(e)]
    if "--lookup" in args:
        todo = [{**e, "lookup": True} for e in todo]
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        results = pool.map(evaluate, todo, chunksize=1)
    rows = [r for r in results if isinstance(r, Row)]
    for e in results:
        if isinstance(e, str):
            print("ERROR", e)

    def mean(f: str) -> float:
        return float(np.mean([getattr(r, f) for r in rows])) if rows else 0.0

    meter_ok = sum(r.meter_ok == "ok" for r in rows)
    meter_eq = sum(r.meter_ok in ("ok", "equiv") for r in rows)
    key_ok = sum(r.keysig % 12 == r.keysig_ref % 12 for r in rows)
    level_ok = sum(abs(np.log2(max(r.tempo_ratio, 1e-3))) < 0.2 for r in rows)
    print(
        f"{len(rows)} performances  downF {mean('down_f'):.3f}  beatF {mean('beat_f'):.3f}  "
        f"beat level {level_ok}/{len(rows)}  meter {meter_ok} ok, {meter_eq} ok|equiv  keysig {key_ok}  "
        f"scoreF1 {mean('score_f1'):.3f}  rhythm {mean('rhythm'):.3f}  hands {mean('hands'):.3f}  durations {mean('durations'):.3f}  "
        f"meter in suggestions {sum(r.meter_suggested for r in rows)} (mean {mean('n_suggested'):.2f} offered)"
    )

    if "--lookup" in args:
        for name, rs in (("matched", [r for r in rows if r.matched]), ("unmatched", [r for r in rows if not r.matched])):
            if rs:
                print(
                    f"  {name:9} n={len(rs):3}  downF {np.mean([r.down_f for r in rs]):.3f}  beatF {np.mean([r.beat_f for r in rs]):.3f}  "
                    f"beat level {sum(abs(np.log2(max(r.tempo_ratio, 1e-3))) < 0.2 for r in rs)}  "
                    f"meter {sum(r.meter_ok == 'ok' for r in rs)} ok, {sum(r.meter_ok in ('ok', 'equiv') for r in rs)} ok|equiv  "
                    f"rhythm {np.mean([r.rhythm for r in rs]):.3f}  durations {np.mean([r.durations for r in rs]):.3f}"
                )

    by_meter: dict[str, list[Row]] = {}
    for r in rows:
        by_meter.setdefault(r.meter_ref, []).append(r)
    for m, rs in sorted(by_meter.items(), key=lambda x: -len(x[1])):
        print(
            f"  {m:6} n={len(rs):3}  downF {np.mean([r.down_f for r in rs]):.3f}  "
            f"beatF {np.mean([r.beat_f for r in rs]):.3f}  meter ok {sum(r.meter_ok == 'ok' for r in rs)}  "
            f"rhythm {np.mean([r.rhythm for r in rs]):.3f}  "
            f"got {dict(sorted({x: sum(r.meter == x for r in rs) for x in {r.meter for r in rs}}.items()))}"
        )

    if "--diff" in args:
        old = {r["id"]: r for r in json.loads((OUT / f"asap_{args[args.index('--diff') + 1]}.json").read_text())}
        changes = []
        for r in rows:
            o = old.get(r.id)
            if o:
                changes.append((r.down_f - o["down_f"], r.rhythm - o["rhythm"], r.id, o, r))
        changes.sort()
        print("\nbiggest downF losses / gains:")
        for c in changes[:8] + changes[-8:]:
            d, dr, i, o, r = c
            print(f"  {d:+.3f} rhythm {dr:+.3f}  {i:60} {o['meter']}->{r.meter} ({r.meter_ref})")
        print(f"  improved {sum(c[0] > 0.02 for c in changes)}, worse {sum(c[0] < -0.02 for c in changes)}")

    if "--save" in args:
        OUT.mkdir(exist_ok=True)
        (OUT / f"asap_{args[args.index('--save') + 1]}.json").write_text(json.dumps([asdict(r) for r in rows], indent=0))


if __name__ == "__main__":
    main()
