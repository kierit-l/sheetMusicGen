"""Benchmark the whole app on the Vienna 4x22 Piano Corpus: 22 pianists x 4 excerpts, from the audio.

    git clone --depth 1 https://github.com/CPJKU/vienna4x22 bench/v4x22/repo
    curl -L -o bench/v4x22/dl/audio.zip \\
        https://repo.mdw.ac.at/projects/IWK/the_vienna_4x22_piano_corpus/data/audio.zip   # 1.3 GB
    uv run python bench/v4x22_bench.py --transcribe   # unzip, transcribe and track beats once (cached)
    uv run python bench/v4x22_bench.py                # notes and beats from the audio, as in the app
    uv run python bench/v4x22_bench.py --midi         # the performance MIDI as a perfect transcription
    uv run python bench/v4x22_bench.py --lookup       # beats and bars from a matched library score
    uv run python bench/v4x22_bench.py --save NAME / --diff NAME

None of the models saw this corpus: beat_this trained on other datasets
(see CPJKU/beat_this_annotations), PM2S on ASAP, A-MAPS and CPM, the
transcription model on MAESTRO. Nothing here is tuned on it either, so all of
it is held out. The recordings were made on a Boesendorfer SE in 1999; the
match files align every performed note to the score (Goebl 1999, CC BY 4.0),
and give the beats and bars: each score position's time is the mean onset of
the notes played there, interpolated between them. Beats are counted as in
ASAP (dotted quarters in 6/8). Metrics are those of asap_bench.py, plus:

  transF1    transcribed notes vs the performance MIDI (pitch, onset +-50 ms)
"""

from __future__ import annotations

import json
import os
import re
import sys
import warnings
import zipfile
from dataclasses import asdict
from fractions import Fraction
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
DATA = ROOT / "v4x22"
REPO = DATA / "repo"
AUDIO = DATA / "audio"
CACHE = DATA / "cache"
OUT = ROOT / "out"
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

TITLES = {  # what a user would type as the title, for --lookup
    "Chopin_op10_no3": "Chopin Etude Op. 10 No. 3",
    "Chopin_op38": "Chopin Ballade No. 2 Op. 38",
    "Mozart_K331_1st-mov": "Mozart Sonata KV 331 Tema",
    "Schubert_D783_no15": "Schubert Deutsche Taenze D. 783 No. 15",
}
STEPS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
ALTER = {"n": 0, "#": 1, "b": -1, "##": 2, "bb": -2}
SNOTE = re.compile(
    r"snote\([^,]+,\[([A-G]),([^\]]+)\],(-?\d+),(-?\d+):[^,]+,[^,]+,[^,]+,(-?[\d.]+),(-?[\d.]+),\[([^\]]*)\]\)"
    r"-(?:note\([^,]+,(\d+),(\d+),(\d+),|deletion)"
)


def parse_match(path: Path) -> dict:
    """Score notes (quarter notes), their performed onsets (seconds), meter and key from a match file."""
    text = path.read_text()
    unit = int(re.search(r"info\(midiClockUnits,(\d+)\)", text).group(1))
    rate = int(re.search(r"info\(midiClockRate,(\d+)\)", text).group(1))
    num, den = map(int, re.search(r"scoreprop\(timeSignature,(\d+/\d+),", text).group(1).split("/"))
    key = re.search(r"scoreprop\(keySignature,(\w+),", text).group(1)
    notes = []
    for step, acc, octave, bar, on_b, off_b, attrs, pitch, tick, _ in SNOTE.findall(text):
        q = 4 / den  # match files count score time in beats of the time signature
        notes.append(
            {
                "pitch": (int(octave) + 1) * 12 + STEPS[step] + ALTER[acc],
                "onset": float(on_b) * q,
                "offset": float(off_b) * q,
                "staff": 1 if "staff2" in attrs else 0,
                "grace": "grace" in attrs or float(off_b) == float(on_b),
                "played": float(tick) * rate / unit / 1e6 if tick else None,
                "performed_pitch": int(pitch) if pitch else None,
            }
        )
    return {"num": num, "den": den, "key": key, "notes": notes}


def annotations(m: dict) -> tuple[list[float], list[float], str]:
    """Beat and downbeat times (seconds) of the performance, and the time signature."""
    num, den = m["num"], m["den"]
    compound = num % 3 == 0 and num > 3
    beat = Fraction(12 if compound else 4, den)
    bar = Fraction(4 * num, den)
    pos: dict[float, list[float]] = {}
    for n in m["notes"]:
        if n["played"] is not None and not n["grace"]:
            pos.setdefault(round(n["onset"], 4), []).append(n["played"])
    xs = np.array(sorted(pos))
    ts = np.array([np.mean(pos[x]) for x in xs])
    ts = np.maximum.accumulate(ts)  # a stray early note can't run time backwards
    first = Fraction(xs[0]).limit_denominator(48)
    k0 = -(-first // beat)
    beats_q = [float(k * beat) for k in range(int(k0), int(Fraction(xs[-1]).limit_denominator(48) // beat) + 1)]
    beats = np.interp(beats_q, xs, ts).tolist()
    downs = [t for q, t in zip(beats_q, beats) if q >= 0 and Fraction(q).limit_denominator(48) % bar == 0]
    return beats, downs, f"{num}/{den}"


def performances() -> list[str]:
    return sorted(p.stem for p in (REPO / "match").glob("*.match"))


def transcribe_all() -> None:
    from sheetmusicgen.pipeline import transcribe_file

    if not AUDIO.exists():
        with zipfile.ZipFile(DATA / "dl" / "audio.zip") as z:
            want = set(performances())
            for info in z.infolist():
                name = Path(info.filename).name
                if name.endswith(".wav") and Path(name).stem in want and not info.filename.startswith("__MACOSX"):
                    AUDIO.mkdir(parents=True, exist_ok=True)
                    (AUDIO / name).write_bytes(z.read(info))
    for pid in performances():
        wav = AUDIO / f"{pid}.wav"
        if not wav.exists():
            print("no audio for", pid)
            continue
        if (CACHE / pid / f"{pid}.notes.json").exists():
            continue
        print("transcribing", pid, flush=True)
        transcribe_file(wav, CACHE / pid)


def midi_notes(path: Path):
    """The performance MIDI as a perfect transcription: notes held on through the sustain pedal."""
    from asap_bench import performance

    return performance(path)


def note_f1(est, ref, tol=0.05) -> float:
    from compare import N, match, prf

    a = [N(n.pitch, n.onset, n.offset) for n in est]
    b = [N(n.pitch, n.onset, n.offset) for n in ref]
    pairs = match(b, a, lambda t: np.asarray(t), tol)
    return prf(len(pairs), len(b), len(a))[2]


def evaluate(args: tuple[str, bool, bool]):
    pid, use_midi, lookup = args
    from asap_bench import Row, f_measure
    from compare import N, duration_scores, match, meter_verdict, prf, score_notes, snap_scale, warp
    from music21 import key as m21key

    from sheetmusicgen.pipeline import Options, Transcription, notate

    try:
        m = parse_match(REPO / "match" / f"{pid}.match")
        beats, downs, meter_ref = annotations(m)
        audio_t = Transcription.load(CACHE / pid / f"{pid}.notes.json")
        perf, pedals, end = midi_notes(REPO / "midi" / f"{pid}.mid")
        trans_f1 = note_f1(audio_t.notes, perf)
        t = audio_t
        if use_midi:  # same beats (from the audio), perfect notes
            t = Transcription(perf, max(end, audio_t.duration), audio_t.raw_midi, pedals, audio_t.beats, audio_t.downbeat_probs)
        piece = pid.rsplit("_p", 1)[0]
        reference = None
        if lookup:
            from sheetmusicgen.lookup import find

            reference = find(t.notes, TITLES[piece])
        res = notate(t, OUT / "v4x22" / pid, "score", Options(title=pid, pdf=False), reference=reference)
    except Exception as e:  # noqa: BLE001
        return f"{pid}: {type(e).__name__}: {e}"

    lo, hi = beats[0], beats[-1]
    ours = [b for b in res.beat_times if lo <= b <= hi]
    tonic, mode = res.key.split()
    keysig = m21key.Key(tonic.replace("b", "-") if len(tonic) > 1 else tonic, mode).sharps
    ref_key = m["key"]
    keysig_ref = m21key.Key(ref_key[0] if not ref_key.endswith("m") else ref_key[:-1].lower()).sharps

    bar = 4 * m["num"] / m["den"]
    ref = [
        N(n["pitch"], n["onset"], n["offset"], abs(n["onset"] / bar - round(n["onset"] / bar)) < 1e-3 and n["onset"] >= 0, n["staff"])
        for n in m["notes"]
        if not n["grace"]
    ]
    shift = min(n.onset for n in ref)  # a pickup starts before 0
    ref = [N(n.pitch, n.onset - shift, n.offset - shift, n.downbeat, n.staff) for n in ref]
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
    row = Row(
        id=pid,
        meter=res.time_sig,
        meter_ref=meter_ref,
        meter_ok=meter_verdict(res.time_sig, meter_ref),
        keysig=keysig,
        keysig_ref=keysig_ref,
        down_f=round(f_measure(res.downbeat_times, downs, lo, hi), 4),
        beat_f=round(f_measure(res.beat_times, beats, lo, hi), 4),
        tempo_ratio=round(len(ours) / len(beats), 3),
        score_f1=round(sf, 4),
        rhythm=round(good / total if total else 0.0, 4),
        hands=round(hands, 4),
        durations=round(durations, 4),
        short=round(short, 3),
        meter_suggested=any(meter_verdict(x, meter_ref) in ("ok", "equiv") for x in [res.time_sig, *res.other_meters]),
        n_suggested=len(res.other_meters),
        notes=len(t.notes),
        matched=reference.id if reference else "",
    )
    return row, trans_f1


def main() -> None:
    args = sys.argv[1:]
    if "--transcribe" in args:
        transcribe_all()
        return
    todo = [(p, "--midi" in args, "--lookup" in args) for p in performances() if (CACHE / p / f"{p}.notes.json").exists()]
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        results = pool.map(evaluate, todo, chunksize=1)
    for r in results:
        if isinstance(r, str):
            print("ERROR", r)
    done = [r for r in results if not isinstance(r, str)]
    rows = [r for r, _ in done]
    tf1 = {r.id: f for r, f in done}

    def line(name: str, rs: list) -> str:
        mean = lambda f: float(np.mean([getattr(r, f) for r in rs]))  # noqa: E731
        level = sum(abs(np.log2(max(r.tempo_ratio, 1e-3))) < 0.2 for r in rs)
        return (
            f"{name:20} n={len(rs):3}  transF1 {np.mean([tf1[r.id] for r in rs]):.3f}  downF {mean('down_f'):.3f}  "
            f"beatF {mean('beat_f'):.3f}  level {level}/{len(rs)}  meter {sum(r.meter_ok == 'ok' for r in rs)} ok, "
            f"{sum(r.meter_ok in ('ok', 'equiv') for r in rs)} ok|equiv  keysig {sum(r.keysig == r.keysig_ref for r in rs)}  "
            f"scoreF1 {mean('score_f1'):.3f}  rhythm {mean('rhythm'):.3f}  hands {mean('hands'):.3f}  durations {mean('durations'):.3f}"
            + (f"  matched {sum(bool(r.matched) for r in rs)}" if "--lookup" in args else "")
        )

    if not rows:
        return
    print(line("all", rows))
    for piece in sorted({r.id.rsplit("_p", 1)[0] for r in rows}):
        rs = [r for r in rows if r.id.startswith(piece + "_p")]
        got = {}
        for r in rs:
            got[r.meter] = got.get(r.meter, 0) + 1
        print(line(piece, rs), f"({rs[0].meter_ref}; got {dict(sorted(got.items(), key=lambda x: -x[1]))})")

    if "--diff" in args:
        old = {r["id"]: r for r in json.loads((OUT / f"v4x22_{args[args.index('--diff') + 1]}.json").read_text())}
        ch = sorted((r.down_f - old[r.id]["down_f"], r.rhythm - old[r.id]["rhythm"], r.id) for r in rows if r.id in old)
        print(f"downF improved {sum(c[0] > 0.02 for c in ch)}, worse {sum(c[0] < -0.02 for c in ch)}; biggest:", ch[:3], ch[-3:])
    if "--save" in args:
        OUT.mkdir(exist_ok=True)
        (OUT / f"v4x22_{args[args.index('--save') + 1]}.json").write_text(json.dumps([asdict(r) for r in rows], indent=0))


if __name__ == "__main__":
    main()
