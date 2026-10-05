"""Compare sheetmusicgen output against reference scores for real recordings.

    uv run python bench/compare.py              # all pieces in bench/pieces.json
    uv run python bench/compare.py satie bach   # pieces whose id contains any of these
    uv run python bench/compare.py --lookup     # beats and bars from the matched library score
                                                # (the references are Mutopia scores too: an upper bound)

Each piece is a YouTube performance plus a public-domain reference score (a
Mutopia Project MIDI rendered from LilyPond, so it is note-for-note the score).
Audio is downloaded and transcribed once into bench/cache/<id>/; notation is
re-run every time, so the numbers track changes to rhythm.py / notation.py.

Performance timing never matches the score, so both comparisons go through
dynamic time warping on chroma:

  transcription F1   raw transcribed notes (seconds) vs the reference warped
                     onto the performance; same pitch, onset within 150 ms.
  score F1           notes of the generated MusicXML vs the reference, both in
                     quarter notes, warped onto each other; same pitch, onset
                     within a 16th (in reference units).
  beat scale         generated quarters per reference quarter (1 = same note
                     values; 2 = everything written twice as long).
  rhythm acc         for consecutive matched onsets: does the generated
                     inter-onset interval equal beat scale x the reference one?
  barlines           of the matched notes that start a bar in the score with
                     the longer bars, the share that also start a bar in the
                     other score (1 = barlines in the right place).
"""

from __future__ import annotations

import io
import json
import sys
import urllib.request
import warnings
import zipfile
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
HANDS = "split" if "--hands-split" in sys.argv else "model"
CACHE = ROOT / "cache"
OUT = ROOT / "out"
sys.path.insert(0, str(ROOT.parent))
warnings.filterwarnings("ignore", message="Tempo, Key or Time signature")

ENHARMONIC = {"Db": "C#", "D#": "Eb", "Gb": "F#", "G#": "Ab", "A#": "Bb", "Cb": "B", "Fb": "E", "E#": "F", "B#": "C"}
RELATIVE_STEPS = 3  # relative minor is 3 semitones below the major tonic
PC = {"C": 0, "C#": 1, "D": 2, "Eb": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11}


# ---------------------------------------------------------------- inputs


def fetch(piece: dict) -> Path:
    """Download the recording and reference score into the cache (once)."""
    from sheetmusicgen.youtube import download_audio

    d = CACHE / piece["id"]
    d.mkdir(parents=True, exist_ok=True)
    ref = d / "reference.mid"
    if not ref.exists():
        data = urllib.request.urlopen(piece["reference"]).read()
        if piece["reference"].endswith(".zip"):
            data = zipfile.ZipFile(io.BytesIO(data)).read(piece["reference_member"])
        ref.write_bytes(data)
    if not list(d.glob("audio.*")):
        path, title = download_audio(piece["youtube"], d)
        path.rename(d / f"audio{path.suffix}")
        (d / "title.txt").write_text(title)
    return d


@dataclass
class N:
    pitch: int
    onset: float  # seconds or quarter notes, depending on the source
    offset: float
    downbeat: bool = False
    staff: int = 0  # 0 = treble (right hand), 1 = bass (left hand)


def reference_notes(path: Path, pickup: float) -> tuple[list[N], list[N], str]:
    """Reference notes in seconds and in quarter notes, plus its first time signature."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(path))
    res = pm.resolution
    q = lambda t: pm.time_to_tick(t) / res  # noqa: E731
    # measure starts in quarters, from the time signature changes; LilyPond's
    # MIDI starts at the pickup, so the first full bar begins at `pickup`.
    tss = [(q(ts.time), ts.numerator * 4 / ts.denominator) for ts in pm.time_signature_changes] or [(0.0, 4.0)]
    end = q(pm.get_end_time())
    bars, pos = [], pickup
    for i, (start, length) in enumerate(tss):
        stop = tss[i + 1][0] if i + 1 < len(tss) else end + length
        pos = max(pos, start)
        while pos < stop - 1e-6:
            bars.append(pos)
            pos += length
    bars = np.array(bars)

    secs, quarters = [], []
    insts = [i for i in pm.instruments if not i.is_drum and i.notes]
    for inst in insts:
        staff = int(np.mean([n.pitch for n in inst.notes]) < max(np.mean([n.pitch for n in i.notes]) for i in insts))
        for n in inst.notes:
            on = q(n.start)
            db = bool(len(bars)) and np.min(np.abs(bars - on)) < 1e-3
            secs.append(N(n.pitch, n.start, n.end, db, staff))
            quarters.append(N(n.pitch, on, q(n.end), db, staff))
    ts = pm.time_signature_changes[0]
    return secs, quarters, f"{ts.numerator}/{ts.denominator}"


def score_notes(musicxml: Path) -> list[N]:
    """Notes of a generated MusicXML in quarter notes from the start, ties merged.

    Ties are merged here rather than with music21's stripTies, which leaves
    tied chords inside voices apart.
    """
    from music21 import converter

    score = converter.parse(str(musicxml))
    out = []
    for staff, part in enumerate(score.parts):
        open_ties: dict[int, N] = {}  # pitch -> note a tie continues
        for m in part.getElementsByClass("Measure"):
            bar = m.barDuration.quarterLength
            pad = bar - m.duration.quarterLength if m.number == 0 or (m.paddingLeft or 0) else 0
            for n in sorted(m.recurse().notes, key=lambda n: n.getOffsetInHierarchy(m)):
                if n.duration.isGrace:
                    continue
                rel = float(n.getOffsetInHierarchy(m))
                on = float(m.offset) + rel
                end = on + float(n.duration.quarterLength)
                for p in n.pitches:
                    tie = (p.tie or n.tie) if hasattr(p, "tie") else n.tie
                    kind = tie.type if tie else None
                    held = open_ties.get(p.midi)
                    if kind in ("stop", "continue") and held is not None and abs(held.offset - on) < 1e-6:
                        held.offset = end
                    else:
                        held = N(p.midi, on, end, abs(rel + pad) < 1e-6, staff)
                        out.append(held)
                    if kind in ("start", "continue"):
                        open_ties[p.midi] = held
                    else:
                        open_ties.pop(p.midi, None)
    return out


# ---------------------------------------------------------------- alignment


def chroma(notes: list[N], fps: float, length: float) -> np.ndarray:
    frames = int(np.ceil(length * fps)) + 1
    c = np.zeros((12, frames))
    for n in notes:
        a = int(n.onset * fps)
        b = max(a + 1, int(n.offset * fps))
        c[n.pitch % 12, a:b] += 1.0
        c[n.pitch % 12, a] += 2.0  # onsets carry most of the timing information
    return c / (np.linalg.norm(c, axis=0, keepdims=True) + 1e-3)


def warp(src: list[N], dst: list[N], fps: float):
    """DTW from src time to dst time; returns a function mapping src times to dst times."""
    import librosa

    end_s = max(n.offset for n in src)
    end_d = max(n.offset for n in dst)
    X, Y = chroma(src, fps, end_s), chroma(dst, fps, end_d)
    _, wp = librosa.sequence.dtw(X=X, Y=Y, metric="euclidean")
    wp = wp[::-1]
    xs = np.unique(wp[:, 0])
    ys = np.array([wp[wp[:, 0] == x, 1].mean() for x in xs])
    return lambda t: np.interp(np.asarray(t) * fps, xs, ys) / fps, (xs / fps, ys / fps)


def match(ref: list[N], est: list[N], ref_to_est, tol: float) -> list[tuple[int, int]]:
    """Greedy one-to-one matching on pitch and warped onset distance."""
    warped = ref_to_est([n.onset for n in ref])
    by_pitch: dict[int, list[int]] = {}
    for j, n in enumerate(est):
        by_pitch.setdefault(n.pitch, []).append(j)
    cands = []
    for i, n in enumerate(ref):
        for j in by_pitch.get(n.pitch, []):
            d = abs(est[j].onset - warped[i])
            if d <= tol:
                cands.append((d, i, j))
    cands.sort()
    used_r, used_e, pairs = set(), set(), []
    for _, i, j in cands:
        if i not in used_r and j not in used_e:
            used_r.add(i)
            used_e.add(j)
            pairs.append((i, j))
    return pairs


def duration_scores(ref: list[N], gen: list[N], pairs, scale: float) -> tuple[float, float]:
    """Share of matched notes with the reference's written length, and the median length ratio."""
    if not pairs:
        return 0.0, 0.0
    ratio = np.array([(gen[j].offset - gen[j].onset) / (scale * (ref[i].offset - ref[i].onset)) for i, j in pairs])
    return float(np.mean(np.abs(ratio - 1) < 1e-3)), float(np.median(ratio))


def prf(n_match: int, n_ref: int, n_est: int) -> tuple[float, float, float]:
    p = n_match / n_est if n_est else 0.0
    r = n_match / n_ref if n_ref else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


# ---------------------------------------------------------------- metrics


def norm_key(k: str) -> tuple[int, str]:
    tonic, mode = k.split()
    tonic = tonic.replace("-", "b")
    return PC[ENHARMONIC.get(tonic, tonic)], mode


def key_verdict(est: str, ref: str) -> str:
    (te, me), (tr, mr) = norm_key(est), norm_key(ref)
    if (te, me) == (tr, mr):
        return "ok"
    if me != mr and (te - tr) % 12 == (RELATIVE_STEPS if mr == "major" else -RELATIVE_STEPS) % 12:
        return "relative"
    if te == tr:
        return "parallel"
    if (te - tr) % 12 in (5, 7) and me == mr:
        return "fifth"
    return "wrong"


def meter_verdict(est: str, ref: str) -> str:
    if est == ref:
        return "ok"
    ne, de = map(int, est.split("/"))
    nr, dr = map(int, ref.split("/"))
    if Fraction(ne, de) == Fraction(nr, dr):
        return "equiv"  # 2/2 vs 4/4, 6/8 vs 3/4 bar length
    compound = lambda n, d: d == 8 and n % 3 == 0  # noqa: E731
    if compound(ne, de) == compound(nr, dr) and Fraction(ne, de) / Fraction(nr, dr) in (2, Fraction(1, 2)):
        return "half/double"
    return "wrong"


def snap_scale(x: float) -> float:
    choices = np.array([0.25, 1 / 3, 0.5, 2 / 3, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
    return float(choices[np.argmin(np.abs(np.log(choices / x)))])


@dataclass
class Report:
    id: str
    key: str
    key_ref: str
    key_ok: str
    meter: str
    meter_ref: str
    meter_ok: str
    bpm: float
    trans_p: float
    trans_r: float
    trans_f1: float
    score_p: float
    score_r: float
    score_f1: float
    beat_scale: float
    rhythm_acc: float
    barlines: float
    hands: float  # matched notes written on the same staff as in the reference
    durations: float  # matched notes written with the reference's length (times beat scale)
    short: float  # median written length / reference length (< 1: notes cut short)
    notes_ref: int
    notes_score: int
    dropped: int
    form: str  # which reference fit the performance: "printed" or "unfolded" repeats


def evaluate(piece: dict) -> Report:
    from sheetmusicgen.pipeline import Options, Transcription, notate

    d = fetch(piece)
    notes_json = d / "audio.notes.json"
    if not notes_json.exists():
        from sheetmusicgen.pipeline import transcribe_file

        transcribe_file(next(d.glob("audio.*")), d)
    t = Transcription.load(notes_json)
    # Beat activations are cached; decoding runs every time, to track changes to it.
    from sheetmusicgen.rhythm import decode_beats
    from sheetmusicgen.transcribe import BEAT_FPS, beat_activations, load_audio

    act = d / "act.npz"
    if not act.exists():
        np.savez(act, *beat_activations(load_audio(next(d.glob("audio.*")))))
    a = np.load(act)
    t.beats, t.downbeat_probs = decode_beats(a["arr_0"], a["arr_1"], BEAT_FPS)
    out = OUT / piece["id"]
    reference = None
    if "--lookup" in sys.argv:  # identify the piece from the video title, as the app would
        from sheetmusicgen.lookup import find

        reference = find(t.notes, (d / "title.txt").read_text() if (d / "title.txt").exists() else "")
    opts = Options(title=piece["id"], pdf="--pdf" in sys.argv, hands=HANDS)
    res = notate(t, out, piece["id"], opts, reference=reference)

    # transcription: reference (score seconds) -> performance seconds. The
    # performer may or may not take the repeats, so try the score as printed
    # and with repeats unfolded (bench/unfold_refs.py) and keep the better fit.
    perf = [N(n.pitch, n.onset, n.offset) for n in t.notes]
    best = None
    for form, name in (("printed", "reference.mid"), ("unfolded", "reference_unfolded.mid")):
        if not (d / name).exists():
            continue
        ref_s, ref_q, meter_ref = reference_notes(d / name, piece.get("pickup", 0))
        to_perf, _ = warp(ref_s, perf, fps=20)
        pairs = match(ref_s, perf, to_perf, tol=0.15)
        tp, tr, tf = prf(len(pairs), len(ref_s), len(perf))
        if best is None or tf > best[0][2]:
            best = ((tp, tr, tf), ref_s, ref_q, meter_ref, form)
    (tp, tr, tf), ref_s, ref_q, meter_ref, form = best

    # score: reference quarters -> generated quarters
    gen = score_notes(res.musicxml)
    to_gen, (xs, ys) = warp(ref_q, gen, fps=8)
    slope = np.diff(ys[[len(ys) // 10, -len(ys) // 10]]) / np.diff(xs[[len(xs) // 10, -len(xs) // 10]])
    scale = snap_scale(float(slope[0]))
    pairs = match(ref_q, gen, to_gen, tol=0.25 * scale)
    sp, sr, sf = prf(len(pairs), len(ref_q), len(gen))
    hands = float(np.mean([ref_q[i].staff == gen[j].staff for i, j in pairs])) if pairs else 0.0
    durations, short = duration_scores(ref_q, gen, pairs, scale)

    # rhythm: consecutive distinct matched onsets, interval ratio == scale
    onsets = sorted({(round(ref_q[i].onset, 4), round(gen[j].onset, 4)) for i, j in pairs})
    good = total = 0
    for (r0, g0), (r1, g1) in zip(onsets, onsets[1:]):
        if r1 - r0 <= 1e-6:
            continue
        total += 1
        good += abs((g1 - g0) - scale * (r1 - r0)) < 1e-3
    rhythm = good / total if total else 0.0

    # barlines: when one score's bars are twice as long as the other's, only
    # every other barline can agree, so divide by the sparser side.
    hit = sum(ref_q[i].downbeat and gen[j].downbeat for i, j in pairs)
    n_ref_db = sum(ref_q[i].downbeat for i, _ in pairs)
    n_gen_db = sum(gen[j].downbeat for _, j in pairs)
    dbf = hit / min(n_ref_db, n_gen_db) if min(n_ref_db, n_gen_db) else 0.0

    return Report(
        id=piece["id"],
        key=res.key,
        key_ref=piece["key"],
        key_ok=key_verdict(res.key, piece["key"]),
        meter=res.time_sig,
        meter_ref=meter_ref,
        meter_ok=meter_verdict(res.time_sig, meter_ref),
        bpm=round(res.bpm, 1),
        trans_p=tp,
        trans_r=tr,
        trans_f1=tf,
        score_p=sp,
        score_r=sr,
        score_f1=sf,
        beat_scale=scale,
        rhythm_acc=rhythm,
        barlines=dbf,
        hands=hands,
        durations=durations,
        short=short,
        notes_ref=len(ref_q),
        notes_score=len(gen),
        dropped=res.dropped_notes,
        form=form,
    )


def main() -> None:
    pieces = json.loads((ROOT / "pieces.json").read_text())
    want = [a for a in sys.argv[1:] if not a.startswith("--")]
    if want:
        pieces = [p for p in pieces if any(w in p["id"] for w in want)]
    reports = []
    hdr = f"{'piece':30} {'key':18} {'meter':14} {'bpm':>5} {'transF1':>7} {'scoreF1':>7} {'scale':>5} {'rhythm':>6} {'bars':>5} {'hands':>5} {'durs':>5} form"
    print(hdr)
    print("-" * len(hdr))
    for p in pieces:
        try:
            r = evaluate(p)
        except Exception as e:  # keep going; one bad download shouldn't stop the run
            print(f"{p['id']:30} ERROR {type(e).__name__}: {e}")
            continue
        reports.append(r)
        key = f"{r.key}{'' if r.key_ok == 'ok' else ' (' + r.key_ok + ')'}"
        meter = f"{r.meter}{'' if r.meter_ok == 'ok' else ' vs ' + r.meter_ref}"
        print(
            f"{r.id:30} {key:18} {meter:14} {r.bpm:5.0f} {r.trans_f1:7.3f} {r.score_f1:7.3f} "
            f"{r.beat_scale:5.2g} {r.rhythm_acc:6.3f} {r.barlines:5.2f} {r.hands:5.2f} {r.durations:5.2f} {r.form}",
            flush=True,
        )
    if reports:
        mean = lambda f: np.mean([getattr(r, f) for r in reports])  # noqa: E731
        print("-" * len(hdr))
        print(
            f"{'mean':30} {sum(r.key_ok == 'ok' for r in reports)}/{len(reports)} keys{'':7} "
            f"{sum(r.meter_ok == 'ok' for r in reports)}/{len(reports)} meters {'':5} "
            f"{mean('trans_f1'):7.3f} {mean('score_f1'):7.3f} {'':5} {mean('rhythm_acc'):6.3f} {mean('barlines'):5.2f} {mean('hands'):5.2f} {mean('durations'):5.2f}  (median length ratio {np.median([r.short for r in reports]):.2f})"
        )
        OUT.mkdir(exist_ok=True)
        (OUT / ("results_lookup.json" if "--lookup" in sys.argv else "results.json")).write_text(json.dumps([asdict(r) for r in reports], indent=1))


if __name__ == "__main__":
    main()
