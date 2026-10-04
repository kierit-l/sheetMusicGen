"""Beat tracking, meter/downbeat estimation and quantization of note timings."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .transcribe import SAMPLE_RATE, Note, Pedal

# Grid choice (subdivisions of a quarter) -> subdivisions of a dotted quarter in compound meter.
COMPOUND_GRID = {1: 1, 2: 3, 3: 3, 4: 6, 6: 6, 8: 12}
CHORD_SPREAD = 0.035  # seconds; onsets closer than this are played "together"
PEDAL_RELEASE_SLACK = 0.12  # seconds; a note ending this close to a pedal release was held by it
PHASE_SWITCH_COST = 5.0  # accent evidence needed to move the downbeat by a tracked beat


@dataclass
class QNote:
    start: Fraction  # in quarter notes from the start of measure 1
    end: Fraction
    pitch: int
    velocity: int
    pedaled: bool = False  # sounded until a pedal release, so `end` is not where the key was let go


@dataclass
class Rhythm:
    notes: list[QNote]
    bpm: float  # notated beats (quarters, or dotted quarters in compound meter) per minute
    time_sig: str  # e.g. "4/4", "6/8"

    @property
    def compound(self) -> bool:
        return self.time_sig.endswith("/8")

    @property
    def bar_length(self) -> Fraction:
        num, den = map(int, self.time_sig.split("/"))
        return Fraction(4 * num, den)


def _extend(beats: np.ndarray, last: float) -> np.ndarray:
    """Extrapolate `beats` at the median tempo so every time in [0, last] maps inside the grid."""
    period = float(np.median(np.diff(beats)))
    head = np.arange(beats[0] - period, -period, -period)[::-1]
    tail = np.arange(beats[-1] + period, last + period, period)
    return np.concatenate([head, beats, tail])


def track_beats(duration: float, notes: list[Note], bpm: float | None) -> np.ndarray:
    """Return beat times (seconds), extended to cover the whole piece of `duration` seconds.

    The tracked pulse may be at any metrical level; `quantize` settles which
    note value it is.
    """
    import librosa

    last = max([duration] + [n.offset for n in notes]) + 1.0

    if bpm:
        period = 60.0 / bpm
        start = notes[0].onset if notes else 0.0
        return np.arange(start, last + period, period)

    # Track beats on an onset envelope built from the transcribed notes: it is
    # far cleaner than the raw audio's, so the beat phase lines up with the notes.
    hop = SAMPLE_RATE // 100
    env = np.zeros(int(last * 100) + 1)
    for n in notes:
        env[int(n.onset * 100)] += n.velocity / 127
    env = np.convolve(env, np.hanning(5), mode="same")
    _, beats = librosa.beat.beat_track(onset_envelope=env, sr=SAMPLE_RATE, hop_length=hop, units="time")
    if len(beats) < 4:
        # Too little rhythmic content to track; fall back to a steady 100 bpm.
        return track_beats(duration, notes, 100.0)
    return _extend(beats, last)


def _onset_clusters(notes: list[Note]) -> list[float]:
    """Onset times with chord notes (played within CHORD_SPREAD) merged into one."""
    times: list[float] = []
    for t in sorted(n.onset for n in notes):
        if not times or t - times[-1] > CHORD_SPREAD:
            times.append(t)
    return times


def beat_division(notes: list[Note], beats: np.ndarray) -> int:
    """3 if the tracked beats split into three equal parts (compound meter), else 2.

    Measures how far off-beat onsets fall from a grid of 3 and of 4 per beat,
    in units of that grid's spacing (randomly placed onsets score 0.25 on
    either). Onsets on the beat fit both grids, so they are left out.
    """
    pos = np.interp(_onset_clusters(notes), beats, np.arange(len(beats)))
    frac = pos % 1
    off = np.minimum(frac, 1 - frac) > 0.1
    if off.sum() < max(8, 0.1 * len(frac)):
        return 2
    frac = frac[off]

    def err(d: int) -> float:
        x = frac * d
        return float(np.mean(np.abs(x - np.round(x))))

    e3, e4 = err(3), err(4)
    return 3 if e4 > 0.05 and e3 < 0.75 * e4 else 2


def _to_ticks(notes: list[Note], beats: np.ndarray, sub: int) -> list[tuple[int, int, Note]]:
    idx = np.arange(len(beats), dtype=float)
    out = []
    for n in notes:
        s = int(round(np.interp(n.onset, beats, idx) * sub))
        e = max(s + 1, int(round(np.interp(n.offset, beats, idx) * sub)))
        out.append((s, e, n))
    return out


def _choose_meter(
    beat_pos: list[int], weights: list[float], candidates: dict[int, float]
) -> tuple[int, int]:
    """Pick beats-per-bar and which beat is the downbeat, from where accents fall.

    `candidates` maps beats-per-bar to a handicap its accent contrast must beat.
    """
    if not beat_pos:
        return next(iter(candidates)), 0

    best = None
    for m, handicap in candidates.items():
        s = np.zeros(m)
        for b, w in zip(beat_pos, weights):
            s[b % m] += w
        # How concentrated the accents are on one phase, relative to uniform.
        contrast = s.max() / (s.mean() + 1e-9) / handicap
        if best is None or contrast > best[0]:
            best = (contrast, m, int(s.argmax()))
    return best[1], best[2]


def _beat_accents(raw: list[tuple[int, int, Note]], subdivisions: int) -> tuple[list[int], list[float]]:
    """Score how 'downbeat-like' each beat with note onsets is.

    Cues: loudness, long notes, and above all a new bass note (the lowest
    onset among nearby beats), since harmony tends to change on the bar.
    """
    by_beat: dict[int, list[tuple[int, Note]]] = {}
    for s, e, n in raw:
        if s % subdivisions == 0:
            by_beat.setdefault(s // subdivisions, []).append((e - s, n))
    lowest = {b: min(n.pitch for _, n in group) for b, group in by_beat.items()}

    beat_pos, weights = [], []
    for b, group in by_beat.items():
        w = sum(n.velocity / 127 * min(length / subdivisions, 4) ** 0.5 for length, n in group)
        neighbours = [lowest[x] for x in range(b - 2, b + 3) if x != b and x in lowest]
        if all(lowest[b] < p for p in neighbours):
            w += 2.0
        beat_pos.append(b)
        weights.append(w)
    return beat_pos, weights


def _group_beats(notes: list[Note], beats: np.ndarray, k: int, division: int, last: float) -> np.ndarray:
    """Slow the tempo `k` times, keeping the most accented of every `k` beats.

    Beat trackers occasionally slip by a beat, which moves which beat of the
    group is accented, so the choice is a Viterbi pass that may change phase
    (at a cost) where the accents clearly move.
    """
    acc = np.zeros(len(beats))
    for b, w in zip(*_beat_accents(_to_ticks(notes, beats, division), division)):
        if 0 <= b < len(acc):
            acc[b] += w

    score = np.zeros(k)  # best total accent on kept beats, per phase
    back = np.zeros((len(beats), k), dtype=int)
    for i, a in enumerate(acc):
        best = int(score.argmax())
        new = np.empty(k)
        for p in range(k):
            stay, switch = score[p], score[best] - PHASE_SWITCH_COST
            back[i, p] = p if stay >= switch else best
            new[p] = max(stay, switch) + (a if i % k == p else 0.0)
        score = new

    p = int(score.argmax())
    keep = []
    for i in range(len(beats) - 1, -1, -1):
        if i % k == p:
            keep.append(i)
        p = back[i, p]
    return _extend(beats[keep[::-1]], last)


def normalize_beats(
    notes: list[Note], beats: np.ndarray, division: int, compound: bool | None = None
) -> tuple[np.ndarray, bool]:
    """Move the tracked pulse to the level a musician would notate as the beat.

    Beat trackers often lock onto half or double that, or onto the eighths of
    a compound meter. Returns the beats and whether the meter is compound
    (`compound` forces it). Duple beats become quarters at 50-160 bpm, triple
    beats dotted quarters at <= 100 bpm (a faster triple pulse is a dotted
    eighth, so pairs of them are merged).
    """
    last = max(beats[-1], max(n.offset for n in notes))
    period = float(np.median(np.diff(beats)))
    if compound is False:
        division = 2
    elif division == 2 and (compound or 60.0 / period > 160):
        # Too fast for quarters: group the pulses by two, or by three if they
        # are the eighths of a compound meter.
        candidates = {3: 1.0} if compound else {2: 1.0, 3: 1.15}
        group, _ = _choose_meter(*_beat_accents(_to_ticks(notes, beats, 2), 2), candidates)
        if group == 3:
            beats, division = _group_beats(notes, beats, 3, 2, last), 3
            period *= 3

    fastest = 100 if division == 3 else 160
    while 60.0 / period > fastest:
        beats = _group_beats(notes, beats, 2, division, last)
        period *= 2
    while division == 2 and 60.0 / period < 50:
        mids = (beats[:-1] + beats[1:]) / 2
        beats = np.sort(np.concatenate([beats, mids]))
        period /= 2
    return beats, division == 3


def _held_by_pedal(n: Note, pedals: list[Pedal]) -> bool:
    return any(down < n.offset and abs(n.offset - up) <= PEDAL_RELEASE_SLACK for down, up in pedals)


def _refine_beats(clusters: list[float], beats: np.ndarray, sub: int) -> set[int]:
    """Beats where a grid of `sub` would merge onsets that were played apart (fast runs)."""
    idx = np.arange(len(beats), dtype=float)
    ticks = [int(round(np.interp(t, beats, idx) * sub)) for t in clusters]
    return {a // sub for a, b in zip(ticks, ticks[1:]) if a == b}


def quantize(
    notes: list[Note],
    beats: np.ndarray,
    subdivisions: int = 4,
    meter: int | None = None,
    compound: bool = False,
    pedals: list[Pedal] = (),
) -> Rhythm:
    """Snap notes to the beat grid and align bars to downbeats.

    `beats` must already be at the notated beat level (quarters, or dotted
    quarters when `compound`). `subdivisions` is the grid per quarter note
    (4 = sixteenths); where it is too coarse for a fast run of sixteenths, that
    beat is notated in 32nds instead. `meter` forces the number of beats per bar.
    Notes ending with a release of `pedals` are flagged, since only the
    pedal, not the key, tells when they stopped.
    """
    beat_ql = Fraction(3, 2) if compound else Fraction(1)
    sub = COMPOUND_GRID.get(subdivisions, 6) if compound else subdivisions
    fine = beat_ql / sub == Fraction(1, 4)  # 16th grid: allow 32nds where needed
    res = 2 * sub if fine else sub  # ticks per beat on the finest grid in use
    refined = _refine_beats(_onset_clusters(notes), beats, sub) if fine else set()

    idx = np.arange(len(beats), dtype=float)

    def to_ticks(t: float) -> int:
        pos = np.interp(t, beats, idx)
        coarse = int(round(pos * sub))
        return int(round(pos * res)) if coarse // sub in refined else coarse * (res // sub)

    raw = []
    for n in notes:
        s = to_ticks(n.onset)
        step = 1 if s // res in refined else res // sub
        e = max(s + step, to_ticks(n.offset))
        raw.append((s, e, n))

    if compound:
        candidates = {meter: 1.0} if meter else {1: 1.0, 2: 1.6}  # 3/8 unless 6/8 is clear
    else:
        candidates = {meter: 1.0} if meter else {4: 1.0, 3: 1.15}  # prefer 4/4 unless 3/4 is clearly better
    m, phase = _choose_meter(*_beat_accents(raw, res), candidates)

    # Shift so the chosen downbeat phase starts a bar, then drop leading empty bars.
    shift = ((m - phase) % m) * res
    bar = m * res
    first = min((s for s, _, _ in raw), default=0) + shift
    shift -= (first // bar) * bar

    tick = beat_ql / res
    out: dict[tuple[int, int], QNote] = {}
    for s, e, n in raw:
        key = (s + shift, n.pitch)
        q = QNote((s + shift) * tick, (e + shift) * tick, n.pitch, n.velocity, _held_by_pedal(n, pedals))
        # Quantization can merge repeated notes; keep the longer one.
        if key not in out or q.end > out[key].end:
            out[key] = q

    period = float(np.median(np.diff(beats)))
    time_sig = f"{3 * m}/8" if compound else f"{m}/4"
    return Rhythm(sorted(out.values(), key=lambda q: (q.start, q.pitch)), 60.0 / period, time_sig)
