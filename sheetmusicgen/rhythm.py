"""Beat tracking, meter/downbeat estimation and quantization of note timings."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Sequence

import numpy as np

from .transcribe import SAMPLE_RATE, Note, Pedal

# Grid choice (subdivisions of a quarter) -> subdivisions of a dotted quarter in compound meter.
COMPOUND_GRID = {1: 1, 2: 3, 3: 3, 4: 6, 6: 6, 8: 12}
CHORD_SPREAD = 0.035  # seconds; onsets closer than this are played "together"
PEDAL_RELEASE_SLACK = 0.12  # seconds; a note ending this close to a pedal release was held by it
PHASE_SWITCH_COST = 5.0  # accent evidence needed to move the downbeat by a tracked beat
DETACHED_SILENCE = 4.0  # seconds of silence that can separate the piece from unrelated material
DETACHED_MAX_SHARE = 0.1  # ...if that material has at most this share of the notes
TRIPLET_BIAS = 0.02  # beats of timing error a triplet beat must save over a duple one
GRID_SWITCH_COST = 0.05  # timing error needed to switch a hand between duple and triplet beats
SHORTEST_NOTE = 0.045  # seconds; 32nds are only allowed where they last at least this long
CUT_TIME_FASTEST = 320  # bpm; quarters known to be quarters are notated in 2/2 up to this tempo
THREES_WINDOW = 24  # pulses; bars of compound and of 4/4 both fit whole (4 x 6 = 3 x 8)
OFFBEAT_BAR_MARGIN = 1.05  # accent ratio needed to start bars halfway between tracked beats
OPENING_CHORD_MARGIN = 1.12  # accent ratio another beat needs to keep an opening chord off the downbeat
BEAT_FASTEST, BEAT_SLOWEST = 240, 30  # bpm range a tracked beat may take
BEAT_THRESHOLD = 0.2  # beat probability below which a beat costs rather than earns
TEMPO_CHANGE_COST = 30.0  # per squared log ratio between consecutive beat intervals
DIVISION_MARGIN = 0.75  # relative grid error a triple division needs to beat the duple one
SNAP_REACH, SNAP_REACH_SHARE = 0.06, 0.2  # seconds / share of a beat a tracked beat may move onto a note
SIXTHS_SHARE = 0.15  # share of off-beat onsets at 1/6 or 5/6 of a beat that shows 16ths in compound meter
DOWNBEAT_MIN_CONTRAST = 0.1  # how much likelier the downbeat phase must be than the others
TWELVE_EIGHT_HANDICAP = 1.2  # contrast ratio 12/8 needs over 6/8
BAR_RESET_COST = 2.0  # log-probability a bar of another length must gain to restart the bar count
HALF_BEATS_BELOW = 80  # bpm; audio beats slower than this may be half notes (see beats_are_halves)
RUNNER_UP_SHARE = 0.6  # share of the best meter's contrast another needs to be suggested too
TEMPO_PRIOR_COST = 3.0  # per beat, per squared log ratio between its interval and the typical one


@dataclass
class QNote:
    start: Fraction  # in quarter notes from the start of measure 1
    end: Fraction
    pitch: int
    velocity: int
    pedaled: bool = False  # sounded until a pedal release, so `end` is not where the key was let go
    right: bool | None = None  # played by the right hand (None: decided by a split point)


@dataclass
class Rhythm:
    notes: list[QNote]
    bpm: float  # notated beats (quarters; dotted quarters in compound meter, halves in cut time) per minute
    time_sig: str  # e.g. "4/4", "6/8"
    beat_times: list[float] = field(default_factory=list)  # seconds of each notated beat
    downbeat_times: list[float] = field(default_factory=list)  # seconds where each bar starts
    # Where each bar starts (quarter notes); bars are `bar_length` long except where
    # these say otherwise. Empty: every bar is `bar_length` long.
    bar_starts: list[Fraction] = field(default_factory=list)
    other_meters: list[str] = field(default_factory=list)  # time signatures that fit nearly as well

    @property
    def compound(self) -> bool:
        return self.time_sig.endswith("/8")

    @property
    def cut(self) -> bool:
        return self.time_sig.endswith("/2")

    @property
    def bar_length(self) -> Fraction:
        num, den = map(int, self.time_sig.split("/"))
        return Fraction(4 * num, den)

    def bars(self, end: Fraction) -> list[tuple[Fraction, Fraction]]:
        """(start, end) of every bar, up to the one holding `end`."""
        starts = list(self.bar_starts) or [Fraction(0)]
        while starts[-1] + self.bar_length < end:
            starts.append(starts[-1] + self.bar_length)
        return list(zip(starts, starts[1:] + [starts[-1] + self.bar_length]))

    def meter_for(self, length: Fraction) -> str:
        """Time signature of a bar `length` quarter notes long, in this piece's beat unit."""
        if length == self.bar_length:
            return self.time_sig
        if self.compound or length.denominator != 1:
            return f"{int(length * 2)}/8"
        if self.cut and length % 2 == 0:
            return f"{int(length) // 2}/2"
        return f"{int(length)}/4"


def drop_detached(notes: list[Note]) -> list[Note]:
    """Drop a short stretch at the start or end that a long silence separates from the piece.

    Recordings (especially videos) often have an intro jingle, an outro or a
    few stray notes; they would skew the key, the tempo and the final bars.
    """
    notes = sorted(notes, key=lambda n: n.onset)
    segments: list[list[Note]] = []
    end = -np.inf
    for n in notes:
        if n.onset - end >= DETACHED_SILENCE:
            segments.append([])
        segments[-1].append(n)
        end = max(end, n.offset)

    def minor(seg: list[Note]) -> bool:
        return len(seg) <= DETACHED_MAX_SHARE * len(notes)

    while len(segments) > 1 and minor(segments[-1]):
        segments.pop()
    while len(segments) > 1 and minor(segments[0]):
        segments.pop(0)
    return [n for seg in segments for n in seg]


def extend_beats(beats: np.ndarray, last: float) -> np.ndarray:
    """Extrapolate `beats` at the median tempo so every time in [0, last] maps inside the grid."""
    period = float(np.median(np.diff(beats)))
    head = np.arange(beats[0] - period, -period, -period)[::-1]
    tail = np.arange(beats[-1] + period, last + period, period)
    return np.concatenate([head, beats, tail])


def _onset_envelope(notes: list[Note], last: float, fps: int) -> np.ndarray:
    """Note onsets weighted by velocity, `fps` frames per second, smoothed over 50 ms."""
    env = np.zeros(int(last * fps) + 1)
    for n in notes:
        env[int(n.onset * fps)] += n.velocity / 127
    return np.convolve(env, np.hanning(max(5, fps // 20)), mode="same")


def track_pulse(notes: list[Note], period: float, last: float) -> np.ndarray:
    """Beats at a known `period` (seconds), placed where the notes are.

    For a pulse found to be a fraction of the tracked beat: dividing tracked
    beats evenly carries over their errors, which are large where
    syncopation pulls the tracker off the notes. The envelope is fine (1 ms)
    so the fixed tempo isn't rounded to whole frames.
    """
    import librosa

    fps, hop = 1000, 512
    env = _onset_envelope(notes, last, fps)
    _, beats = librosa.beat.beat_track(onset_envelope=env, sr=fps * hop, hop_length=hop, units="time", bpm=60.0 / period)
    if len(beats) < 4:
        return np.arange(notes[0].onset if notes else 0.0, last + period, period)
    return extend_beats(beats, last)


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
    env = _onset_envelope(notes, last, 100)
    _, beats = librosa.beat.beat_track(onset_envelope=env, sr=SAMPLE_RATE, hop_length=hop, units="time")
    if len(beats) < 4:
        # Too little rhythmic content to track; fall back to a steady 100 bpm.
        return track_beats(duration, notes, 100.0)
    return extend_beats(beats, last)


def _beat_path(gain: np.ndarray, periods: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Frames of the best beat sequence: each beat earns `gain` at its frame plus `prior` for
    its interval (`periods`, in frames); changing the interval costs TEMPO_CHANGE_COST times
    its squared log ratio."""
    n = len(gain)
    logp = np.log(periods)
    jump = TEMPO_CHANGE_COST * (logp[:, None] - logp[None, :]) ** 2  # [new, old]
    score = np.full((n, len(periods)), -np.inf)
    back = np.full((n, len(periods)), -1, dtype=np.int32)  # previous period index; -1 = first beat
    rows = np.arange(len(periods))
    for t in range(n):
        score[t] = gain[t]  # a first beat may start anywhere
        prev = t - periods
        ok = prev >= 0
        if ok.any():
            cand = score[prev[ok]] - jump[ok]  # [new period, old period]
            best = cand.argmax(axis=1)
            val = cand[np.arange(ok.sum()), best] + gain[t] + prior[ok]
            better = val > score[t, ok]
            score[t, rows[ok][better]] = val[better]
            back[t, rows[ok][better]] = best[better]

    t, k = np.unravel_index(int(np.argmax(score)), score.shape)
    beats = []
    while True:
        beats.append(t)
        prev_k = back[t, k]
        if prev_k < 0:
            break
        t, k = t - periods[k], prev_k
    return np.array(beats[::-1])


def decode_beats(beat_act: np.ndarray, downbeat_act: np.ndarray, fps: float) -> tuple[list[float], list[float]]:
    """Beats (seconds) and how likely each starts a bar, from framewise beat/downbeat probabilities.

    Picking the activation peaks loses beats wherever the music goes quiet
    (a held chord, a soft passage), and every lost beat doubles a note value.
    Instead a dynamic program chooses the beat sequence: each beat earns its
    activation less a threshold, so beats can't be added or dropped for free,
    and changing the interval between beats costs the squared log of the
    change. Small steps still add up to a new metrical level (a run of
    eighths drawing the beat onto them), so a second pass also charges each
    beat for straying from the first pass's median tempo.
    """
    n = len(beat_act)
    periods = np.arange(int(fps * 60 / BEAT_FASTEST), int(fps * 60 / BEAT_SLOWEST) + 1)
    if n < 2 * periods[-1]:
        return [], []
    gain = beat_act - BEAT_THRESHOLD
    beats = _beat_path(gain, periods, np.zeros(len(periods)))
    if len(beats) > 4:
        # The interval covering most of the time: a stretch at double tempo has twice the beats.
        ioi = np.sort(np.diff(beats))
        typical = ioi[np.searchsorted(np.cumsum(ioi), ioi.sum() / 2)]
        beats = _beat_path(gain, periods, -TEMPO_PRIOR_COST * np.log(periods / typical) ** 2)

    downbeat = [float(downbeat_act[max(0, b - 2) : b + 3].max()) for b in beats]
    return (beats / fps).tolist(), downbeat


def snap_beats(beats: np.ndarray, notes: list[Note]) -> np.ndarray:
    """Move beats tracked on the audio onto the notes played with them.

    The beat model works in 20 ms frames and lags the attacks slightly;
    that jitter blurs where notes fall within the beat (a triplet eighth is
    only a sixth of a beat away from a 16th). A beat with a chord or note
    close by moves onto it; one without (a rest, a held chord) moves by the
    median shift of the others.
    """
    clusters = np.array(_onset_clusters(notes))
    if len(clusters) < 2 or len(beats) < 2:
        return beats
    reach = np.minimum(SNAP_REACH, SNAP_REACH_SHARE * np.median(np.diff(beats)))
    j = np.clip(np.searchsorted(clusters, beats), 1, len(clusters) - 1)
    nearest = np.where(np.abs(clusters[j] - beats) < np.abs(clusters[j - 1] - beats), clusters[j], clusters[j - 1])
    shift = nearest - beats
    close = np.abs(shift) <= reach
    if not close.any():
        return beats
    out = np.where(close, nearest, beats + np.median(shift[close]))
    return np.maximum.accumulate(out)  # keep them in order where two beats took one onset


def rescale_beats(
    beats: np.ndarray, factor: int, downbeat_probs: Sequence[tuple[float, float]] = ()
) -> tuple[np.ndarray, int | None]:
    """Beats `factor` (2 or 1/2 as -2) times as fast, and the beats per bar on the old beats if known.

    Doubling halves every beat, so every note value doubles; halving (-2)
    keeps every other beat, on the side the downbeats favour.
    """
    found = _downbeat_meter(beats, downbeat_probs, False, None) if len(downbeat_probs) else None
    m = found[0] if found else None
    if factor == 2:
        return np.sort(np.concatenate([beats, (beats[:-1] + beats[1:]) / 2])), m
    phase = 0
    if len(downbeat_probs):
        times, probs = np.array(downbeat_probs, dtype=float).reshape(-1, 2).T
        pos = np.round(np.interp(times, beats, np.arange(len(beats)))).astype(int)
        phase = int(np.bincount(pos % 2, weights=probs, minlength=2).argmax())
    return beats[phase::2], m


def beats_are_halves(notes: list[Note], beats: np.ndarray, duration: float) -> bool:
    """Whether slow beats tracked on the audio are half notes, the notes moving in quarters.

    The beat model sometimes hears a piece whose harmony and bass move in
    half notes as a slow 2/2. Written as quarters, everything would come
    out in values twice too short; a musician would write cut time. Taken
    to be so only when tracking the notes' own onsets finds a beat at twice
    the speed, too.
    """
    period = float(np.median(np.diff(beats)))
    if 60.0 / period >= HALF_BEATS_BELOW:
        return False
    onset_period = float(np.median(np.diff(track_beats(duration, notes, None))))
    return abs(np.log2(period / onset_period) - 1) < 0.1


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
    frac = _offbeat_fractions(notes, beats)
    if frac is None:
        return 2
    e3, e4 = _grid_fit(frac, 3), _grid_fit(frac, 4)
    if e4 > 0.05 and e3 < 0.75 * e4:
        return 3
    return 3 if beats_in_sixths(notes, beats) else 2


def _offbeat_fractions(notes: list[Note], beats: np.ndarray) -> np.ndarray | None:
    """Where off-beat onsets fall within their beat (0-1), or None when too few are off the beat."""
    pos = np.interp(_onset_clusters(notes), beats, np.arange(len(beats)))
    frac = pos % 1
    off = np.minimum(frac, 1 - frac) > 0.1
    return frac[off] if off.sum() >= max(8, 0.1 * len(frac)) else None


def _grid_fit(frac: np.ndarray, d: int) -> float:
    x = frac * d
    return float(np.mean(np.abs(x - np.round(x))))


def beats_in_sixths(notes: list[Note], beats: np.ndarray) -> bool:
    """Whether `beats` split into six (16ths of a compound beat, sextuplets).

    These fit a grid of 3 no better than one of 4. Telling them by a grid of
    6 needs onsets at a sixth or five sixths of the beat: two against three
    in a simple meter fits that grid too, but never plays there.
    """
    frac = _offbeat_fractions(notes, beats)
    if frac is None:
        return False
    sixths = np.mean(np.minimum(np.abs(frac - 1 / 6), np.abs(frac - 5 / 6)) < 1 / 24)
    return bool(sixths >= SIXTHS_SHARE and _grid_fit(frac, 6) < DIVISION_MARGIN * _grid_fit(frac, 8))


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
    m, phase, _ = _rank_meters(beat_pos, weights, candidates)
    return m, phase


def _rank_meters(
    beat_pos: list[int], weights: list[float], candidates: dict[int, float]
) -> tuple[int, int, list[int]]:
    """`_choose_meter`, plus the other beats-per-bar that came close (RUNNER_UP_SHARE)."""
    if not beat_pos:
        return next(iter(candidates)), 0, []
    scores = {}
    for m, handicap in candidates.items():
        s = np.zeros(m)
        for b, w in zip(beat_pos, weights):
            s[b % m] += w
        # How concentrated the accents are on one phase, relative to uniform (1 = none).
        scores[m] = (s.max() / (s.mean() + 1e-9) / handicap - 1, int(s.argmax()))
    best = max(scores, key=lambda m: scores[m][0])
    close = [m for m in scores if m != best and scores[m][0] >= RUNNER_UP_SHARE * scores[best][0]]
    return best, scores[best][1], close


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


def _accent_curve(notes: list[Note], beats: np.ndarray, division: int) -> np.ndarray:
    """Accent (see `_beat_accents`) on every one of `beats`, 0 where no note starts."""
    acc = np.zeros(len(beats))
    for b, w in zip(*_beat_accents(_to_ticks(notes, beats, division), division)):
        if 0 <= b < len(acc):
            acc[b] += w
    return acc


def _threes_hold(notes: list[Note], pulses: np.ndarray) -> bool:
    """Whether accents on `pulses` (thirds of a tracked beat) recur in a compound meter rather than in 4/4.

    The tracked beat splits in three both in compound meter and in a simple
    meter whose syncopation (3 + 3 + 2 eighths per bar) caught the tracker.
    They differ in the bar: accents recur every six pulses in a compound
    meter (a bar of 6/8, two of 3/8, or a 3/8 bar in sixteenths), every eight
    in 4/4. Measured within windows of 24 pulses, where both fit whole, so
    tempo drift doesn't blur them.
    """
    acc = _accent_curve(notes, pulses, 2)
    contrast = {6: [], 8: []}
    weights = []
    for start in range(0, len(acc) - THREES_WINDOW + 1, THREES_WINDOW // 2):
        a = acc[start : start + THREES_WINDOW]
        if not a.any():
            continue
        weights.append(a.sum())
        for k in contrast:
            phases = np.bincount(np.arange(THREES_WINDOW) % k, weights=a, minlength=k)
            contrast[k].append(phases.max() / phases.mean())
    if not weights:
        return True
    return np.average(contrast[6], weights=weights) >= np.average(contrast[8], weights=weights)


def _group_beats(
    notes: list[Note], beats: np.ndarray, k: int, division: int, last: float, switch_cost: float = PHASE_SWITCH_COST
) -> np.ndarray:
    """Slow the tempo `k` times, keeping the most accented of every `k` beats.

    Beat trackers occasionally slip by a beat, which moves which beat of the
    group is accented, so the choice is a Viterbi pass that may change phase
    (at `switch_cost`) where the accents clearly move. Every change leaves a
    beat `k` times too short, so steady beats should not allow it.
    """
    acc = _accent_curve(notes, beats, division)
    score = np.zeros(k)  # best total accent on kept beats, per phase
    back = np.zeros((len(beats), k), dtype=int)
    for i, a in enumerate(acc):
        best = int(score.argmax())
        new = np.empty(k)
        for p in range(k):
            stay, switch = score[p], score[best] - switch_cost
            back[i, p] = p if stay >= switch else best
            new[p] = max(stay, switch) + (a if i % k == p else 0.0)
        score = new

    p = int(score.argmax())
    keep = []
    for i in range(len(beats) - 1, -1, -1):
        if i % k == p:
            keep.append(i)
        p = back[i, p]
    return extend_beats(beats[keep[::-1]], last)


def normalize_beats(
    notes: list[Note], beats: np.ndarray, division: int, compound: bool | None = None
) -> tuple[np.ndarray, bool, bool]:
    """Move the tracked pulse to the level a musician would notate as the beat.

    Beat trackers often lock onto half or double that, onto the eighths of a
    compound meter, or onto the dotted quarters of a syncopated simple meter.
    Returns the beats, whether the meter is compound (`compound` forces it)
    and whether it is cut time (2/2, the beats being quarters). Duple beats
    become quarters at 50-160 bpm, triple beats dotted quarters at <= 100 bpm
    (a faster triple pulse is a dotted eighth, so pairs of them are merged).
    Where syncopation showed the pulse to be eighths, the quarters they pair
    into stay quarters up to CUT_TIME_FASTEST, in cut time when above 160.
    """
    last = max(beats[-1], max(n.offset for n in notes))
    period = float(np.median(np.diff(beats)))
    eighths = False  # the beats are known to be eighths, tracked at a steady third of the tracked beat
    if division == 3 and not compound:
        # A beat that splits in three is the dotted beat of a compound meter,
        # unless the meter is simple and syncopation (3 + 3 + 2) caught the
        # tracker: then its thirds are eighths, to be grouped in twos.
        # (Forced into a simple meter, a compound piece's thirds may be 16ths.)
        pulses = track_pulse(notes, period / 3, last)
        threes = _threes_hold(notes, pulses)
        if compound is False or not threes:
            beats, compound, eighths = pulses, False, not threes
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

    if eighths:
        # Pair the eighths into quarters, keeping to one phase: tracked at a
        # fixed tempo, they don't slip the way freely tracked beats do.
        beats = _group_beats(notes, beats, 2, division, last, switch_cost=np.inf)
        period *= 2
    fastest = 100 if division == 3 else CUT_TIME_FASTEST if eighths else 160
    while 60.0 / period > fastest:
        beats = _group_beats(notes, beats, 2, division, last, np.inf if eighths else PHASE_SWITCH_COST)
        period *= 2
    while division == 2 and 60.0 / period < 50:
        mids = (beats[:-1] + beats[1:]) / 2
        beats = np.sort(np.concatenate([beats, mids]))
        period /= 2
    return beats, division == 3, eighths and 60.0 / period > 160


def _offbeat_bars(notes: list[Note], beats: np.ndarray, candidates: dict[int, float]) -> bool:
    """Whether bars start halfway between `beats`, i.e. the tracker locked onto the off-beats.

    In steady eighths (or 16ths, tracked as eighths) on-beats and off-beats
    look alike, and an accompaniment on the off-beats can draw the tracker
    there. The bar shows which is which: the downbeat's accents (a new bass
    note, long notes) recur once per bar. So pick the meter and downbeat on a
    grid of half beats, and see whether the downbeat lands between beats.
    Notes off that grid (triplets) are left out rather than rounded onto it.
    """
    half = np.interp([n.onset for n in notes], beats, np.arange(len(beats))) * 2
    on_grid = [n for n, h in zip(notes, half) if abs(h - round(h)) < 0.2]
    pos, weights = _beat_accents(_to_ticks(on_grid, beats, 2), 1)
    if not pos:
        return False
    best = {False: 0.0, True: 0.0}
    for m, handicap in candidates.items():
        s = np.zeros(2 * m)
        for b, w in zip(pos, weights):
            s[b % (2 * m)] += w
        contrast = s / (s.mean() + 1e-9) / handicap
        best[False] = max(best[False], contrast[0::2].max())
        best[True] = max(best[True], contrast[1::2].max())
    return best[True] > OFFBEAT_BAR_MARGIN * best[False]


def _opening_downbeat(raw: list[tuple[int, int, Note]], res: int, beat_pos: list[int], weights: list[float], m: int, phase: int) -> int:
    """Move the downbeat to an opening chord when the accents barely prefer another beat.

    Where every beat is played alike the accents can't tell the bar apart,
    but a piece starts either on the downbeat or with an upbeat, and upbeats
    are mostly single melody notes rather than chords.
    """
    first = min(s for s, _, _ in raw)
    if first % res or sum(s == first for s, _, _ in raw) < 2:
        return phase
    acc = np.zeros(m)
    for b, w in zip(beat_pos, weights):
        acc[b % m] += w
    opening = (first // res) % m
    return opening if acc[opening] * OPENING_CHORD_MARGIN >= acc[phase] else phase


def _downbeat_meter(
    beats: np.ndarray, downbeat_probs: Sequence[tuple[float, float]], compound: bool, meter: int | None
) -> tuple[int, int, list[int]] | None:
    """Beats per bar and the downbeat phase from how likely each tracked beat is to start a bar.

    For each bar length, the phase whose beats are most likely downbeats
    should stand out from the others; a half-bar accent in 4/4 makes the
    second-best phase of a 4-beat bar fairly likely too, but no 2-beat bar
    explains both. `downbeat_probs` holds (time, probability) per tracked
    beat; `meter` forces the beats per bar, leaving only the phase to settle.
    Returns the beats per bar, the phase and the other bar lengths that came
    close (RUNNER_UP_SHARE), or None when no bar length stands out.
    """
    times, probs = np.array(downbeat_probs, dtype=float).reshape(-1, 2).T
    pos = np.round(np.interp(times, beats, np.arange(len(beats)))).astype(int)
    if len(pos) < 16:
        return None
    scores = {}
    for m in [meter] if meter else (2, 3, 4):
        phase_mean = np.bincount(pos % m, weights=probs, minlength=m) / np.maximum(np.bincount(pos % m, minlength=m), 1)
        top = int(phase_mean.argmax())
        contrast = phase_mean[top] - np.delete(phase_mean, top).mean()
        if compound and m == 4:
            contrast /= TWELVE_EIGHT_HANDICAP  # 6/8 is far more common than 12/8, and keeps every barline
        scores[m] = (contrast, top)
    m = max(scores, key=lambda k: scores[k][0])
    contrast, phase = scores[m]
    if not meter and contrast < DOWNBEAT_MIN_CONTRAST:
        return None
    close = [k for k in scores if k != m and scores[k][0] >= RUNNER_UP_SHARE * contrast]
    return m, phase, close


def _track_bars(beats: np.ndarray, downbeat_probs: Sequence[tuple[float, float]], m: int) -> list[int]:
    """Beats (indices into `beats`) that start a bar of `m` beats, following the tracked downbeats.

    One missed or extra beat, a bar of another length or a pickup into a
    new section would put every later barline in the wrong place if the
    bars just counted on from the first; so a Viterbi pass counts beats
    within the bar and may restart the count, at BAR_RESET_COST, where the
    downbeat probabilities clearly move. Each restart leaves one bar of
    another length.
    """
    times, probs = np.array(downbeat_probs, dtype=float).reshape(-1, 2).T
    pos = np.round(np.interp(times, beats, np.arange(len(beats)))).astype(int)
    lo = int(pos.min())
    p = np.full(int(pos.max()) - lo + 1, 0.5)  # no evidence where no beat was tracked
    p[pos - lo] = probs
    p = np.clip(p, 0.02, 0.98)
    emit = np.column_stack([np.log(p)] + [np.log(1 - p)] * (m - 1))  # [beat, position in bar]
    score = emit[0].copy()
    back = np.zeros((len(p), m), dtype=int)
    for i in range(1, len(p)):
        counted = np.roll(score, 1)  # position k follows k - 1
        best = int(score.argmax())
        restart = counted < score[best] - BAR_RESET_COST
        back[i] = np.where(restart, best, (np.arange(m) - 1) % m)
        score = np.where(restart, score[best] - BAR_RESET_COST, counted) + emit[i]
    k = int(score.argmax())
    starts = []
    for i in range(len(p) - 1, -1, -1):
        if k == 0:
            starts.append(lo + i)
        k = back[i, k]
    return starts[::-1]


def _held_by_pedal(n: Note, pedals: list[Pedal]) -> bool:
    return any(down < n.offset and abs(n.offset - up) <= PEDAL_RELEASE_SLACK for down, up in pedals)


def _refine_beats(clusters: list[float], beats: np.ndarray, sub: int) -> set[int]:
    """Beats where a grid of `sub` would merge onsets that were played apart (fast runs)."""
    idx = np.arange(len(beats), dtype=float)
    ticks = [int(round(np.interp(t, beats, idx) * sub)) for t in clusters]
    return {a // sub for a, b in zip(ticks, ticks[1:]) if a == b}


def _grid_error(frac: np.ndarray, d: int) -> float:
    """Total distance, in beats, from positions within a beat to a grid of `d` per beat."""
    x = frac * d
    return float(np.sum(np.abs(x - np.round(x)))) / d


def _triplet_beats(pos: np.ndarray, duple: int) -> set[int]:
    """Beats (at positions `pos` of one hand's onsets) played in triplets rather than on a grid of `duple`.

    A Viterbi pass over the beats with onsets: each costs the timing error of
    its onsets on its grid, and switching grids costs extra, so a sloppy beat
    follows its neighbours instead of flipping on its own.
    """
    beat_of = np.floor(pos).astype(int)
    frac = pos - beat_of
    beat_ids = np.unique(beat_of)
    if not len(beat_ids):
        return set()
    costs = []
    for b in beat_ids:
        f = frac[beat_of == b]
        costs.append((_grid_error(f, duple), _grid_error(f, 3) + TRIPLET_BIAS))

    total = np.array(costs[0])
    back = np.zeros((len(beat_ids), 2), dtype=int)
    for i, c in enumerate(costs[1:], 1):
        new = np.empty(2)
        for state in (0, 1):
            stay, switch = total[state], total[1 - state] + GRID_SWITCH_COST
            back[i, state] = state if stay <= switch else 1 - state
            new[state] = min(stay, switch) + c[state]
        total = new

    state = int(total.argmin())
    out = set()
    for i in range(len(beat_ids) - 1, -1, -1):
        if state:
            out.add(int(beat_ids[i]))
        state = back[i, state]
    return out


def quantize(
    notes: list[Note],
    beats: np.ndarray,
    subdivisions: int = 4,
    meter: int | None = None,
    compound: bool = False,
    pedals: list[Pedal] = (),
    split: int = 60,
    cut: bool = False,
    downbeat_probs: Sequence[tuple[float, float]] = (),
    right_hand: Sequence[bool] | None = None,
) -> Rhythm:
    """Snap notes to the beat grid and align bars to downbeats.

    `beats` must already be at the notated beat level (quarters, or dotted
    quarters when `compound`). `subdivisions` is the grid per quarter note
    (4 = sixteenths). The grid is chosen per beat and per hand (`right_hand`
    says which hand plays each of `notes`; without it, `split` is where the
    right hand starts): where it is too coarse for a fast run of
    sixteenths that beat is notated in 32nds, and in simple meter a hand's
    beat can be in triplets instead (two against three between the hands is
    common). `meter` forces the number of beats per bar. `cut` writes 4/4
    as 2/2, the beats staying quarters. Notes ending with a release of
    `pedals` are flagged, since only the pedal, not the key, tells when they
    stopped. `downbeat_probs`, (time, probability of starting a bar) for
    beats tracked on the audio, set the bars instead of the accents when
    they clearly show them.
    """
    from_downbeats = _downbeat_meter(beats, downbeat_probs, compound, meter) if len(downbeat_probs) else None
    if compound:
        candidates = {meter: 1.0} if meter else {1: 1.0, 2: 1.6}  # 3/8 unless 6/8 is clear
    else:
        candidates = {meter: 1.0} if meter else {4: 1.0, 3: 1.15}  # prefer 4/4 unless 3/4 is clearly better
        if not from_downbeats and _offbeat_bars(notes, beats, candidates):
            last = max(beats[-1], max(n.offset for n in notes))
            beats = extend_beats((beats[:-1] + beats[1:]) / 2, last)

    beat_ql = Fraction(3, 2) if compound else Fraction(1)
    sub = COMPOUND_GRID.get(subdivisions, 6) if compound else subdivisions
    period = float(np.median(np.diff(beats)))
    # 16th grid: allow 32nds where needed, unless too fast to play
    fine = beat_ql / sub == Fraction(1, 4) and period / (2 * sub) >= SHORTEST_NOTE
    triplets = not compound and sub in (2, 4, 8)
    res = math.lcm(2 * sub if fine else sub, 3 if triplets else 1)  # ticks per beat, fitting every grid in use

    if right_hand is None:
        right_hand = [n.pitch >= split for n in notes]
    idx = np.arange(len(beats), dtype=float)
    grids: dict[bool, dict[int, int]] = {}  # right hand? -> beat -> grid, where not `sub`
    for hand in (True, False):
        clusters = _onset_clusters([n for n, r in zip(notes, right_hand) if r == hand])
        grid = dict.fromkeys(_refine_beats(clusters, beats, sub), 2 * sub) if fine else {}
        if triplets:
            grid.update(dict.fromkeys(_triplet_beats(np.interp(clusters, beats, idx), sub), 3))
        grids[hand] = grid

    def to_ticks(t: float, grid: dict[int, int]) -> int:
        pos = float(np.interp(t, beats, idx))
        d = grid.get(math.floor(pos), sub)
        return int(round(pos * d)) * (res // d)

    raw = []
    for n, right in zip(notes, right_hand):
        grid = grids[right]
        s = to_ticks(n.onset, grid)
        step = res // grid.get(s // res, sub)
        e = max(s + step, to_ticks(n.offset, grid))
        raw.append((s, e, n))

    if from_downbeats:
        m, phase, close = from_downbeats
        starts = _track_bars(beats, downbeat_probs, m) or [phase]
    else:
        beat_pos, weights = _beat_accents(raw, res)
        m, phase, close = _rank_meters(beat_pos, weights, candidates)
        starts = [_opening_downbeat(raw, res, beat_pos, weights, m, phase)]

    # Bars run on at m beats before and after the tracked ones; the first
    # bar is the one holding the first note, the last the one holding the end.
    first = min((s for s, _, _ in raw), default=0) // res
    last = -(-max((e for _, e, _ in raw), default=1) // res)
    while starts[0] > first:
        starts.insert(0, starts[0] - m)
    while starts[-1] + m < last:
        starts.append(starts[-1] + m)
    starts = [b for b in starts if b < last]
    starts = starts[max(i for i, b in enumerate(starts) if b <= first) :]
    shift = -starts[0] * res

    tick = beat_ql / res
    out: dict[tuple[int, int], QNote] = {}
    for (s, e, n), right in zip(raw, right_hand):
        key = (s + shift, n.pitch)
        q = QNote((s + shift) * tick, (e + shift) * tick, n.pitch, n.velocity, _held_by_pedal(n, pedals), right)
        # Quantization can merge repeated notes; keep the longer one.
        if key not in out or q.end > out[key].end:
            out[key] = q

    # Where the notated beats and bars fall in the recording.
    step = 2 if cut and m % 2 == 0 else 1
    ends = starts[1:] + [starts[-1] + m]
    beat_times = np.interp([b for a, z in zip(starts, ends) for b in range(a, z, step)], idx, beats).tolist()
    downbeat_times = np.interp(starts, idx, beats).tolist()
    bar_starts = [(b - starts[0]) * beat_ql for b in starts]

    def signature(m: int) -> str:
        if compound:
            return f"{3 * m}/8"
        return f"{m // 2}/2" if cut and m % 2 == 0 else f"{m}/4"

    time_sig = signature(m)
    bpm = 60.0 / period / (2 if time_sig.endswith("/2") else 1)
    notes_out = sorted(out.values(), key=lambda q: (q.start, q.pitch))
    others = [signature(k) for k in close if not meter]
    return Rhythm(notes_out, bpm, time_sig, beat_times, downbeat_times, bar_starts, others)
