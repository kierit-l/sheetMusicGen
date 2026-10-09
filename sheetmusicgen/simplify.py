"""An easier arrangement: the melody over block chords.

Transcribing a piano cover faithfully writes down everything the pianist
played: octave doublings, inner voices, broken-chord accompaniments,
ornaments, 16th-note fills. That is far more than someone reading the piece
needs. `arrange` reduces it to what an easy-piano edition prints:

- the right hand plays the melody, one note at a time, on an eighth-note
  grid (16ths in slow pieces). The melody is the top voice, except for the
  accompaniment under it (see `melody`);
- the left hand plays held block chords, one per bar or per half bar where
  the harmony changes: the bass note and up to two chord tones close above
  it, taken from everything that isn't melody.
"""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction

from .rhythm import QNote

MELODY_FASTEST = 0.2  # seconds; melody onsets round to 16ths where these last this long, else to eighths
MELODY_LEAP, MELODY_REACH = 7, 2  # semitones / beats; see melody()
MELODY_SPIKE = 12  # semitones above the neighbouring melody notes that make a note a stray one
SEGMENT_COST = 0.6  # misfit (beats x velocity of notes off the chord) a chord change must remove
CHORD_TONES = 2  # pitch classes written above the bass
CHORD_TONE_SHARE = 0.3  # of the strongest pitch class's weight, needed to be written
BASS_RANGE = (36, 52)  # C2-E3: the bass note is moved by octaves into this range
CHORD_FLOOR = 47  # chord tones go above B2 at least, where close chords aren't muddy


def _top_voice(notes: list[QNote], beat: Fraction) -> list[QNote]:
    by_start: dict[Fraction, list[QNote]] = defaultdict(list)
    for q in notes:
        by_start[q.start].append(q)
    out: list[QNote] = []
    for start in sorted(by_start):
        top = max(by_start[start], key=lambda q: q.pitch)
        if out:
            last = out[-1]
            if last.end > start and not last.pedaled and top.pitch < last.pitch:
                continue
            if start - last.start <= MELODY_REACH * beat and top.pitch < last.pitch - MELODY_LEAP:
                continue
        out.append(top)
    return out


def melody(notes: list[QNote], beat: Fraction) -> list[QNote]:
    """The top voice of `notes` (the right hand's), skipping the accompaniment under it.

    Lower notes are accompaniment while a melody note is still held down, and for
    MELODY_REACH beats after one starts if they lie more than MELODY_LEAP
    below it: a right hand playing the tune over broken chords fills the
    gaps between melody notes with chord tones further down. Notes held by
    the pedal don't count as held, since everything under it rings on.
    A lone note far above the tune around it (a misheard overtone) is left
    out, and the tune found again without it, since it hid the notes after it.
    """
    stray: set[int] = set()
    while True:
        out = _top_voice([q for q in notes if id(q) not in stray], beat)
        spikes = set()
        for i, q in enumerate(out):
            around = sorted(o.pitch for o in out[max(0, i - 3) : i] + out[i + 1 : i + 4])
            if around and q.pitch - around[len(around) // 2] > MELODY_SPIKE:
                spikes.add(id(q))
        if not spikes:
            return out
        stray |= spikes


def _on_grid(line: list[QNote], beat: Fraction, grid: Fraction) -> list[QNote]:
    """`line` with onsets rounded to `grid`, one note per position, each held to the next."""
    best: dict[Fraction, QNote] = {}
    for q in line:
        at = round(q.start / grid) * grid
        if at not in best or abs(q.start - at) < abs(best[at].start - at):
            best[at] = q
    starts = sorted(best)
    out = []
    for i, at in enumerate(starts):
        q = best[at]
        # Releases round up to the grid; a gap shorter than a beat is closed (legato).
        end = max(at + grid, -(-q.end // grid) * grid)
        if i + 1 < len(starts):
            nxt = starts[i + 1]
            end = nxt if nxt - end < beat else min(end, nxt)
        out.append(QNote(at, end, q.pitch, q.velocity, False, True))
    return out


def _partitions(length: Fraction, beat: Fraction) -> list[list[Fraction]]:
    """Ways to split a bar into chord segments: whole, halves, or 2 + 1 / 1 + 2 beats in triple time.

    A bar of more than four beats (where the bars restarted around free
    timing) is cut into twos, the last three if odd, so no chord lasts longer
    than a whole note.
    """
    beats = length / beat
    if beats > 4:
        n = int(beats)
        parts = [2 * beat] * (n // 2 - 1) + [length - 2 * beat * (n // 2 - 1)]
        return [parts]
    out = [[length]]
    if beats.denominator == 1 and beats % 2 == 0:
        out.append([length / 2, length / 2])
    elif beats == 3:
        out += [[2 * beat, beat], [beat, 2 * beat]]
    return out


def _weights(notes: list[QNote], a: Fraction, b: Fraction) -> dict[int, float]:
    """Pitch-class weight in [a, b): beats sounding there x velocity."""
    w: dict[int, float] = defaultdict(float)
    for q in notes:
        overlap = min(q.end, b) - max(q.start, a)
        if overlap > 0:
            w[q.pitch % 12] += float(overlap) * q.velocity / 127
    return w


def _chord(notes: list[QNote], a: Fraction, b: Fraction) -> tuple[list[int], float] | None:
    """The block chord for [a, b) (MIDI pitches, bass first) and the weight of notes it leaves out."""
    sounding = [q for q in notes if q.start < b and q.end > a]
    if not sounding:
        return None
    starting = [q for q in sounding if q.start >= a] or sounding
    bass = min(q.pitch for q in starting)
    w = _weights(sounding, a, b)
    strongest = max(w.values())
    pcs = [bass % 12]
    for pc in sorted(w, key=lambda pc: -w[pc]):
        # Skip a step from a tone already chosen (a passing note, a suspension):
        # chords of thirds and fifths read at a glance, clusters don't.
        near = any(min((pc - x) % 12, (x - pc) % 12) <= 2 for x in pcs)
        if len(pcs) <= CHORD_TONES and not near and w[pc] >= CHORD_TONE_SHARE * strongest:
            pcs.append(pc)
    misfit = sum(v for pc, v in w.items() if pc not in pcs)

    lo, hi = BASS_RANGE
    while bass > hi:
        bass -= 12
    while bass < lo:
        bass += 12
    floor = max(bass, CHORD_FLOOR)
    tones = sorted(floor + (pc - floor) % 12 for pc in pcs[1:])  # never the bass's pitch class, so above it
    return [bass] + tones, misfit


def accompaniment(notes: list[QNote], bars: list[tuple[Fraction, Fraction]], beat: Fraction) -> list[QNote]:
    """Block chords for `notes`, one per bar or per part of a bar where the harmony changes."""
    out = []
    for start, end in bars:
        best = None
        for parts in _partitions(end - start, beat):
            cost, chords, a = SEGMENT_COST * (len(parts) - 1), [], start
            for length in parts:
                found = _chord(notes, a, a + length)
                if found:
                    cost += found[1]
                    chords.append((a, a + length, found[0]))
                a += length
            if best is None or cost < best[0]:
                best = (cost, chords)
        for a, b, pitches in best[1]:
            # Two halves with the same chord are one held chord.
            if out and out[-1][2] == pitches and out[-1][1] == a and a != start and b - out[-1][0] <= 4 * beat:
                out[-1] = (out[-1][0], b, pitches)
            else:
                out.append((a, b, pitches))
    return [QNote(a, b, p, 64, False, False) for a, b, pitches in out for p in pitches]


def arrange(
    notes: list[QNote], right: list[bool], bars: list[tuple[Fraction, Fraction]], beat: Fraction, quarter_seconds: float
) -> list[QNote]:
    """The melody (right hand) over block chords (left hand).

    `right` says which hand played each note, `bars` are (start, end) in
    quarter notes, `beat` the beat's length and `quarter_seconds` how long a
    quarter note lasts.
    """
    tune = melody([q for q, r in zip(notes, right) if r], beat)
    chosen = {id(q) for q in tune}
    rest = [q for q in notes if id(q) not in chosen]
    grid = Fraction(1, 4) if quarter_seconds / 4 >= MELODY_FASTEST else Fraction(1, 2)
    return _on_grid(tune, beat, grid) + accompaniment(rest, bars, beat)
