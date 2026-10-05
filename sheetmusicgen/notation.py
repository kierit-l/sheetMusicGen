"""Quantized notes -> a two-staff piano score (music21)."""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from fractions import Fraction

from music21 import (
    chord,
    clef,
    instrument,
    key,
    layout,
    metadata,
    meter,
    note,
    pitch,
    scale,
    stream,
    tempo,
    tie,
)

from .rhythm import QNote, Rhythm

SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "D-", "D", "E-", "E", "F", "G-", "G", "A-", "A", "B-", "B"]
KEY_TIE = 0.03  # key-profile correlations this close are a tie
GAP_CLOSE_SHARE = 1.0  # a gap before the next onset up to this share of the note's length is closed...
GAP_CLOSE_MAX = 0.5  # ...and up to this many quarter notes
REST_END_GRID = 0.5  # quarter notes; note ends before a rest round to this grid (0 = off)
CLEF_CHANGE_COST = 12  # ledger lines a clef change must save (over its bar and the bars after)
AWAY_COST = 2  # ledger lines per bar a staff's other clef must save to be kept on
STAFF_LINES = {"treble": ("E4", "F5"), "bass": ("G2", "A3")}  # bottom and top line


def detect_key(notes: list[QNote]) -> key.Key:
    """The key profile that fits the notes best, unless a near tie is settled by the ending.

    Pieces spend long stretches in related keys (the dominant, the
    subdominant), so profiles can rank them almost equally; the lowest note of
    the final chord is then the better guide to the tonic.
    """
    s = stream.Stream()
    for q in notes:
        n = note.Note(q.pitch)
        n.quarterLength = min(q.end - q.start, 4)
        s.append(n)
    if not notes:
        return key.Key("C")
    best = s.analyze("key")
    last = max(q.start for q in notes)
    final_bass = min(q.pitch for q in notes if q.start == last) % 12
    for k in [best] + best.alternateInterpretations:
        if best.correlationCoefficient - k.correlationCoefficient > KEY_TIE:
            break
        if k.tonic.pitchClass == final_bass:
            return k
    return best


def speller(k: key.Key):
    """Map MIDI numbers to note names that fit the key signature."""
    names = list(SHARP_NAMES if k.sharps >= 0 else FLAT_NAMES)
    sc = scale.HarmonicMinorScale(k.tonic) if k.mode == "minor" else scale.MajorScale(k.tonic)
    pitches = sc.getPitches(k.tonic.name + "4", k.tonic.name + "5")
    if k.mode == "minor":
        # Natural 7th too, so both forms of the minor 7th degree are spelled in-key.
        pitches += scale.MinorScale(k.tonic).getPitches(k.tonic.name + "4", k.tonic.name + "5")
    for p in pitches:
        names[p.pitchClass] = p.name

    def spell(midi: int) -> pitch.Pitch:
        p = pitch.Pitch(names[midi % 12])
        # Octave follows the written letter (e.g. B#3 sounds as C4).
        p.octave = 4
        p.octave += (midi - p.midi) // 12
        return p

    return spell


def is_right(q: QNote, split: int) -> bool:
    """Whether the right hand plays `q`: as transcribed, else from where it lies against `split`."""
    return q.right if q.right is not None else q.pitch >= split


def release_pedaled(notes: list[QNote], split: int, bar_ends: list[Fraction]) -> list[QNote]:
    """Give pedal-held notes the length a pianist would write, not how long they rang.

    A note that is part of a fast figure carried on by the other hand (an
    arpeggio climbing from the bass into the treble) ends where the other hand
    takes over; any other pedal-held note ends at the barline at the latest,
    except the final chord, which is left to ring.
    """
    onsets: dict[Fraction, list[QNote]] = defaultdict(list)
    for q in notes:
        onsets[q.start].append(q)
    times = sorted(onsets)
    onsets_by_hand = {hand: {t for t in times if any(is_right(q, split) == hand for q in onsets[t])} for hand in (True, False)}
    staff_times = {hand: sorted(ts) for hand, ts in onsets_by_hand.items()}

    out = []
    for q in notes:
        if not q.pedaled or q.start == times[-1]:
            out.append(q)
            continue
        i = bisect_right(bar_ends, q.start)
        end = min(q.end, bar_ends[i]) if i < len(bar_ends) else q.end
        hand = is_right(q, split)
        mine = staff_times[hand]
        i = bisect_right(mine, q.start) - 1
        j = bisect_right(times, q.start)
        nxt = times[j] if j < len(times) else None
        if i > 0 and nxt is not None and nxt not in onsets_by_hand[hand]:
            step = q.start - mine[i - 1]
            near = min(abs(o.pitch - q.pitch) for o in onsets[nxt])
            if nxt - q.start <= step <= Fraction(1, 2) and near <= 12:
                end = min(end, nxt)
        out.append(QNote(q.start, end, q.pitch, q.velocity, q.pedaled, q.right))
    return out


def _pieces(start: Fraction, end: Fraction, beat: Fraction) -> list[tuple[Fraction, Fraction]]:
    """Split a span so that triplet positions only occur within a beat.

    A note from the second triplet of one beat to the middle of the next has
    no single written value; cut at the beat lines it becomes tied values
    that each fit the grid of their beat.
    """
    cuts = [start]
    if (start / beat).denominator % 3 == 0:
        cuts.append(min(end, (start // beat + 1) * beat))
    if (end / beat).denominator % 3 == 0:
        cuts.append(max(cuts[-1], end // beat * beat))
    cuts.append(end)
    return [(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]


def build_staff(notes: list[QNote], spell, part: stream.Stream, total: Fraction, beat: Fraction = Fraction(1)) -> None:
    """Fill one staff with notes/chords/rests up to `total`, one voice per staff.

    Notes starting together become a chord; each chord lasts until the next
    onset on this staff at the latest, which keeps the notation to a single
    readable voice. `beat` is the length of a beat, where triplets are cut.
    """
    groups: dict[Fraction, list[QNote]] = defaultdict(list)
    for q in notes:
        groups[q.start].append(q)
    onsets = sorted(groups)

    def rests(a: Fraction, b: Fraction) -> None:
        for x, y in _pieces(a, b, beat):
            part.insert(x, note.Rest(quarterLength=y - x))

    cursor = Fraction(0)
    for i, start in enumerate(onsets):
        rests(cursor, start)
        group = groups[start]
        end = max(q.end for q in group)
        if i + 1 < len(onsets):
            nxt = onsets[i + 1]
            # Players release a little early; close small gaps instead of
            # writing fussy dotted-note + short-rest rhythms.
            gap = nxt - end
            if 0 < gap <= min(Fraction(GAP_CLOSE_MAX), (end - start) * Fraction(GAP_CLOSE_SHARE)):
                end = nxt
            end = min(end, nxt)
        nxt = onsets[i + 1] if i + 1 < len(onsets) else total
        if end < nxt and REST_END_GRID:
            # A release before a rest is heard loosely: write it on a plain
            # value rather than as a tie to a 16th and a dotted rest.
            unit = Fraction(REST_END_GRID).limit_denominator(12)
            rounded = round(end / unit) * unit
            if rounded > start:
                end = min(rounded, nxt)
        end = min(end, total)

        pitches = [spell(q.pitch) for q in sorted(group, key=lambda q: q.pitch)]
        pieces = _pieces(start, end, beat)
        for j, (a, b) in enumerate(pieces):
            el = note.Note(pitches[0]) if len(pitches) == 1 else chord.Chord(pitches)
            el.quarterLength = b - a
            el.volume.velocity = max(q.velocity for q in group)
            if len(pieces) > 1:
                el.tie = tie.Tie("start" if j == 0 else "stop" if j == len(pieces) - 1 else "continue")
            part.insert(a, el)
        cursor = end
    rests(cursor, total)


def _ledger_lines(p: pitch.Pitch, which: str) -> int:
    bottom, top = (pitch.Pitch(n).diatonicNoteNum for n in STAFF_LINES[which])
    return max(bottom - p.diatonicNoteNum, p.diatonicNoteNum - top, 0) // 2


def choose_clefs(measures: list[stream.Measure], home: str) -> list[str]:
    """The clef for each bar of a staff: its own, unless the other saves many ledger lines.

    A left hand playing high in the treble (or a right hand low in the bass)
    for a while is easier to read with the clef changed; a single high chord
    is not worth two clef changes. Decided for all bars at once (Viterbi).
    """
    clefs = ("treble", "bass")
    cost = []
    for m in measures:
        ps = [p for n in m.recurse().notes for p in n.pitches]
        cost.append({c: sum(_ledger_lines(p, c) for p in ps) + (AWAY_COST if c != home else 0) for c in clefs})
    if not cost:
        return []
    best = {c: (cost[0][c] + (CLEF_CHANGE_COST if c != home else 0), [c]) for c in clefs}
    for bar in cost[1:]:
        best = {
            c: min(
                ((total + bar[c] + (CLEF_CHANGE_COST if prev != c else 0), path + [c]) for prev, (total, path) in best.items()),
                key=lambda t: t[0],
            )
            for c in clefs
        }
    return min(best.values(), key=lambda t: t[0])[1]


def build_score(rhythm: Rhythm, title: str, split: int = 60) -> tuple[stream.Score, key.Key]:
    k = detect_key(rhythm.notes)
    spell = speller(k)
    bars = rhythm.bars(max((q.end for q in rhythm.notes), default=Fraction(0)))
    notes = release_pedaled(rhythm.notes, split, [b for _, b in bars])

    right = [q for q in notes if is_right(q, split)]
    left = [q for q in notes if not is_right(q, split)]
    total = bars[-1][1]

    score = stream.Score()
    score.metadata = metadata.Metadata(title=title, composer="Transcribed by sheetmusicgen")

    parts = []
    for name, staff_notes, staff_clef in (
        ("RH", right, clef.TrebleClef()),
        ("LH", left, clef.BassClef()),
    ):
        p = stream.PartStaff(id=name)
        p.insert(0, instrument.Piano())
        p.insert(0, staff_clef)
        p.insert(0, key.KeySignature(k.sharps))
        # A bar of another length (where the tracked bars restarted) gets its own time signature.
        current = None
        for start, end in bars:
            sig = rhythm.meter_for(end - start)
            if sig != current:
                p.insert(start, meter.TimeSignature(sig))
                current = sig
        build_staff(staff_notes, spell, p, total, Fraction(3, 2) if rhythm.compound else Fraction(1))
        parts.append(p)

    beat_unit = note.Note(type="half" if rhythm.cut else "quarter", dots=1 if rhythm.compound else 0)
    parts[0].insert(0, tempo.MetronomeMark(number=round(rhythm.bpm), referent=beat_unit))

    for p in parts:
        p.makeMeasures(inPlace=True)
        p.makeTies(inPlace=True)
        measures = list(p.getElementsByClass(stream.Measure))
        home = "treble" if p.id == "RH" else "bass"
        current = home
        for m, c in zip(measures, choose_clefs(measures, home)):
            if c != current:
                m.remove(list(m.getElementsByClass(clef.Clef)))
                m.insert(0, clef.TrebleClef() if c == "treble" else clef.BassClef())
                current = c
        # Show only the accidentals the key signature and the bar so far don't imply.
        stream.makeNotation.makeAccidentalsInMeasureStream(p, useKeySignature=True)
        for m in p.getElementsByClass(stream.Measure):
            m.makeBeams(inPlace=True)
        score.insert(0, p)
    score.insert(0, layout.StaffGroup(parts, name="Piano", abbreviation="Pno.", symbol="brace"))
    return score, k
