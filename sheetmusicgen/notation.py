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
)

from .rhythm import QNote, Rhythm

SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "D-", "D", "E-", "E", "F", "G-", "G", "A-", "A", "B-", "B"]


def detect_key(notes: list[QNote]) -> key.Key:
    s = stream.Stream()
    for q in notes:
        n = note.Note(q.pitch)
        n.quarterLength = min(q.end - q.start, 4)
        s.append(n)
    if not notes:
        return key.Key("C")
    return s.analyze("key")


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


def release_pedaled(notes: list[QNote], split: int, bar: Fraction) -> list[QNote]:
    """Give pedal-held notes the length a pianist would write, not how long they rang.

    A note that is part of a fast figure carried on by the other hand (an
    arpeggio climbing from the bass into the treble) ends where the other hand
    takes over; any other pedal-held note ends at the barline at the latest.
    """
    onsets: dict[Fraction, list[QNote]] = defaultdict(list)
    for q in notes:
        onsets[q.start].append(q)
    times = sorted(onsets)
    onsets_by_hand = {hand: {t for t in times if any((q.pitch >= split) == hand for q in onsets[t])} for hand in (True, False)}
    staff_times = {hand: sorted(ts) for hand, ts in onsets_by_hand.items()}

    out = []
    for q in notes:
        if not q.pedaled:
            out.append(q)
            continue
        end = min(q.end, (q.start // bar + 1) * bar)
        hand = q.pitch >= split
        mine = staff_times[hand]
        i = bisect_right(mine, q.start) - 1
        j = bisect_right(times, q.start)
        nxt = times[j] if j < len(times) else None
        if i > 0 and nxt is not None and nxt not in onsets_by_hand[hand]:
            step = q.start - mine[i - 1]
            near = min(abs(o.pitch - q.pitch) for o in onsets[nxt])
            if nxt - q.start <= step <= Fraction(1, 2) and near <= 12:
                end = min(end, nxt)
        out.append(QNote(q.start, end, q.pitch, q.velocity, q.pedaled))
    return out


def build_staff(notes: list[QNote], spell, part: stream.PartStaff) -> None:
    """Fill one staff with notes/chords/rests, one voice per staff.

    Notes starting together become a chord; each chord lasts until the next
    onset on this staff at the latest, which keeps the notation to a single
    readable voice.
    """
    groups: dict[Fraction, list[QNote]] = defaultdict(list)
    for q in notes:
        groups[q.start].append(q)
    onsets = sorted(groups)

    cursor = Fraction(0)
    for i, start in enumerate(onsets):
        if start > cursor:
            part.insert(float(cursor), note.Rest(quarterLength=start - cursor))
        group = groups[start]
        end = max(q.end for q in group)
        if i + 1 < len(onsets):
            nxt = onsets[i + 1]
            # Players release a little early; close small gaps instead of
            # writing fussy dotted-note + short-rest rhythms.
            gap = nxt - end
            if 0 < gap <= min(Fraction(1, 2), (end - start) / 2):
                end = nxt
            end = min(end, nxt)
        length = end - start

        pitches = [spell(q.pitch) for q in sorted(group, key=lambda q: q.pitch)]
        el = note.Note(pitches[0]) if len(pitches) == 1 else chord.Chord(pitches)
        el.quarterLength = length
        el.volume.velocity = max(q.velocity for q in group)
        part.insert(start, el)
        cursor = end


def build_score(rhythm: Rhythm, title: str, split: int = 60) -> tuple[stream.Score, key.Key]:
    k = detect_key(rhythm.notes)
    spell = speller(k)
    notes = release_pedaled(rhythm.notes, split, rhythm.bar_length)

    # Hand split: notes at/above the split point go to the treble staff.
    right = [q for q in notes if q.pitch >= split]
    left = [q for q in notes if q.pitch < split]

    bar = rhythm.bar_length
    total = max((q.end for q in notes), default=bar)
    total = Fraction(-(-total // bar) * bar)  # round up to a full bar

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
        p.insert(0, meter.TimeSignature(rhythm.time_sig))
        build_staff(staff_notes, spell, p)
        last = max((e.offset + e.quarterLength for e in p.notesAndRests), default=0)
        if last < total:
            p.insert(last, note.Rest(quarterLength=total - Fraction(last)))
        parts.append(p)

    beat_unit = note.Note(type="quarter", dots=1 if rhythm.compound else 0)
    parts[0].insert(0, tempo.MetronomeMark(number=round(rhythm.bpm), referent=beat_unit))

    for p in parts:
        p.makeMeasures(inPlace=True)
        p.makeTies(inPlace=True)
        for m in p.getElementsByClass(stream.Measure):
            m.makeBeams(inPlace=True)
        score.insert(0, p)
    score.insert(0, layout.StaffGroup(parts, name="Piano", abbreviation="Pno.", symbol="brace"))
    return score, k
