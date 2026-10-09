from fractions import Fraction as F

from music21 import chord, converter
from test_pipeline import transcription

from sheetmusicgen.pipeline import Options, notate
from sheetmusicgen.rhythm import QNote
from sheetmusicgen.simplify import accompaniment, arrange, melody

BAR = [(F(0), F(4))]


def q(start, end, pitch, pedaled=False, right=True, velocity=80):
    return QNote(F(start), F(end), pitch, velocity, pedaled, right)


def test_melody_skips_broken_chords_under_it():
    # E5 and D5 held for two beats each, an arpeggio in eighths below them.
    tune = [q(0, 2, 76), q(2, 4, 74)]
    figure = [q(x / 2, x / 2 + F(1, 2), p) for x, p in zip(range(8), [60, 64, 67, 64, 59, 62, 67, 62])]
    assert [n.pitch for n in melody(tune + figure, F(1))] == [76, 74]


def test_melody_steps_down_under_the_pedal():
    # Everything rings on under the pedal; the tune still moves down by step.
    line = [q(x / 2, 4, p, pedaled=True) for x, p in enumerate([81, 83, 81, 80, 81, 76])]
    assert [n.pitch for n in melody(line, F(1))] == [81, 83, 81, 80, 81, 76]


def test_melody_drops_a_stray_high_note():
    line = [q(x, x + 1, p) for x, p in enumerate([69, 71, 108, 72, 71, 69])]
    assert [n.pitch for n in melody(line, F(1))] == [69, 71, 72, 71, 69]


def test_block_chords_follow_the_harmony():
    # C major broken chords for half a bar, G major for the other half.
    notes = [q(x / 2, x / 2 + F(1, 2), p, right=False) for x, p in zip(range(8), [36, 43, 52, 43, 43, 50, 59, 50])]
    chords = accompaniment(notes, BAR, F(1))
    first = sorted(n.pitch % 12 for n in chords if n.start == 0)
    second = sorted(n.pitch % 12 for n in chords if n.start == 2)
    assert first == [0, 4, 7] and second == [2, 7, 11]
    assert all(n.end - n.start == 2 for n in chords)
    assert min(n.pitch for n in chords if n.start == 0) == 36  # the bass note stays the bass


def test_a_steady_chord_is_held_for_the_bar():
    notes = [q(x / 2, x / 2 + F(1, 2), p, right=False) for x, p in zip(range(8), [36, 43, 52, 43] * 2)]
    chords = accompaniment(notes, BAR, F(1))
    assert {(n.start, n.end) for n in chords} == {(0, 4)}


def test_melody_grid_follows_the_tempo():
    line = [q(x / 4, x / 4 + F(1, 4), 72 + x % 3) for x in range(8)]  # 16ths
    slow = arrange(line, [True] * 8, BAR, F(1), quarter_seconds=1.0)  # a 16th lasts 0.25 s
    fast = arrange(line, [True] * 8, BAR, F(1), quarter_seconds=0.4)  # 0.1 s: too fast for an easy part
    assert len([n for n in slow if n.right]) == 8
    tune = [n for n in fast if n.right]
    assert len(tune) <= 5 and all(n.start % F(1, 2) == 0 for n in tune)


def melody_in_thirds_over_broken_chords():
    """4 bars of 4/4: a tune in thirds (quarters) over broken chords in eighths, as (beat, beats, pitch, velocity)."""
    tune = [76, 74, 72, 74, 76, 76, 76, 74, 74, 74, 76, 79, 79, 77, 76, 72]
    events = [(b, 1, p, 90) for b, p in enumerate(tune)] + [(b, 1, p - 4, 60) for b, p in enumerate(tune)]
    for bar, root in enumerate([48, 43, 45, 48]):
        events += [(4 * bar + i / 2, 0.5, root + [0, 7, 16, 7][i % 4], 55) for i in range(8)]
    return events


def test_notate_simplified(tmp_path):
    t = transcription(melody_in_thirds_over_broken_chords(), 100, tmp_path)
    full = notate(t, tmp_path / "full", "song", Options(pdf=False))
    easy = notate(t, tmp_path / "easy", "song", Options(pdf=False, simplify=True))
    assert easy.time_sig == full.time_sig and easy.key == full.key
    rh, lh = converter.parse(str(easy.musicxml)).parts
    assert not any(isinstance(e, chord.Chord) for e in rh.recurse().notes)  # one melody note at a time
    assert all(len(e.pitches) <= 3 for e in lh.recurse().notes)
    assert all(float(e.offset) % 0.5 == 0 for e in rh.recurse().notes)
    count = lambda r: sum(len(e.pitches) for e in converter.parse(str(r.musicxml)).recurse().notes)  # noqa: E731
    assert count(easy) < count(full)
