from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from make_test_audio import c_major_4_4, fur_elise_3_8, render, syncopated_4_4, waltz_f_3_4

from sheetmusicgen import transcribe as transcribe_mod
from sheetmusicgen.pipeline import Options, Transcription, notate, transcribe_file
from sheetmusicgen.transcribe import Note


def as_notes(events, bpm):
    """Known (start beat, beats, pitch, velocity) events -> what a perfect transcriber would return."""
    beat = 60 / bpm
    notes = [Note(0.5 + s * beat, 0.5 + (s + d) * beat, p, v) for s, d, p, v in events]
    return sorted(notes, key=lambda n: (n.onset, n.pitch))


def transcription(events, bpm, tmp_path):
    notes = as_notes(events, bpm)
    return Transcription(notes, max(n.offset for n in notes) + 1, tmp_path / "x.transcribed.mid")


@pytest.mark.parametrize("events, bpm, meter", [(c_major_4_4(), 100, "4/4"), (waltz_f_3_4(), 132, "3/4")])
def test_notate_recovers_tempo_and_meter(tmp_path, events, bpm, meter):
    t = transcription(events, bpm, tmp_path)
    r = notate(t, tmp_path, "song", Options(pdf=False))
    assert r.time_sig == meter
    assert abs(r.bpm - bpm) / bpm < 0.05
    assert r.musicxml.exists() and r.score_midi.exists()
    assert r.pdf is None


def test_notate_writes_pdf(tmp_path):
    r = notate(transcription(c_major_4_4(), 100, tmp_path), tmp_path, "song", Options(pdf_engine="verovio"))
    assert r.pdf_error is None
    assert r.pdf.read_bytes().startswith(b"%PDF")
    assert r.outputs[0] == r.pdf


def test_notate_respects_forced_options(tmp_path):
    t = transcription(c_major_4_4(), 100, tmp_path)
    r = notate(t, tmp_path, "song", Options(bpm=90, time_sig="3/4", pdf=False))
    assert r.time_sig == "3/4"
    assert round(r.bpm) == 90


def test_notate_rejects_empty(tmp_path):
    t = transcription(c_major_4_4(), 100, tmp_path)
    with pytest.raises(ValueError):
        notate(t, tmp_path, "song", Options(min_velocity=128, pdf=False))


def test_transcription_roundtrip(tmp_path):
    t = transcription(waltz_f_3_4(), 132, tmp_path)
    t.pedals = [(0.5, 1.75), (2.0, 3.25)]
    t.beats, t.downbeat_probs = [0.5, 0.95, 1.4], [0.9, 0.1, 0.1]
    t.save(tmp_path / "n.json")
    assert Transcription.load(tmp_path / "n.json") == t


def test_transcribe_file_enforces_length_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(transcribe_mod, "load_audio", lambda p: np.zeros(16000 * 120, dtype=np.float32))
    with pytest.raises(ValueError, match="limit"):
        transcribe_file(tmp_path / "long.mp3", tmp_path, max_seconds=60)


@pytest.mark.slow
def test_end_to_end_with_model(tmp_path):
    audio = tmp_path / "c_major.mp3"
    render(c_major_4_4(), 100, audio)
    t = transcribe_file(audio, tmp_path, device="cpu")
    r = notate(t, tmp_path, "c_major", Options(pdf=False))
    assert r.time_sig in ("4/4", "2/2")  # the harmony moves in half notes, so cut time is a fair reading
    assert r.key == "C major"
    assert Path(tmp_path / "c_major.notes.json").exists()


def test_half_note_tempo_mark_is_drawn():
    # No common system font has a half note symbol, so it must become a path, not text.
    from sheetmusicgen.render import _replace_smufl_text

    svg = (
        '<text x="100" y="200" font-size="0px"><tspan><tspan font-family="Leipzig" font-size="720px">\ueca3</tspan>'
        '</tspan><tspan font-size="405px"> = 119</tspan></text>'
    )
    out = _replace_smufl_text(svg)
    assert "Leipzig" not in out and "\ueca3" not in out
    assert out.startswith('<path transform="translate(100,200)')
    assert "= 119" in out


def test_rendering_works_off_the_main_thread(tmp_path):
    # Gradio runs handlers on worker threads, where Verovio can't find its fonts by default.
    from concurrent.futures import ThreadPoolExecutor

    from sheetmusicgen.render import render_svg_pages

    t = transcription(c_major_4_4(), 100, tmp_path)
    with ThreadPoolExecutor(1) as pool:
        r = pool.submit(notate, t, tmp_path, "song", Options(pdf_engine="verovio")).result()
        pages = pool.submit(render_svg_pages, r.musicxml).result()
    assert r.pdf_error is None
    assert "<svg" in pages[0]


def score_notes(path):
    from music21 import converter

    s = converter.parse(path)
    return sorted(
        (float(n.getOffsetInHierarchy(s)), p.midi, float(n.quarterLength))
        for n in s.flatten().notes
        if not (n.tie and n.tie.type != "start")
        for p in n.pitches
    )


def test_compound_meter(tmp_path):
    # Sixteenths at 0.165 s: the tracker's beat is a dotted eighth, which a 4/4 grid would garble.
    t = transcription(fur_elise_3_8(), 60 / 0.165, tmp_path)
    r = notate(t, tmp_path, "elise", Options(pdf=False))
    assert r.time_sig == "3/8"
    assert abs(r.bpm - 60) < 4  # dotted quarters
    notes = score_notes(r.musicxml)
    bass = [o for o, p, _ in notes if p in (40, 45)]
    assert bass and all(o % 1.5 == 0 for o in bass)  # every bass arpeggio starts a bar
    motif = [o for o, p, _ in notes if p >= 71][:6]
    assert [b - a for a, b in zip(motif, motif[1:])] == [0.25] * 5  # even sixteenths


def test_syncopated_simple_meter_is_not_compound(tmp_path):
    # Chords on 3 + 3 + 2 eighths lead the tracker to dotted quarters that split in three, like 6/8.
    r = notate(transcription(syncopated_4_4(), 150, tmp_path), tmp_path, "sync", Options(pdf=False))
    assert r.time_sig == "4/4"
    assert abs(r.bpm - 150) < 5
    bass = [o for o, p, _ in score_notes(r.musicxml) if p == 45]
    assert bass and len({o % 4 for o in bass}) == 1  # the root falls on the same beat of every bar


def test_fast_syncopated_simple_meter_is_cut_time(tmp_path):
    # As above at 230 bpm: the quarters stay quarters, two half-note beats to the bar.
    r = notate(transcription(syncopated_4_4(), 230, tmp_path), tmp_path, "sync", Options(pdf=False))
    assert r.time_sig == "2/2"
    assert abs(r.bpm - 115) < 4  # half notes
    notes = score_notes(r.musicxml)
    bass = [o for o, p, _ in notes if p == 45]
    assert bass and len({o % 4 for o in bass}) == 1
    assert all(o % 0.5 == 0 for o, _, _ in notes)  # straight eighths: no 16ths, 32nds or triplets


def test_cut_time_can_be_forced(tmp_path):
    t = transcription(c_major_4_4(), 100, tmp_path)
    r = notate(t, tmp_path, "song", Options(bpm=60, time_sig="2/2", pdf=False))
    assert r.time_sig == "2/2"
    assert round(r.bpm) == 60


def test_forced_simple_meter_splits_a_triple_beat(tmp_path):
    # The tracked beat is a dotted quarter splitting in three; forced 4/4 must not take it for a quarter.
    t = transcription(syncopated_4_4(), 150, tmp_path)
    r = notate(t, tmp_path, "sync", Options(time_sig="4/4", pdf=False))
    assert abs(r.bpm - 150) < 5


def test_fast_runs_get_32nds(tmp_path):
    events = [(b, 1, 48, 90) for b in range(8)]
    events += [(4 + i / 8, 1 / 8, 72 + i % 5, 70) for i in range(8)]  # one beat of 32nds
    r = notate(transcription(events, 80, tmp_path), tmp_path, "run", Options(pdf=False))
    run = sorted(o for o, p, _ in score_notes(r.musicxml) if p >= 72)
    assert len(run) == 8
    assert [b - a for a, b in zip(run, run[1:])] == [0.125] * 7


def test_pedal_held_notes_are_shortened(tmp_path):
    from sheetmusicgen.notation import release_pedaled
    from sheetmusicgen.rhythm import QNote

    F = Fraction
    notes = [
        QNote(F(0), F(3), 45, 60, True),  # bass arpeggio rung on by the pedal...
        QNote(F(1, 4), F(3), 52, 60, True),
        QNote(F(1, 2), F(3), 57, 60, True),
        QNote(F(3, 4), F(3), 60, 60, True),  # ...and continued by the right hand
        QNote(F(1), F(5, 4), 64, 60),
        QNote(F(3, 2), F(9, 2), 40, 60, True),  # long pedal note, alone in its staff
        QNote(F(3, 2), F(9, 2), 76, 60),  # held key: left alone
        QNote(F(9, 2), F(9), 43, 60, True),  # final chord: left to ring
    ]
    out = {q.pitch: q.end for q in release_pedaled(notes, 60, [F(3, 2) * k for k in range(1, 8)])}
    assert out[57] == F(3, 4)  # handed over to the right hand
    assert out[40] == F(3)  # stops at the barline
    assert out[76] == F(9, 2)
    assert out[43] == F(9)


def test_title_with_decomposed_umlaut():
    from sheetmusicgen.pipeline import safe_stem

    assert safe_stem("Beethoven - Für Elise") == "Beethoven_-_Für_Elise"


def test_jittery_on_beat_playing_stays_simple_meter(tmp_path):
    # Slightly early/late notes on the beat must not look like a triple subdivision.
    rng = np.random.default_rng(0)
    t = transcription(waltz_f_3_4(), 132, tmp_path)
    for n in t.notes:
        d = float(rng.uniform(-0.04, 0.02))
        n.onset, n.offset = n.onset + d, n.offset + d
    t.notes.sort(key=lambda n: (n.onset, n.pitch))
    assert notate(t, tmp_path, "w", Options(pdf=False)).time_sig == "3/4"


def test_unrelated_outro_after_long_silence_is_dropped(tmp_path):
    t = transcription(c_major_4_4(), 100, tmp_path)
    end = max(n.offset for n in t.notes)
    jingle = [Note(end + 8 + i * 0.3, end + 8.25 + i * 0.3, p, 70) for i, p in enumerate((61, 66, 70))]
    t.notes += jingle
    t.duration = jingle[-1].offset + 1
    r = notate(t, tmp_path, "song", Options(pdf=False))
    assert r.dropped_notes == len(jingle)
    assert r.key == "C major"
    assert all(p not in (61, 66, 70) for _, p, _ in score_notes(r.musicxml))


def test_long_pause_inside_the_piece_is_kept():
    from sheetmusicgen.rhythm import drop_detached

    half = [Note(i * 0.5, i * 0.5 + 0.4, 60 + i % 12, 70) for i in range(40)]
    notes = half + [Note(n.onset + 30, n.offset + 30, n.pitch, n.velocity) for n in half]
    assert len(drop_detached(notes)) == len(notes)


def test_key_near_tie_is_settled_by_the_final_chord():
    from sheetmusicgen.notation import detect_key
    from sheetmusicgen.rhythm import QNote

    F = Fraction
    # Mostly A-major material (no D#), ending on an E major chord over E in the bass.
    pcs = [64, 66, 68, 69, 71, 73, 64, 71, 68, 73, 69, 66, 62, 64]
    notes = [QNote(F(i), F(i + 1), p, 70) for i, p in enumerate(pcs * 4)]
    last = F(len(notes))
    notes += [QNote(last, last + 4, p, 70) for p in (40, 52, 56, 59, 64)]
    assert str(detect_key(notes)) == "E major"


def test_two_against_three_between_hands(tmp_path):
    # Right hand in eighths, left hand in triplet eighths, as in Debussy's first Arabesque.
    events = []
    for beat in range(16):
        events += [(beat + i / 2, 1 / 2, 76 + 2 * i, 80) for i in range(2)]
        events += [(beat + i / 3, 1 / 3, (40, 47, 52)[i], 60) for i in range(3)]
    r = notate(transcription(events, 80, tmp_path), tmp_path, "two3", Options(pdf=False))
    notes = score_notes(r.musicxml)
    rh = [o % 1 for o, p, _ in notes if p >= 60]
    lh = [o % 1 for o, p, _ in notes if p < 60]
    assert set(rh) == {0, 0.5}
    assert {round(x, 3) for x in lh} == {0, 0.333, 0.667}


def test_key_signature_accidentals_are_not_repeated(tmp_path):
    # D major scale: F# and C# are in the key signature and need no accidental.
    events = [(i, 1, p, 80) for i, p in enumerate([62, 64, 66, 67, 69, 71, 73, 74] * 2)]
    events += [(b, 4, p, 70) for b in range(0, 16, 4) for p in (38, 45)]
    r = notate(transcription(events, 100, tmp_path), tmp_path, "d", Options(pdf=False))
    assert r.key == "D major"
    assert "<accidental>" not in r.musicxml.read_text()


def test_bars_start_on_the_bass_when_beats_were_tracked_on_the_offbeats():
    # Steady eighths with a long bass note opening each bar (as in Bach's C major
    # prelude): if the tracker sat on the off-beats, the bars still start on the bass.
    from sheetmusicgen.rhythm import quantize

    beat = 0.5
    events = []
    for bar in range(8):
        events.append((4 * bar, 4, 36 + (bar % 2) * 5, 80))
        events += [(4 * bar + i / 2, 1 / 2, (64, 67, 72, 76)[i % 4], 64) for i in range(8)]
    notes = as_notes(events, 60 / beat)
    offbeats = np.arange(0.5 + beat / 2 - beat, 0.5 + 34 * beat, beat)
    r = quantize(notes, offbeats)
    assert r.time_sig == "4/4"
    assert {q.start % 4 for q in r.notes if q.pitch < 60} == {0}
    assert {q.start % Fraction(1, 2) for q in r.notes} == {0}


def test_opening_chord_starts_the_bar_when_accents_are_nearly_even():
    # Every beat a chord, the second beat of each bar a touch louder: the
    # piece's opening chord, not that beat, is the downbeat.
    from sheetmusicgen.rhythm import quantize

    events = [(b, 1, p, 76 if b % 4 == 1 else 72) for b in range(32) for p in (48, 64, 67)]
    notes = as_notes(events, 100)
    beats = np.arange(0.5, 0.5 + 34 * 0.6, 0.6)
    r = quantize(notes, beats, meter=4)
    assert r.notes[0].start == 0


def test_notes_go_to_the_hand_that_plays_them():
    from sheetmusicgen.rhythm import extend_beats, quantize

    # A left-hand melody above middle C, under right-hand chords.
    notes = [Note(0.5 + i * 0.5, 1.0 + i * 0.5, 62 + i % 3, 80) for i in range(16)]
    notes += [Note(0.5 + i, 1.5 + i, p, 70) for i in range(8) for p in (72, 76)]
    notes.sort(key=lambda n: (n.onset, n.pitch))
    beats = extend_beats(np.arange(0.5, 10, 0.5), 10)
    r = quantize(notes, beats, right_hand=[n.pitch >= 70 for n in notes])
    assert all(q.right == (q.pitch >= 70) for q in r.notes)
    from sheetmusicgen.notation import build_score

    score, _ = build_score(r, "t")
    treble, bass = ({p.midi for p in part.pitches} for part in score.parts)
    assert treble == {72, 76} and bass == {62, 63, 64}


@pytest.mark.slow
def test_hand_model_puts_a_bass_line_in_the_left_hand():
    from sheetmusicgen.pm2s import right_hand_probs

    # Alberti bass under a melody, both crossing middle C.
    notes = []
    for bar in range(16):
        t = 0.5 + bar * 2
        notes += [Note(t + i * 0.25, t + i * 0.25 + 0.2, p, 60) for i, p in enumerate([48, 55, 52, 55] * 2)]
        notes += [Note(t + i * 0.5, t + i * 0.5 + 0.45, p, 85) for i, p in enumerate([72, 74, 76, 74])]
    notes.sort(key=lambda n: (n.onset, n.pitch))
    right = right_hand_probs(notes) > 0.5
    assert np.mean([r == (n.pitch >= 60) for n, r in zip(notes, right)]) > 0.9


def test_cli_runs_with_every_progress_step(tmp_path, monkeypatch, capsys):
    import sheetmusicgen
    from sheetmusicgen import pipeline

    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"")

    def fake_transcribe(path, outdir, progress=lambda m: None, **kwargs):
        for msg in ("Loading", "Transcribing", "Tracking beats"):  # the beat model may outlast the transcriber
            progress(msg)
        return transcription(waltz_f_3_4(), 132, tmp_path)

    monkeypatch.setattr(pipeline, "transcribe_file", fake_transcribe)
    assert sheetmusicgen.main([str(audio), "-o", str(tmp_path), "--no-pdf"]) == 0
    out = capsys.readouterr().out
    assert "3/4" in out and (tmp_path / "song.musicxml").exists()


def test_clef_changes_for_long_passages_out_of_a_staffs_range():
    from music21 import clef as m21clef

    from sheetmusicgen.notation import build_score
    from sheetmusicgen.rhythm import QNote, Rhythm

    F = Fraction
    # The left hand climbs into the treble for four bars, and plays one high chord alone later.
    left = [48 if b < 4 or b >= 8 else 74 for b in range(16) for _ in range(4)]
    left[14 * 4] = 76
    notes = [QNote(F(i), F(i + 1), p, 70, right=False) for i, p in enumerate(left)]
    notes += [QNote(F(i), F(i + 1), 79, 70, right=True) for i in range(64)]
    score, _ = build_score(Rhythm(notes, 100, "4/4"), "t")
    lh = score.parts[1]
    changes = [(c.getContextByClass("Measure").number, type(c).__name__) for c in lh.recurse().getElementsByClass(m21clef.Clef)]
    assert changes == [(1, "BassClef"), (5, "TrebleClef"), (9, "BassClef")]


def test_note_released_before_a_rest_is_written_on_a_plain_value():
    from sheetmusicgen.notation import build_score
    from sheetmusicgen.rhythm import QNote, Rhythm

    F = Fraction
    # A chord held a quarter and a 16th, then silence until beat 3: a quarter and rests, not a tie to a 16th.
    notes = [QNote(F(0), F(5, 4), p, 70, right=True) for p in (64, 67)] + [QNote(F(2), F(4), 72, 70, right=True)]
    notes += [QNote(F(i), F(i + 1), 48, 70, right=False) for i in range(4)]
    score, _ = build_score(Rhythm(notes, 100, "4/4"), "t")
    rh = [(float(e.offset), float(e.quarterLength), e.isRest) for e in score.parts[0].recurse().notesAndRests]
    assert rh[0] == (0.0, 1.0, False) and not any(e[2] is False and e[1] == 0.25 for e in rh)
