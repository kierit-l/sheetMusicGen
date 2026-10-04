from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from make_test_audio import c_major_4_4, fur_elise_3_8, render, waltz_f_3_4

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
    assert r.time_sig == "4/4"
    assert r.key == "C major"
    assert Path(tmp_path / "c_major.notes.json").exists()


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
    ]
    out = {q.pitch: q.end for q in release_pedaled(notes, 60, F(3, 2))}
    assert out[57] == F(3, 4)  # handed over to the right hand
    assert out[40] == F(3)  # stops at the barline
    assert out[76] == F(9, 2)


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
