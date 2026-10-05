"""Beats tracked on the audio: decoding the model's activations and using them to notate."""

from fractions import Fraction

import numpy as np
from make_test_audio import waltz_f_3_4
from test_pipeline import as_notes, transcription

from sheetmusicgen.notation import build_score
from sheetmusicgen.pipeline import Options, notate
from sheetmusicgen.rhythm import QNote, Rhythm, beat_division, decode_beats, extend_beats, quantize, snap_beats
from sheetmusicgen.transcribe import Note

FPS = 50


def activations(beat_frames, strength, length, downbeat_every=4):
    beat, down = np.zeros(length), np.zeros(length)
    for i, (f, s) in enumerate(zip(beat_frames, strength)):
        beat[f] = s
        down[f] = 0.9 if i % downbeat_every == 0 else 0.05
    return beat, down


def test_decoding_fills_beats_the_model_barely_saw():
    frames = np.arange(50, 3000, 25)  # 120 bpm
    strength = np.full(len(frames), 0.9)
    strength[20:28] = 0.05  # a quiet passage: peak picking would lose these beats
    beats, probs = decode_beats(*activations(frames, strength, 3100), FPS)
    assert np.allclose(np.diff(beats), 0.5, atol=0.021)
    assert len(beats) >= len(frames) - 1
    down = np.array(beats)[np.array(probs) > 0.5]
    assert np.allclose(np.diff(down), 2.0, atol=0.05)


def test_decoding_does_not_drift_onto_eighths():
    # Off-beat eighths show up weakly in the activation in the second half (a
    # busy passage); stronger than BEAT_THRESHOLD they would count as beats.
    frames = np.arange(50, 4000, 30)
    beat, down = activations(frames, np.full(len(frames), 0.9), 4100)
    for f in frames[len(frames) // 2 :]:
        beat[f + 15] = 0.15
    beats, _ = decode_beats(beat, down, FPS)
    assert np.allclose(np.diff(beats), 0.6, atol=0.03)


def test_beats_snap_onto_the_notes():
    notes = [Note(1.0 + i * 0.5, 1.2 + i * 0.5, 60, 80) for i in range(8)]
    beats = np.array([0.5, 1.02, 1.49, 2.03, 2.5, 3.02, 3.5, 4.01, 4.5, 5.0])
    snapped = snap_beats(beats, notes)
    assert np.allclose(snapped[1:9], [n.onset for n in notes])
    assert abs(snapped[0] - 0.5) < 0.03 and abs(snapped[9] - 5.0) < 0.03  # no note: median shift


def grid(notes, beat):
    last = notes[-1].offset + 1
    return extend_beats(np.arange(0.5, last, beat), last)


def test_downbeat_probabilities_set_meter_and_phase():
    notes = as_notes(waltz_f_3_4(), 132)
    beat = 60 / 132
    beats = grid(notes, beat)
    # Bars of three, but starting a beat later than the accents would put them.
    probs = [(t, 0.8 if i % 3 == 1 else 0.1) for i, t in enumerate(beats[: len(beats) - 4])]
    r = quantize(notes, beats, downbeat_probs=probs)
    assert r.time_sig == "3/4"
    assert all(abs(t - beats[1]) % (3 * beat) < 1e-6 or abs(abs(t - beats[1]) % (3 * beat) - 3 * beat) < 1e-6 for t in r.downbeat_times)


def test_bars_restart_where_the_downbeats_move():
    beat = 0.5
    notes = [Note(0.5 + i * beat, 0.5 + (i + 1) * beat, 60 + i % 5, 80) for i in range(48)]
    beats = grid(notes, beat)
    # 4/4, but one bar has only three beats (beat 20 starts a bar instead of beat 21).
    down = {0, 4, 8, 12, 16, 19, 23, 27, 31, 35, 39, 43, 47}
    probs = [(beats[i], 0.9 if i in down else 0.05) for i in range(48)]
    r = quantize(notes, beats, downbeat_probs=probs)
    assert r.time_sig == "4/4"
    starts = [int(b) for b in r.bar_starts]
    assert starts[:7] == [0, 4, 8, 12, 16, 19, 23]
    lengths = [b - a for a, b in r.bars(Fraction(48))]
    assert lengths.count(3) == 1


def test_irregular_bar_gets_its_own_time_signature():
    F = Fraction
    notes = [QNote(F(i), F(i + 1), 60 + i % 7, 80) for i in range(15)]
    score, _ = build_score(Rhythm(notes, 100, "4/4", bar_starts=[F(0), F(4), F(7), F(11)]), "t")
    measures = list(score.parts[0].getElementsByClass("Measure"))
    assert [m.barDuration.quarterLength for m in measures] == [4, 3, 4, 4]
    assert measures[1].timeSignature.ratioString == "3/4" and measures[2].timeSignature.ratioString == "4/4"


def test_compound_bars_of_four_beats_are_written_in_6_8():
    beat = 0.6
    notes = [Note(0.5 + i * beat / 3, 0.5 + (i + 1) * beat / 3, 60 + i % 5, 80) for i in range(96)]
    beats = grid(notes, beat)
    probs = [(beats[i], 0.9 if i % 4 == 0 else 0.6 if i % 2 == 0 else 0.05) for i in range(32)]
    assert quantize(notes, beats, compound=True, downbeat_probs=probs).time_sig == "6/8"


def test_sixteenths_in_compound_meter_divide_the_beat_in_three():
    beat = 0.9
    notes = []
    for b in range(24):  # 6/8 figures: three eighths, then six sixteenths
        offsets = (0, 1 / 3, 2 / 3) if b % 2 else (0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6)
        notes += [Note(0.5 + (b + o) * beat, 0.5 + (b + o) * beat + 0.1, 64, 80) for o in offsets]
    assert beat_division(notes, np.arange(0.5, 24 * beat + 2, beat)) == 3


def test_notate_uses_audio_beats(tmp_path):
    t = transcription(waltz_f_3_4(), 132, tmp_path)
    beat = 60 / 132
    # The model tracked the beats and put the bars one beat later than the accents suggest.
    t.beats = list(np.arange(0.5, t.duration, beat))
    t.downbeat_probs = [0.8 if i % 3 == 1 else 0.1 for i in range(len(t.beats))]
    r = notate(t, tmp_path, "w", Options(pdf=False))
    assert r.time_sig == "3/4"
    assert abs(r.bpm - 132) < 2
    assert min(abs(d - t.beats[1]) for d in r.downbeat_times) < 1e-6


def test_forced_meter_that_disagrees_with_audio_beats_tracks_onsets(tmp_path):
    t = transcription(waltz_f_3_4(), 132, tmp_path)
    t.beats = list(np.arange(0.5, t.duration, 60 / 132))
    t.downbeat_probs = [0.8 if i % 3 == 0 else 0.1 for i in range(len(t.beats))]
    r = notate(t, tmp_path, "w", Options(time_sig="6/8", pdf=False))
    assert r.time_sig == "6/8"


def test_note_values_can_be_doubled_or_halved(tmp_path):
    t = transcription(waltz_f_3_4(), 132, tmp_path)
    t.beats = list(np.arange(0.5, t.duration, 60 / 132))
    t.downbeat_probs = [0.8 if i % 3 == 0 else 0.1 for i in range(len(t.beats))]
    durations = lambda r: sorted({float(e.quarterLength) for e in build_notes(r)})  # noqa: E731
    auto = notate(t, tmp_path, "a", Options(pdf=False))
    double = notate(t, tmp_path, "d", Options(pdf=False, note_values="double"))
    assert auto.time_sig == "3/4" and double.time_sig == "3/2"
    assert durations(double) == [2 * d for d in durations(auto)]
    halve = notate(t, tmp_path, "h", Options(pdf=False, note_values="halve"))
    assert abs(halve.bpm - 66) < 2


def build_notes(result):
    from music21 import converter

    return list(converter.parse(str(result.musicxml)).flatten().notes)


def test_close_meters_are_offered_as_alternatives():
    beat = 0.5
    notes = [Note(0.5 + i * beat, 0.5 + (i + 1) * beat, 60 + i % 5, 80) for i in range(48)]
    beats = grid(notes, beat)
    # Downbeats every 4 beats, with the half bar nearly as strong: 2/4 is a fair reading too.
    probs = [(beats[i], 0.9 if i % 4 == 0 else 0.5 if i % 2 == 0 else 0.05) for i in range(48)]
    r = quantize(notes, beats, downbeat_probs=probs)
    assert r.time_sig == "4/4" and r.other_meters == ["2/4"]
    clear = [(beats[i], 0.9 if i % 4 == 0 else 0.05) for i in range(48)]
    assert quantize(notes, beats, downbeat_probs=clear).other_meters == []

