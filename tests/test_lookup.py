"""Looking a recording up in the score library and notating it with the score's beats and bars."""

import json

import numpy as np
import pretty_midi
import pytest
from make_test_audio import c_major_4_4, fur_elise_3_8, waltz_f_3_4
from test_pipeline import transcription

from sheetmusicgen import lookup
from sheetmusicgen.pipeline import Options, Transcription, notate
from sheetmusicgen.transcribe import Note


def waltz_with_pickup():
    """The test waltz twice over, after a one-beat pickup: bars start on beats 1, 4, 7, ..."""
    body = waltz_f_3_4()
    span = max(s + d for s, d, _, _ in body)
    return [(0, 1, 67, 80)] + [(s + 1 + k * span, d, p, v) for k in (0, 1) for s, d, p, v in body]


def write_entry(index, entry_id, title, composer, events, bars):
    pm = pretty_midi.PrettyMIDI(initial_tempo=60)  # one quarter per second
    inst = pretty_midi.Instrument(0)
    inst.notes = [pretty_midi.Note(v, p, float(s), float(s + d)) for s, d, p, v in events]
    pm.instruments.append(inst)
    path = index / "midi" / f"{entry_id}.printed.mid"
    path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(path))
    return {
        "id": entry_id, "title": title, "piece": "", "subtitle": "", "opus": "", "composer": composer,
        "composer_id": "", "instrument": "Piano", "license": "Public Domain", "key": "",
        "profile": lookup._pc_profile(inst.notes),
        "forms": {"printed": {"midi": str(path.relative_to(index)), "bars": [[str(q), n, d] for q, n, d in bars],
                              "seconds": pm.get_end_time(), "notes": len(inst.notes)}},
    }


@pytest.fixture
def library(tmp_path):
    index = tmp_path / "scores"
    waltz = waltz_with_pickup()
    end = max(s + d for s, d, _, _ in waltz)
    c = c_major_4_4() * 2
    c = [(s + 24 * (i >= len(c) // 2), d, p, v) for i, (s, d, p, v) in enumerate(c)]
    entries = [
        write_entry(index, "Test/waltz", "Waltz in F", "Anna Example", waltz, [(q, 3, 4) for q in range(1, int(end), 3)]),
        write_entry(index, "Test/cmajor", "Song in C", "Otto Other", c, [(q, 4, 4) for q in range(0, 48, 4)]),
    ]
    (index / "index.json").write_text(json.dumps({"entries": entries}))
    lookup._INDEX.clear()
    return index


def played(events, bpm=120, rubato=0.15):
    """A performance of score `events` (in beats): the tempo swells and relaxes by `rubato`."""
    beats = np.arange(0, max(s + d for s, d, _, _ in events) + 2)
    period = 60 / bpm * (1 + rubato * np.sin(beats / 5))
    times = 0.5 + np.concatenate([[0], np.cumsum(period[:-1])])
    at = lambda b: float(np.interp(b, beats, times))  # noqa: E731
    return sorted((Note(at(s), at(s + d), p, v) for s, d, p, v in events), key=lambda n: (n.onset, n.pitch)), at


def test_finds_the_piece_and_carries_its_bars_onto_the_recording(library):
    notes, at = played(waltz_with_pickup())
    ref = lookup.find(notes, "Anna Example - Waltz (live)", index=library)
    assert ref is not None and ref.id == "Test/waltz"
    assert ref.time_sig == "3/4" and ref.f1 > 0.9
    want = [at(b) for b in range(1, 37, 3)]
    assert np.allclose(ref.downbeats[: len(want)], want, atol=0.05)
    assert abs(ref.beats[0] - at(0)) < 0.05  # the pickup beat


def test_finds_the_piece_without_a_title(library):
    notes, _ = played(waltz_with_pickup())
    ref = lookup.find(notes, "", index=library)
    assert ref is not None and ref.id == "Test/waltz"


def test_rejects_a_piece_not_in_the_library(library):
    notes, _ = played(fur_elise_3_8())
    assert lookup.find(notes, "Waltz in F", index=library) is None


def test_title_rules_out_other_composers(library):
    entries, idf, composers = lookup.load_index(library)
    scores = lookup.title_scores("Otto Other: Waltz", entries, idf, composers)
    assert scores[[e["id"] for e in entries].index("Test/waltz")] < 0


def test_no_library_no_match(tmp_path):
    notes, _ = played(waltz_with_pickup())
    assert lookup.find(notes, "Waltz in F") is None  # conftest points the library at an empty directory


def test_notate_follows_the_matched_score(library, tmp_path):
    notes, at = played(waltz_with_pickup())
    ref = lookup.find(notes, "Waltz in F", index=library)
    t = Transcription(notes, notes[-1].offset + 1, tmp_path / "x.transcribed.mid")
    r = notate(t, tmp_path, "song", Options(pdf=False), reference=ref)
    assert r.time_sig == "3/4" and r.reference and "Waltz in F" in r.reference
    downs = np.array(r.downbeat_times)
    for b in range(1, 37, 3):  # every bar of the score starts a bar of ours
        assert np.min(np.abs(downs - at(b))) < 0.05

    # a forced meter means the user disagrees: the score is not used
    forced = notate(t, tmp_path, "song", Options(pdf=False, time_sig="4/4"), reference=ref)
    assert forced.reference is None and forced.time_sig == "4/4"


def test_reference_roundtrip():
    ref = lookup.Reference("a/b", "Waltz", "Anna", "printed", 0.9, "3/4", [0.5, 1.0], [1.0])
    assert lookup.Reference.from_dict(json.loads(json.dumps(ref.to_dict()))) == ref


def test_unsupported_meter_gives_no_structure():
    assert lookup._time_sig([["0", 5, 8], ["5/2", 5, 8]]) is None
    assert lookup._time_sig([["0", 6, 8], ["3", 2, 4], ["5", 6, 8]]) == ("6/8", 1.5)


def test_plain_transcription_still_notates_without_library(tmp_path):
    t = transcription(c_major_4_4(), 100, tmp_path)
    assert notate(t, tmp_path, "song", Options(pdf=False)).reference is None
