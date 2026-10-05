from pathlib import Path

import gradio as gr
import pytest
from make_test_audio import c_major_4_4
from test_pipeline import as_notes

from sheetmusicgen import transcribe as transcribe_mod
from sheetmusicgen import web


def no_progress(*args, **kwargs):
    pass


@pytest.fixture
def fake_model(monkeypatch, tmp_path):
    """Skip ffmpeg and the neural network: 'transcribe' returns known notes."""
    import numpy as np

    monkeypatch.setattr(web, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(transcribe_mod, "load_audio", lambda p: np.zeros(16000 * 20, dtype=np.float32))

    def fake_transcribe(audio, device, midi_path, on_segment=None):
        Path(midi_path).write_bytes(b"MThd")
        if on_segment:
            for i in range(1, 4):
                on_segment(i, 3)
        return as_notes(c_major_4_4(), 100), []

    monkeypatch.setattr(transcribe_mod, "transcribe", fake_transcribe)
    upload = tmp_path / "My Song (live).mp3"
    upload.write_bytes(b"not really audio")
    return upload


def test_build_ui():
    assert isinstance(web.build_ui(), gr.Blocks)


def test_transcribe_then_renotate(fake_model):
    summary, preview, files, state, _ = web.transcribe_upload(
        str(fake_model), "", "", 0, "auto", 4, "auto", 60, 0, progress=no_progress
    )
    assert "4/4" in summary and "C major" in summary
    assert "<svg" in preview
    names = sorted(f.rsplit("/", 1)[-1] for f in files)
    assert names == [
        "My_Song_live.musicxml",
        "My_Song_live.pdf",
        "My_Song_live.score.mid",
        "My_Song_live.transcribed.mid",
    ]
    assert state["stem"] == "My_Song_live"

    summary, _, _ = web.renotate(state, "Other", 0, "3/4", 2, "auto", 60, 0, progress=no_progress)
    assert "3/4" in summary


def test_renotate_without_transcription():
    with pytest.raises(gr.Error):
        web.renotate(None, "", 0, "auto", 4, "auto", 60, 0, progress=no_progress)


def test_bad_audio_cleans_up(fake_model, monkeypatch):
    def broken(path):
        raise RuntimeError("ffmpeg could not decode it")

    monkeypatch.setattr(transcribe_mod, "load_audio", broken)
    with pytest.raises(gr.Error, match="decode"):
        web.transcribe_upload(str(fake_model), "", "", 0, "auto", 4, "auto", 60, 0, progress=no_progress)
    assert list(web.JOBS_DIR.iterdir()) == []


@pytest.fixture
def fake_download(fake_model, monkeypatch):
    calls = []

    def download(url, outdir, max_seconds=None, progress=None):
        calls.append(url)
        path = outdir / "dQw4w9WgXcQ.webm"
        path.write_bytes(b"not really audio")
        return path, "Nocturne Op. 9 No. 2"

    monkeypatch.setattr(web, "download_audio", download)
    return calls


def test_transcribe_youtube_link(fake_download):
    summary, _, files, state, _ = web.transcribe_upload(
        None, " https://youtu.be/dQw4w9WgXcQ ", "", 0, "auto", 4, "auto", 60, 0, progress=no_progress
    )
    assert fake_download == ["https://youtu.be/dQw4w9WgXcQ"]
    assert "C major" in summary
    assert state["stem"] == "Nocturne_Op_9_No_2"
    assert any(f.endswith("Nocturne_Op_9_No_2.musicxml") for f in files)
    assert not list(Path(state["job"]).glob("*.webm"))  # downloaded audio is not kept


def test_upload_wins_over_link(fake_download, fake_model):
    _, _, _, state, _ = web.transcribe_upload(
        str(fake_model), "https://youtu.be/x", "", 0, "auto", 4, "auto", 60, 0, progress=no_progress
    )
    assert fake_download == [] and state["stem"] == "My_Song_live"


@pytest.mark.parametrize("url", ["", "   ", "not a link"])
def test_needs_upload_or_link(url):
    with pytest.raises(gr.Error):
        web.transcribe_upload(None, url, "", 0, "auto", 4, "auto", 60, 0, progress=no_progress)


def test_failed_download_cleans_up(fake_model, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("could not download: Video unavailable")

    monkeypatch.setattr(web, "download_audio", broken)
    with pytest.raises(gr.Error, match="unavailable"):
        web.transcribe_upload(None, "https://youtu.be/x", "", 0, "auto", 4, "auto", 60, 0, progress=no_progress)
    assert list(web.JOBS_DIR.iterdir()) == []
