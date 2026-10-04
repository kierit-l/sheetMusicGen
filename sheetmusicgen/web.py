"""Web UI: upload a recording or paste a YouTube link, get sheet music (Gradio)."""

from __future__ import annotations

import html
import os
import shutil
import tempfile
import time
from pathlib import Path

import gradio as gr

from .pipeline import Options, Result, Transcription, notate, safe_stem, transcribe_file
from .youtube import download_audio, is_url

JOBS_DIR = Path(os.environ.get("SHEETMUSICGEN_JOBS_DIR", Path(tempfile.gettempdir()) / "sheetmusicgen-jobs"))
JOB_TTL = float(os.environ.get("SHEETMUSICGEN_JOB_TTL_HOURS", 6)) * 3600
MAX_MINUTES = float(os.environ.get("SHEETMUSICGEN_MAX_MINUTES", 10))
DEVICE = os.environ.get("SHEETMUSICGEN_DEVICE", "auto")

GRIDS = [("Sixteenths", 4), ("Eighths", 2), ("Triplets", 3), ("Quarters", 1), ("Sextuplets", 6), ("32nds", 8)]
TIME_SIGS = ["auto", "2/4", "3/4", "4/4", "6/4", "3/8", "6/8", "9/8", "12/8"]

INTRO = """\
# sheetmusicgen
Upload a **solo piano** recording (mp3, wav, m4a, flac, ...) or paste a YouTube link and
get an engraved score, editable MusicXML and MIDI. Transcription takes a while; changing
the notation options afterwards and pressing **Re-notate** is quick.
"""


def _cleanup_old_jobs() -> None:
    if not JOBS_DIR.exists():
        return
    cutoff = time.time() - JOB_TTL
    for d in JOBS_DIR.iterdir():
        if d.is_dir() and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)


def _options(title, bpm, time_sig, grid, split, min_velocity) -> Options:
    return Options(
        title=title.strip() or None,
        bpm=float(bpm) if bpm else None,
        time_sig=time_sig,
        grid=int(grid),
        split=int(split),
        min_velocity=int(min_velocity),
    )


def _preview(result: Result) -> str:
    from .render import render_svg_pages

    try:
        pages = render_svg_pages(result.musicxml)
    except Exception as e:
        return f"<p>Preview unavailable ({html.escape(str(e))}); download the files below.</p>"
    body = "".join(f'<div class="smg-page">{svg}</div>' for svg in pages)
    return (
        "<style>.smg-page{background:#fff;border-radius:6px;padding:8px;margin:0 0 12px}"
        ".smg-page svg{width:100%;height:auto}</style>" + body
    )


def _beat_name(result: Result) -> str:
    return "dotted quarters/min" if result.time_sig.endswith("/8") else "bpm"


def _summary(result: Result) -> str:
    lines = [
        f"**{result.note_count}** notes · **~{result.bpm:.0f} {_beat_name(result)}** · "
        f"**{result.time_sig}** · key of **{result.key}**"
    ]
    if result.pdf_error:
        lines.append(f"PDF rendering failed ({result.pdf_error}); the MusicXML opens in MuseScore.")
    return "\n\n".join(lines)


def _notate_job(state: dict, opts: Options, progress: gr.Progress):
    job = Path(state["job"])
    t = Transcription.load(job / f"{state['stem']}.notes.json")
    progress(0.8, desc="Building the score")
    try:
        result = notate(t, job, state["stem"], opts, progress=lambda m: progress(0.9, desc=m))
    except ValueError as e:
        raise gr.Error(str(e)) from None
    job.touch()  # keep the job alive while it is being used
    return _summary(result), _preview(result), [str(p) for p in result.outputs]


def transcribe_upload(audio_path, url, title, bpm, time_sig, grid, split, min_velocity, progress=gr.Progress()):
    url = (url or "").strip()
    if not audio_path and not url:
        raise gr.Error("Upload a recording or paste a YouTube link first.")
    if not audio_path and not is_url(url):
        raise gr.Error("That doesn't look like a link; it should start with https://")
    _cleanup_old_jobs()
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    job = Path(tempfile.mkdtemp(dir=JOBS_DIR))
    if audio_path:
        src = Path(audio_path)
        name = src.stem
        stem = safe_stem(name)
        audio = job / f"{stem}{src.suffix.lower()}"
        shutil.copy(src, audio)
    else:
        progress(0.02, desc="Downloading audio")
        try:
            src, name = download_audio(
                url, job, max_seconds=MAX_MINUTES * 60, progress=lambda m: progress(0.02, desc=m)
            )
        except (RuntimeError, ValueError) as e:
            shutil.rmtree(job, ignore_errors=True)
            raise gr.Error(str(e)) from None
        stem = safe_stem(name)
        audio = src.rename(job / f"{stem}{src.suffix.lower()}")

    progress(0.05, desc="Loading audio")
    try:
        transcribe_file(
            audio,
            job,
            device=DEVICE,
            max_seconds=MAX_MINUTES * 60,
            progress=lambda m: progress(0.1, desc=m + " (this is the slow part)"),
        )
    except (RuntimeError, ValueError) as e:
        shutil.rmtree(job, ignore_errors=True)
        raise gr.Error(str(e)) from None
    audio.unlink()

    state = {"job": str(job), "stem": stem}
    opts = _options(title or name, bpm, time_sig, grid, split, min_velocity)
    return *_notate_job(state, opts, progress), state, gr.update(interactive=True)


def renotate(state, title, bpm, time_sig, grid, split, min_velocity, progress=gr.Progress()):
    if not state or not Path(state["job"]).exists():
        raise gr.Error("No transcription to re-use (it may have expired); transcribe a recording first.")
    return _notate_job(state, _options(title or state["stem"], bpm, time_sig, grid, split, min_velocity), progress)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="sheetmusicgen", delete_cache=(3600, JOB_TTL)) as demo:
        gr.Markdown(INTRO)
        state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                audio = gr.Audio(sources=["upload"], type="filepath", label="Piano recording")
                url = gr.Textbox(label="...or a YouTube link", placeholder="https://www.youtube.com/watch?v=...")
                title = gr.Textbox(label="Title", placeholder="defaults to the file name or video title")
                with gr.Accordion("Notation options", open=False):
                    bpm = gr.Number(label="Tempo (bpm)", value=0, minimum=0, maximum=300, info="0 = detect")
                    time_sig = gr.Dropdown(TIME_SIGS, value="auto", label="Time signature")
                    grid = gr.Dropdown(GRIDS, value=4, label="Rhythmic grid", info="shortest note value")
                    split = gr.Slider(36, 84, value=60, step=1, label="Hand split (MIDI note)", info="60 = middle C")
                    min_velocity = gr.Slider(0, 127, value=0, step=1, label="Drop notes quieter than")
                with gr.Row():
                    go = gr.Button("Transcribe", variant="primary")
                    again = gr.Button("Re-notate", interactive=False)
                gr.Markdown(f"Recordings up to {MAX_MINUTES:.0f} minutes. Mixed music (voice, band) is "
                            "transcribed as if it were all piano.")
            with gr.Column(scale=2):
                summary = gr.Markdown()
                files = gr.File(label="Downloads", file_count="multiple", interactive=False)
                preview = gr.HTML()

        options = [title, bpm, time_sig, grid, split, min_velocity]
        go.click(transcribe_upload, [audio, url, *options], [summary, preview, files, state, again])
        again.click(renotate, [state, *options], [summary, preview, files])
    return demo


def main() -> None:
    from .transcribe import load_model, pick_device

    load_model(pick_device(DEVICE))  # load the weights before the first request
    demo = build_ui()
    demo.queue(default_concurrency_limit=1, max_size=20)  # one transcription at a time
    # Gradio's API and settings panels are decorated with emojis; leave them out.
    demo.launch(max_file_size="100mb", footer_links=["gradio"])


if __name__ == "__main__":
    main()
