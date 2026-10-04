"""The two pipeline stages, shared by the CLI and the web app.

Transcription is slow (a neural network over the whole recording); notation is
fast. Keeping them separate lets callers re-notate with different options
(tempo, meter, grid, hand split) without transcribing again.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .transcribe import SAMPLE_RATE, Note, Pedal

Progress = Callable[[str], None]


def safe_stem(name: str) -> str:
    """A file-name-safe version of `name` (titles, uploaded file names)."""
    return re.sub(r"[^\w\-]+", "_", unicodedata.normalize("NFC", name)).strip("_")[:80] or "score"


@dataclass
class Transcription:
    notes: list[Note]
    duration: float  # seconds of audio
    raw_midi: Path
    pedals: list[Pedal] = field(default_factory=list)

    def save(self, path: Path) -> None:
        data = {
            "duration": self.duration,
            "raw_midi": str(self.raw_midi),
            "notes": [asdict(n) for n in self.notes],
            "pedals": self.pedals,
        }
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Path) -> Transcription:
        data = json.loads(path.read_text())
        pedals = [tuple(p) for p in data.get("pedals", [])]
        return cls([Note(**n) for n in data["notes"]], data["duration"], Path(data["raw_midi"]), pedals)


@dataclass
class Options:
    title: str | None = None
    bpm: float | None = None
    time_sig: str = "auto"  # "auto", "N/4" or compound "N/8" (3/8, 6/8, 9/8, 12/8)
    grid: int = 4
    split: int = 60
    min_velocity: int = 0
    pdf: bool = True
    pdf_engine: str = "auto"


@dataclass
class Result:
    musicxml: Path
    score_midi: Path
    raw_midi: Path
    pdf: Path | None
    bpm: float  # notated beats per minute (dotted quarters in compound meter)
    time_sig: str  # e.g. "3/4", "6/8"
    key: str  # e.g. "Bb major"
    note_count: int
    pdf_engine: str | None = None
    pdf_error: str | None = None
    outputs: list[Path] = field(default_factory=list)


def transcribe_file(
    audio_path: Path,
    outdir: Path,
    device: str = "auto",
    max_seconds: float | None = None,
    progress: Progress = lambda msg: None,
) -> Transcription:
    """Decode and transcribe `audio_path`; writes `<stem>.transcribed.mid` and `<stem>.notes.json`."""
    from .transcribe import load_audio, pick_device, transcribe

    outdir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem
    progress(f"Loading {audio_path.name}")
    audio = load_audio(audio_path)
    duration = len(audio) / SAMPLE_RATE
    if max_seconds and duration > max_seconds:
        raise ValueError(f"recording is {duration / 60:.1f} min; the limit is {max_seconds / 60:.0f} min")

    device = pick_device(device)
    progress(f"Transcribing {duration:.0f} s of audio on {device}")
    raw_midi = outdir / f"{stem}.transcribed.mid"
    notes, pedals = transcribe(audio, device=device, midi_path=str(raw_midi))
    t = Transcription(notes, duration, raw_midi, pedals)
    t.save(outdir / f"{stem}.notes.json")
    return t


def notate(t: Transcription, outdir: Path, stem: str, opts: Options, progress: Progress = lambda msg: None) -> Result:
    """Quantize, build the score and write MusicXML, MIDI and (optionally) PDF."""
    from .notation import build_score
    from .render import render_pdf
    from .rhythm import beat_division, normalize_beats, quantize, track_beats

    notes = [n for n in t.notes if n.velocity >= opts.min_velocity]
    if not notes:
        raise ValueError("no piano notes were detected")

    progress("Detecting beats and quantizing")
    beats = track_beats(t.duration, notes, opts.bpm)
    division = beat_division(notes, beats)
    if opts.time_sig == "auto":
        meter, compound = None, None
    else:
        num, den = map(int, opts.time_sig.split("/"))
        compound = den == 8
        meter = num // 3 if compound else num
    if opts.bpm:  # a forced tempo is taken to be the notated beat
        compound = division == 3 if compound is None else compound
    else:
        beats, compound = normalize_beats(notes, beats, division, compound)
    rhythm = quantize(notes, beats, subdivisions=opts.grid, meter=meter, compound=compound, pedals=t.pedals)
    title = unicodedata.normalize("NFC", opts.title or stem)
    score, k = build_score(rhythm, title, split=opts.split)

    progress("Writing score")
    outdir.mkdir(parents=True, exist_ok=True)
    xml_path = outdir / f"{stem}.musicxml"
    score_midi = outdir / f"{stem}.score.mid"
    score.write("musicxml", fp=str(xml_path))
    score.write("midi", fp=str(score_midi))

    result = Result(
        musicxml=xml_path,
        score_midi=score_midi,
        raw_midi=t.raw_midi,
        pdf=None,
        bpm=rhythm.bpm,
        time_sig=rhythm.time_sig,
        key=f"{k.tonic.name.replace('-', 'b')} {k.mode}",
        note_count=len(notes),
        outputs=[xml_path, score_midi, t.raw_midi],
    )
    if opts.pdf:
        pdf_path = outdir / f"{stem}.pdf"
        try:
            result.pdf_engine = render_pdf(xml_path, pdf_path, opts.pdf_engine)
            result.pdf = pdf_path
            result.outputs.insert(0, pdf_path)
        except Exception as e:  # PDF is a convenience; MusicXML is the real output
            result.pdf_error = str(e)
    return result
