"""The two pipeline stages, shared by the CLI and the web app.

Transcription is slow (a neural network over the whole recording); notation is
fast. Keeping them separate lets callers re-notate with different options
(tempo, meter, grid, hand split) without transcribing again.
"""

from __future__ import annotations

import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from .transcribe import SAMPLE_RATE, Note, Pedal

if TYPE_CHECKING:
    from .lookup import Reference

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
    beats: list[float] = field(default_factory=list)  # seconds, tracked on the audio
    downbeat_probs: list[float] = field(default_factory=list)  # per beat: how likely it starts a bar

    def save(self, path: Path) -> None:
        data = {
            "duration": self.duration,
            "raw_midi": str(self.raw_midi),
            "notes": [asdict(n) for n in self.notes],
            "pedals": self.pedals,
            "beats": self.beats,
            "downbeat_probs": self.downbeat_probs,
        }
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Path) -> Transcription:
        data = json.loads(path.read_text())
        pedals = [tuple(p) for p in data.get("pedals", [])]
        return cls(
            [Note(**n) for n in data["notes"]],
            data["duration"],
            Path(data["raw_midi"]),
            pedals,
            data.get("beats", []),
            data.get("downbeat_probs", []),
        )


@dataclass
class Options:
    title: str | None = None
    bpm: float | None = None
    time_sig: str = "auto"  # "auto", "N/4", cut time "N/2" or compound "N/8" (3/8, 6/8, 9/8, 12/8)
    grid: int = 4
    split: int = 60
    note_values: str = "auto"  # "double" or "halve" every note value (the beat was tracked at the wrong level)
    hands: str = "model"  # "model": the PM2S hand-part model decides; "split": the right hand plays from `split` up
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
    dropped_notes: int = 0  # notes left out as unrelated material (an intro or outro)
    pdf_engine: str | None = None
    pdf_error: str | None = None
    outputs: list[Path] = field(default_factory=list)
    beat_times: list[float] = field(default_factory=list)  # seconds of each notated beat
    downbeat_times: list[float] = field(default_factory=list)  # seconds where each bar starts
    other_meters: list[str] = field(default_factory=list)  # time signatures that fit nearly as well
    reference: str | None = None  # the library score whose beats and bars were used


def transcribe_file(
    audio_path: Path,
    outdir: Path,
    device: str = "auto",
    max_seconds: float | None = None,
    progress: Progress = lambda msg: None,
    on_segment: Callable[[int, int], None] | None = None,
) -> Transcription:
    """Decode and transcribe `audio_path`; writes `<stem>.transcribed.mid` and `<stem>.notes.json`.

    `on_segment(done, total)` reports the model's progress through the recording.
    """
    from .transcribe import load_audio, pick_device, track_audio_beats, transcribe

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
    # The beat model runs on the CPU, so it can track beats while the GPU transcribes.
    with ThreadPoolExecutor(max_workers=1) as pool:
        beat_job = pool.submit(track_audio_beats, audio)
        notes, pedals = transcribe(audio, device=device, midi_path=str(raw_midi), on_segment=on_segment)
        if not beat_job.done():
            progress("Tracking beats")
        beats, downbeat_probs = beat_job.result()
    t = Transcription(notes, duration, raw_midi, pedals, beats, downbeat_probs)
    t.save(outdir / f"{stem}.notes.json")
    return t


def find_reference(t: Transcription, title: str = "", progress: Progress = lambda msg: None) -> Reference | None:
    """The library score this recording plays, if the score library is installed and one fits (see lookup.py)."""
    from .lookup import find

    progress("Looking the piece up in the score library")
    return find(t.notes, title, progress=progress)


def notate(
    t: Transcription,
    outdir: Path,
    stem: str,
    opts: Options,
    progress: Progress = lambda msg: None,
    reference: Reference | None = None,
) -> Result:
    """Quantize, build the score and write MusicXML, MIDI and (optionally) PDF.

    A matched library score (`reference`) sets the beats, meter and bars,
    unless tempo, meter or note values are forced.
    """
    from .notation import build_score
    from .render import render_pdf
    from .rhythm import beat_division, beats_are_halves, beats_in_sixths, drop_detached, extend_beats, normalize_beats, quantize, rescale_beats, snap_beats, track_beats

    notes = [n for n in t.notes if n.velocity >= opts.min_velocity]
    if not notes:
        raise ValueError("no piano notes were detected")
    kept = drop_detached(notes)
    dropped = len(notes) - len(kept)
    notes = kept

    progress("Detecting beats and quantizing")
    cut = None
    if opts.time_sig == "auto":
        meter, compound = None, None
    else:
        num, den = map(int, opts.time_sig.split("/"))
        compound, cut = den == 8, den == 2
        meter = num // 3 if compound else 2 * num if cut else num  # in beats: dotted quarters or quarters
    downbeat_probs: list[tuple[float, float]] = []
    audio_beats = None
    use_reference = reference is not None and opts.time_sig == "auto" and not opts.bpm and opts.note_values == "auto"
    if use_reference:
        num, den = map(int, reference.time_sig.split("/"))
        compound, cut = den == 8, den == 2
        meter = num // 3 if compound else 2 * num if cut else num
        last = max([t.duration] + [n.offset for n in notes]) + 1.0
        audio_beats = beats = extend_beats(snap_beats(np.asarray(reference.beats), notes), last)
        if not compound and beats_in_sixths(notes, beats):
            # Sextuplets (or triplet 16ths) can't be written in a simple meter
            # here; its compound equivalent writes them as 16ths: 4/4 -> 12/8.
            compound, cut = True, False  # `meter` counts the same beats, quarters becoming dotted quarters
        # the score's bar starts, as near-certain downbeats among the beats
        reach = 0.25 * float(np.median(np.diff(beats)))
        downs = np.asarray(reference.downbeats)
        near = np.min(np.abs(beats[:, None] - downs[None, :]), axis=1) < reach if len(downs) else np.zeros(len(beats), bool)
        unsure = [any(a <= b <= z for a, z in reference.gaps) for b in beats]
        downbeat_probs = [(float(b), 0.5 if u else 0.95 if d else 0.05) for b, d, u in zip(beats, near, unsure)]
    elif t.beats and len(t.beats) >= 8 and not opts.bpm:
        # Beats tracked on the audio are at the notated level already; use
        # them unless a forced meter divides the beat differently.
        last = max([t.duration] + [n.offset for n in notes]) + 1.0
        audio_beats = extend_beats(snap_beats(np.asarray(t.beats), notes), last)
        division = beat_division(notes, audio_beats)
        if compound is None or compound == (division == 3):
            beats, compound = audio_beats, division == 3
            downbeat_probs = list(zip(t.beats, t.downbeat_probs))
            cut = bool(cut)
            if not compound and cut is False and meter in (None, 4) and beats_are_halves(notes, beats, t.duration):
                # Slow half-note beats: count the quarters, in 2/2 (two half notes a bar).
                beats = np.sort(np.concatenate([beats, (beats[:-1] + beats[1:]) / 2]))
                meter, cut = 4, True
        else:
            audio_beats = None
    if audio_beats is None:
        # A forced tempo is taken to be the notated beat (in cut time, the half note).
        beats = track_beats(t.duration, notes, opts.bpm and (2 * opts.bpm if cut else opts.bpm))
        division = beat_division(notes, beats)
        if opts.bpm:
            compound = division == 3 if compound is None else compound
        else:
            beats, compound, detected_cut = normalize_beats(notes, beats, division, compound)
            cut = detected_cut if cut is None else cut
    if opts.note_values in ("double", "halve") and not compound and not opts.bpm:
        beats, m = rescale_beats(beats, 2 if opts.note_values == "double" else -2, downbeat_probs)
        if opts.note_values == "double":
            # The old beats become half notes: 2 a bar -> 2/2, 3 -> 3/2, 4 -> 4/2.
            meter, cut = meter or 2 * (m or 2), True
        else:
            meter, cut = None if opts.time_sig == "auto" else meter, False
            # Only beats that are still beats can start a bar.
            reach = 0.25 * float(np.median(np.diff(beats)))
            downbeat_probs = [(t_, p) for t_, p in downbeat_probs if np.min(np.abs(beats - t_)) < reach]
    right_hand = None
    if opts.hands == "model":
        from .pm2s import right_hand_probs

        progress("Assigning notes to hands")
        try:
            right_hand = list(right_hand_probs(notes) > 0.5)
        except OSError as e:  # weights not downloaded and no network: split at a fixed note instead
            progress(f"Hand model unavailable ({e}); splitting hands at MIDI note {opts.split}")
    rhythm = quantize(
        notes,
        beats,
        subdivisions=opts.grid,
        meter=meter,
        compound=compound,
        pedals=t.pedals,
        split=opts.split,
        cut=bool(cut),
        downbeat_probs=downbeat_probs,
        right_hand=right_hand,
    )
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
        dropped_notes=dropped,
        outputs=[xml_path, score_midi, t.raw_midi],
        beat_times=rhythm.beat_times,
        downbeat_times=rhythm.downbeat_times,
        other_meters=rhythm.other_meters,
        reference=reference.describe() if use_reference else None,
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
