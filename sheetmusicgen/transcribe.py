"""Audio -> note events, using ByteDance's high-resolution piano transcription model."""

from __future__ import annotations

import functools
import math
import os
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

SAMPLE_RATE = 16000
CHECKPOINT_URL = (
    "https://zenodo.org/records/4034264/files/"
    "CRNN_note_F1=0.9677_pedal_F1=0.9186.pth"
)
CHECKPOINT_PATH = (
    Path.home() / "piano_transcription_inference_data" / "note_F1=0.9677_pedal_F1=0.9186.pth"
)
CHECKPOINT_SIZE = 171_966_578


@dataclass
class Note:
    onset: float  # seconds
    offset: float  # seconds
    pitch: int  # MIDI note number
    velocity: int  # 0-127


Pedal = tuple[float, float]  # sustain pedal (down, up) in seconds

GPU_BATCH = 16  # model segments per forward call on MPS/CUDA; larger gains nothing on an M-series GPU
BEAT_MODEL = "final0"  # beat_this checkpoint (downloads ~78 MB to the torch hub cache on first use)


def load_audio(path: str | os.PathLike) -> np.ndarray:
    """Decode any ffmpeg-readable file (mp3, wav, m4a, ...) to mono 16 kHz float32."""
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
        "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-",
    ]
    try:
        raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    except FileNotFoundError:
        raise RuntimeError("ffmpeg is required to decode audio (brew install ffmpeg)") from None
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg could not decode {path}: {e.stderr.decode().strip()}") from None
    return np.frombuffer(raw, dtype=np.float32).copy()


def ensure_checkpoint() -> Path:
    """Download the model weights (~165 MB) on first use."""
    if CHECKPOINT_PATH.exists() and CHECKPOINT_PATH.stat().st_size == CHECKPOINT_SIZE:
        return CHECKPOINT_PATH
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading piano transcription model (~165 MB) to {CHECKPOINT_PATH} ...")
    tmp = CHECKPOINT_PATH.with_suffix(".part")
    urllib.request.urlretrieve(CHECKPOINT_URL, tmp)
    tmp.rename(CHECKPOINT_PATH)
    return CHECKPOINT_PATH


def pick_device(device: str = "auto") -> str:
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@functools.cache
def load_model(device: str = "cpu"):
    """Load the transcription model once per device (a long-running server reuses it)."""
    import torch
    from piano_transcription_inference import PianoTranscription

    checkpoint = ensure_checkpoint()
    # The library only moves the model for CUDA; load on CPU and move it ourselves.
    transcriptor = PianoTranscription(checkpoint_path=str(checkpoint), device=torch.device("cpu"))
    if device != "cpu":
        transcriptor.model.to(torch.device(device))
    transcriptor.model.eval()
    return transcriptor


def transcribe(
    audio: np.ndarray,
    device: str = "cpu",
    midi_path: str | None = None,
    on_segment: Callable[[int, int], None] | None = None,
) -> tuple[list[Note], list[Pedal]]:
    """Run the piano model over the audio and return the detected notes and sustain pedal.

    `on_segment(done, total)` is called after each batch of model segments, for progress reporting.
    """
    import torch
    from piano_transcription_inference.utilities import RegressionPostProcessor, write_events_to_midi

    transcriptor = load_model(device)
    # Same steps as PianoTranscription.transcribe, but the library feeds the
    # model one segment at a time; batching them is ~10x faster on a GPU (same notes).
    n = transcriptor.segment_samples
    padded = np.concatenate([audio, np.zeros(math.ceil(len(audio) / n) * n - len(audio), np.float32)])
    segments = transcriptor.enframe(padded[None], n)  # half-overlapping, (N, segment_samples)
    total = len(segments)
    batch = 1 if device == "cpu" else GPU_BATCH  # batching only slows the CPU down
    outputs: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for start in range(0, total, batch):
            out = transcriptor.model(torch.from_numpy(segments[start : start + batch]).to(device))
            for key, value in out.items():
                outputs.setdefault(key, []).append(value.cpu().numpy())
            if on_segment:
                on_segment(min(start + batch, total), total)
    output_dict = {key: transcriptor.deframe(np.concatenate(v))[: len(audio)] for key, v in outputs.items()}

    post = RegressionPostProcessor(
        transcriptor.frames_per_second,
        classes_num=transcriptor.classes_num,
        onset_threshold=transcriptor.onset_threshold,
        offset_threshold=transcriptor.offset_threshod,  # sic, the library's spelling
        frame_threshold=transcriptor.frame_threshold,
        pedal_offset_threshold=transcriptor.pedal_offset_threshold,
    )
    note_events, pedal_events = post.output_dict_to_midi_events(output_dict)
    if midi_path:
        write_events_to_midi(0, note_events, pedal_events, midi_path)

    notes = [
        Note(float(e["onset_time"]), float(e["offset_time"]), int(e["midi_note"]), int(e["velocity"]))
        for e in note_events
    ]
    notes.sort(key=lambda n: (n.onset, n.pitch))
    pedals = [(float(e["onset_time"]), float(e["offset_time"])) for e in pedal_events]
    return notes, sorted(pedals)


@functools.cache
def load_beat_model(checkpoint: str = BEAT_MODEL):
    from beat_this.inference import Audio2Frames

    # The model is small; on Apple Silicon the CPU is faster than MPS for it.
    return Audio2Frames(checkpoint_path=checkpoint, device="cpu")


BEAT_FPS = 50  # frame rate of the beat model's output


def beat_activations(audio: np.ndarray, checkpoint: str = BEAT_MODEL) -> tuple[np.ndarray, np.ndarray]:
    """Framewise beat and downbeat probabilities (BEAT_FPS per second) from the beat_this model."""
    import torch

    beat, downbeat = load_beat_model(checkpoint)(audio, SAMPLE_RATE)
    return torch.sigmoid(beat).numpy(), torch.sigmoid(downbeat).numpy()


def track_audio_beats(audio: np.ndarray, checkpoint: str = BEAT_MODEL) -> tuple[list[float], list[float]]:
    """Beats (seconds) of the recording, and how likely each is to start a bar.

    A model trained on annotated music (beat_this) copes with rubato and
    finds the bar far better than tracking the transcribed onsets does.
    """
    from .rhythm import decode_beats

    return decode_beats(*beat_activations(audio, checkpoint), BEAT_FPS)
