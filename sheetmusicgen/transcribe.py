"""Audio -> note events, using ByteDance's high-resolution piano transcription model."""

from __future__ import annotations

import functools
import os
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

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
    audio: np.ndarray, device: str = "cpu", midi_path: str | None = None
) -> tuple[list[Note], list[Pedal]]:
    """Run the piano model over the audio and return the detected notes and sustain pedal."""
    import torch

    transcriptor = load_model(device)
    with torch.no_grad():
        result = transcriptor.transcribe(audio, midi_path)

    notes = [
        Note(float(e["onset_time"]), float(e["offset_time"]), int(e["midi_note"]), int(e["velocity"]))
        for e in result["est_note_events"]
    ]
    notes.sort(key=lambda n: (n.onset, n.pitch))
    pedals = [(float(e["onset_time"]), float(e["offset_time"])) for e in result["est_pedal_events"]]
    return notes, sorted(pedals)
