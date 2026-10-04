"""Synthesize piano-like test recordings from known note lists (for sanity checks)."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100


def tone(midi, dur, vel):
    f = 440.0 * 2 ** ((midi - 69) / 12)
    t = np.arange(int(SR * (dur + 0.3))) / SR
    y = sum((0.6 ** h) * np.sin(2 * np.pi * f * (h + 1) * t * (1 + 0.0004 * h)) for h in range(8))
    env = np.exp(-t * (1.5 + f / 400)) * np.minimum(1, t / 0.004)
    env[t > dur] *= np.exp(-(t[t > dur] - dur) * 30)
    return y * env * (vel / 127)


def render(events, bpm, path):
    beat = 60 / bpm
    total = max(s + d for s, d, _, _ in events) * beat + 1.5
    out = np.zeros(int(SR * total))
    for start, dur, pitch, vel in events:
        y = tone(pitch, dur * beat, vel)
        i = int((start * beat + 0.5) * SR)
        out[i : i + len(y)] += y[: len(out) - i]
    out /= np.abs(out).max() * 1.1
    wav = path.with_suffix(".wav")
    sf.write(wav, out, SR)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), str(path)], check=True)
    wav.unlink()


def c_major_4_4():
    # (start beat, length in beats, midi pitch, velocity)
    melody = [(0,1,64),(1,1,62),(2,1,60),(3,1,62),(4,1,64),(5,1,64),(6,2,64),
              (8,1,62),(9,1,62),(10,2,62),(12,1,64),(13,1,67),(14,2,67),
              (16,0.5,64),(16.5,0.5,65),(17,1,64),(18,1,62),(19,1,60),(20,4,60)]
    ev = [(s, d, p, 80) for s, d, p in melody]
    bass = [48, 43, 48, 43, 48, 43, 48, 48]  # I-V alternation
    ch = {48: [48, 52, 55], 43: [43, 50, 53]}
    for bar, root in enumerate(bass[:6]):
        ev.append((bar * 4, 2, root, 95))
        for p in ch[root][1:]:
            ev.append((bar * 4 + 2, 2, p, 60))
    return ev


def waltz_f_3_4():
    mel = [(0,2,72),(2,1,69),(3,2,70),(5,1,67),(6,3,65),(9,1,69),(10,1,70),(11,1,72),
           (12,2,74),(14,1,72),(15,3,72)]
    ev = [(s, d, p, 80) for s, d, p in mel]
    roots = [41, 36, 41, 46, 36, 41]
    chords = {41: [57, 60], 36: [55, 58], 46: [58, 62]}
    for bar, r in enumerate(roots):
        ev.append((bar * 3, 1, r, 100))
        for beat in (1, 2):
            for p in chords[r]:
                ev.append((bar * 3 + beat, 1, p, 55))
    return ev


def fur_elise_3_8():
    """Opening of Für Elise, in sixteenths (render at ~360 'bpm' for a dotted quarter of ~60)."""
    motif = [76, 75, 76, 71, 74, 72]
    ev = [(0, 1, 76, 70), (1, 1, 75, 70)]  # pickup
    bar = 2
    for _ in range(2):
        ev += [(bar + i, 1, p, 70) for i, p in enumerate(motif)]
        for top, bass, rh in [(69, (45, 52, 57), (60, 64, 69)), (71, (40, 52, 56), (64, 68, 71)),
                              (72, (45, 52, 57), (64, 76, 75))]:
            bar += 6
            ev += [(bar, 2, top, 80)] + [(bar + i, 1, p, 60) for i, p in enumerate(bass)]
            ev += [(bar + 3 + i, 1, p, 60) for i, p in enumerate(rh)]
        bar += 6
        ev += [(bar + i, 1, p, 70) for i, p in enumerate(motif)]
        for top, bass, rh in [(69, (45, 52, 57), (60, 64, 69)), (71, (40, 52, 56), (64, 72, 71))]:
            bar += 6
            ev += [(bar, 2, top, 80)] + [(bar + i, 1, p, 60) for i, p in enumerate(bass)]
            ev += [(bar + 3 + i, 1, p, 60) for i, p in enumerate(rh)]
        bar += 6
        ev += [(bar, 4, 69, 80)] + [(bar + i, 1, p, 60) for i, p in enumerate((45, 52, 57))]
        bar += 6
    return ev


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/audio")
    out.mkdir(parents=True, exist_ok=True)
    render(c_major_4_4(), 100, out / "c_major_4_4.mp3")
    render(waltz_f_3_4(), 132, out / "waltz_f_3_4.mp3")
    print("wrote", *out.glob("*.mp3"))
