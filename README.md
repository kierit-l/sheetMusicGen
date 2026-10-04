# sheetmusicgen

Turn a piano recording (mp3, wav, m4a, flac, ... or a YouTube link) into piano
sheet music.

```
uv run sheetmusicgen song.mp3 -o out/
uv run sheetmusicgen "https://www.youtube.com/watch?v=..." -o out/
```

produces (for links, files are named after the video title and the downloaded
audio is deleted afterwards):

| file | what it is |
| --- | --- |
| `song.pdf` | engraved grand-staff score |
| `song.musicxml` | editable score: open in MuseScore, Sibelius, Finale, Dorico, ... |
| `song.score.mid` | MIDI of the quantized score |
| `song.transcribed.mid` | raw, unquantized transcription (expressive timing, pedal) |

## Web app

```
uv run sheetmusicgen-web     # then open http://127.0.0.1:7860
```

Upload a recording or paste a YouTube link, get a score preview plus
PDF/MusicXML/MIDI downloads.
Changing tempo, meter, grid or hand split and pressing **Re-notate** reuses the
transcription, so it takes about a second.

Environment variables: `SHEETMUSICGEN_MAX_MINUTES` (default 10),
`SHEETMUSICGEN_DEVICE` (`auto`/`cpu`/`mps`/`cuda`), `SHEETMUSICGEN_JOB_TTL_HOURS`
(default 6, how long results are kept), `SHEETMUSICGEN_JOBS_DIR`.

## How it works

0. **Download** the audio track with [yt-dlp](https://github.com/yt-dlp/yt-dlp)
   if the input is a link (other sites yt-dlp supports work too).
1. **Decode** the audio to 16 kHz mono with ffmpeg.
2. **Transcribe** notes (pitch, onset, offset, velocity) and the sustain pedal with ByteDance's
   high-resolution piano transcription model (weights, ~165 MB, download to
   `~/piano_transcription_inference_data/` on first run). Runs on Apple
   Silicon GPU (MPS) or CUDA when available.
3. **Beat-track** on an onset envelope built from the transcribed notes. Check
   whether beats split in two or in three (compound meter: 3/8, 6/8), move the
   pulse to the level a musician would write as the beat, then pick the meter
   (4/4, 3/4, 3/8 or 6/8) and the downbeat from where loud, long and bass
   notes fall, allowing for the tracker slipping a beat.
4. **Quantize** onsets/durations to the beat grid (16ths by default, 32nds
   inside fast runs), which follows tempo changes because it is anchored to
   the tracked beats.
5. **Notate** with music21: detect the key, spell accidentals to fit it, split
   hands at middle C, shorten notes that only rang on because of the pedal,
   build treble + bass staves, write MusicXML.
6. **Engrave** a PDF with Verovio + cairo (or MuseScore if installed).

## Options

```
--bpm 72             force the tempo instead of detecting it (dotted quarters in x/8)
--time-sig 6/8       force the meter: 2/4 3/4 4/4 6/4 3/8 6/8 9/8 12/8
                     (auto picks 4/4, 3/4, 3/8 or 6/8)
--grid 2|3|4|6|8     subdivisions per beat (3 = triplets, 2 = simpler rhythms)
--split 60           MIDI note where the right hand starts (60 = middle C)
--min-velocity 20    drop quiet ghost notes
--title "..."        score title
--device cpu|mps     override the torch device
--pdf-engine musescore   use MuseScore for nicer engraving (if installed)
--no-pdf             only write MusicXML/MIDI
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) plus `brew install ffmpeg cairo`.
`uv sync` installs the Python dependencies.

## Limitations

- Works best on **solo piano** recordings. With other instruments or voice in
  the mix, it'll still transcribe them as if they were piano notes.
- Each staff is written as a single voice: notes starting together form a
  chord that lasts until the next onset in that hand. Held notes under a
  moving melody get shortened.
- YouTube links: playlists and live streams are rejected, and the length limit
  is checked before downloading. YouTube sometimes blocks downloads from cloud
  IPs or after it changes its site; if a link
  fails, `uv lock --upgrade-package yt-dlp` or upload the audio file instead.
- The hand split is a fixed pitch, not real fingering analysis.
- Meter detection only chooses between 4/4, 3/4, 3/8 and 6/8 (6/8 only with
  clear evidence); use `--time-sig` for others.
- Tuplets inside a beat (e.g. triplet 16ths in a 3/8 piece) are snapped to
  the grid; there is no per-beat tuplet detection.
- Treat the output as a strong first draft. Open the MusicXML in MuseScore
  (free) to fix details.

## Development

```
uv run pytest            # fast tests, no model needed
uv run pytest -m slow    # also runs the real transcription model
```

`tests/make_test_audio.py` synthesizes small test recordings with known notes.

## License

MIT, see [LICENSE](LICENSE). The transcription model weights are downloaded
from Zenodo on first run and are not part of this repository.
