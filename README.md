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
Changing tempo, meter, grid, note values or hand split (or ticking **Easy
arrangement**) and pressing **Re-notate** reuses the transcription, so it
takes about a second. When
another meter fits nearly as well, the summary suggests it (the CLI prints it
too).

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
3. **Beat-track** the audio with [beat_this](https://github.com/CPJKU/beat_this)
   (weights, ~78 MB, download on first run; runs while the notes are being
   transcribed). Its beat probabilities are decoded into a beat sequence whose
   tempo may drift but not jump a level, and each beat is moved onto the notes
   played with it. Whether beats split in two or three (6/8, with
   sixteenths too) comes from where the notes fall. Bars come from the
   model's downbeat probabilities: the bar length (2, 3 or 4 beats) whose
   downbeats stand out most (where none does, the notes' accents pick it),
   then a pass that may restart the count where
   the downbeats clearly move, leaving a bar of another length with its own
   time signature (a missed beat no longer shifts every later barline).
   Slow beats that are really half notes (most notes would otherwise be
   32nds) are written in 2/2. Without audio
   beats (old transcriptions, or a forced meter the beats contradict), beats
   are tracked on the transcribed onsets and the meter is picked from where
   loud, long and bass notes fall.
   **Score library (optional):** if the piece is one of the ~700 solo
   keyboard scores of the [Mutopia Project](https://www.mutopiaproject.org)
   (see Setup), it is recognised from the title and the notes, confirmed by
   aligning the transcription to the score, and the score's own beats, meter
   and barlines replace the detected ones. Anything else (or with
   `--no-lookup`, or a forced tempo, meter or note values) is notated from
   the detected beats as above.
4. **Quantize** onsets/durations to the beat grid (16ths by default, 32nds
   inside fast runs), which follows tempo changes because it is anchored to
   the tracked beats. In simple meters each hand's beats are written in
   triplets where they fit them better, so one hand can play two against the
   other's three. A short stretch cut off from the piece by 4+ seconds of
   silence at the start or end (an intro jingle, an outro) is left out.
5. **Notate** with music21: detect the key (a near tie between related keys
   goes to the one the final bass note is the tonic of), spell accidentals to fit it, shorten notes that only rang on because of the pedal,
   build treble + bass staves (changing clef where a hand stays far out of
   its staff's range), write MusicXML.
6. **Engrave** a PDF with Verovio + cairo (or MuseScore if installed).

## Easy arrangement

A faithful transcription writes down everything the pianist played, which
for a piano cover means octave doublings, inner voices, broken-chord
accompaniments and fills. `--simplify` (the web app's **Easy arrangement**)
writes what an easy-piano edition would print instead:

- **Right hand: the melody**, one note at a time. It is the top voice,
  except for notes under a melody note still held down, or a leap of more
  than a fifth below one that just started (a right hand filling the gaps
  in the tune with broken chords). A lone note more than an octave above
  the tune around it is taken for a misheard overtone and left out. Onsets
  round to eighths, or to 16ths where those last at least 0.2 s (slow pieces).
- **Left hand: block chords**, one per bar, or per half bar (2 + 1 beats in
  3/4) where the harmony changes enough to be worth it: the bass note and
  the two strongest other pitch classes of everything that isn't melody,
  skipping tones a step from one already chosen, in close position.

It works on the transcription, so the meter, key and bars are the same as in
the full score.

## Options

```
--bpm 72             force the tempo instead of detecting it (dotted quarters in x/8, halves in 2/2)
--time-sig 6/8       force the meter: 2/4 3/4 4/4 2/2 6/4 3/8 6/8 9/8 12/8
                     (auto picks 2/4, 3/4, 4/4, 2/2, 3/8, 6/8 or 9/8)
--grid 2|3|4|6|8     subdivisions per beat (3 = triplets, 2 = simpler rhythms)
--split 60           split hands at this MIDI note (60 = middle C) instead of using the hand model
--note-values double write every note twice as long (or `halve`), if the beat came out at the wrong level
--min-velocity 20    drop quiet ghost notes
--simplify           easy arrangement: the melody over block chords (see above)
--title "..."        score title
--device cpu|mps     override the torch device
--pdf-engine musescore   use MuseScore for nicer engraving (if installed)
--no-pdf             only write MusicXML/MIDI
--no-lookup          don't use a matching score from the score library
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) plus `brew install ffmpeg cairo`.
`uv sync` installs the Python dependencies.

Score library (optional, ~17 MB, takes about 10 minutes; needs `brew install lilypond`):

```
git clone --depth 1 https://github.com/MutopiaProject/MutopiaProject /tmp/mutopia
uv run python -m sheetmusicgen.lookup build /tmp/mutopia
```

It is written to `~/.cache/sheetmusicgen/scores` (`SHEETMUSICGEN_SCORES`
overrides). Each piece is rendered to MIDI as printed and with repeats
unfolded, with every barline and time signature logged.

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
- Hands are assigned per note by a model (about 9 in 10 notes on the staff
  the score uses); `--split` falls back to a fixed pitch.
- Meter detection chooses between 2/4, 3/4, 4/4, 2/2, 3/8, 6/8 and 9/8
  (12/8 is written as 6/8, 3/8 often as 3/4); the beat level is sometimes off by a
  factor of two, which doubles or halves every note value. Use `--note-values`
  and `--time-sig` (or the web app's Note values and Time signature) to correct it.
- Triplet eighths in simple meters are detected per beat and hand; other
  tuplets (triplet 16ths, quintuplets, duplets in 6/8) are snapped to the grid.
- Heavy rubato and dense irregular tuplets (Chopin ballades, nocturnes) can
  still defeat the rhythm: expect wrong note values in such passages.
- The score library knows each piece as printed and with every repeat
  played. A performance that takes some repeats but not others fits neither
  exactly; the bars may drift in the part played twice. An edition's
  choices win where it is used: Mutopia's Beethoven Op. 78 is in 4/4, for
  example, where other editions write 2/4.
- Treat the output as a strong first draft. Open the MusicXML in MuseScore
  (free) to fix details.

## Development

```
uv run pytest            # fast tests, no model needed
uv run pytest -m slow    # also runs the real transcription model
```

`tests/make_test_audio.py` synthesizes small test recordings with known notes.

### Benchmarks

`bench/` compares the output with reference scores on real performances:

```
uv run python bench/compare.py              # 14 YouTube recordings vs Mutopia scores
git clone --depth 1 https://github.com/fosfrancesco/asap-dataset bench/asap
uv run python bench/maestro_audio.py        # MAESTRO audio of ~150 ASAP pieces (~600 MB)
uv run python bench/asap_bench.py --audio --tune      # tune constants on these only
uv run python bench/asap_bench.py --audio --heldout   # honest score
git clone --depth 1 https://github.com/CPJKU/vienna4x22 bench/v4x22/repo    # + its audio, see v4x22_bench.py
uv run python bench/v4x22_bench.py --transcribe        # 88 recordings, from the audio as in the app
uv run python bench/v4x22_bench.py
```

All three take `--lookup` to use the score library. The YouTube references are
Mutopia scores themselves, so that is an upper bound; ASAP's scores come from
other editions, and its results are split into pieces found in the library
and pieces not found.

The Vienna 4x22 corpus (22 pianists, 4 excerpts, recorded on a
Boesendorfer SE) is new to every model used here and was never tuned on;
it runs the whole app, transcription included, on the recordings.

ASAP pairs MAESTRO performances with their scores and annotated beats,
downbeats and meters. Beats are tracked with the beat_this cross-validation
model that never saw the piece, and `--pm2s-unseen` restricts to the pieces
the hand model never trained on.

#### Results

Held-out ASAP performances (51 pieces, not used for tuning), scored against
ASAP's annotated beats and downbeats:

| metric | onset-only baseline | this pipeline |
| --- | --- | --- |
| beat F1 | 0.51 | 0.74 |
| downbeat F1 | 0.25 | 0.53 |

Note F1 of the final score against the reference is 93%. The onset-only
baseline tracks beats from transcribed note onsets alone; the custom beat,
downbeat and meter logic in `sheetmusicgen/rhythm.py` produces the second
column. Reproduce with `uv run python bench/asap_bench.py --audio --heldout`.

## License

MIT, see [LICENSE](LICENSE). Model weights (transcription, beat_this, PM2S)
are downloaded on first run and are not part of this repository.
`sheetmusicgen/pm2s.py` reproduces PM2S inference code (MIT, Copyright (c)
2022 Lele Liu). The `piano_transcription_inference` package declares no
license; check its terms before distributing.
