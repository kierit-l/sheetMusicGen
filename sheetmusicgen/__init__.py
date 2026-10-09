"""Turn a piano recording (mp3/wav/... or a YouTube link) into piano sheet music."""

from __future__ import annotations

import argparse
import itertools
import sys
import tempfile
import time
from pathlib import Path


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="sheetmusicgen",
        description="Transcribe a piano recording into sheet music (MusicXML, PDF, MIDI).",
    )
    ap.add_argument("audio", help="input audio file (mp3, wav, m4a, flac, ...) or a YouTube link")
    ap.add_argument("-o", "--outdir", type=Path, help="output directory (default: next to the input, or the current directory for links)")
    ap.add_argument("--title", help="title printed on the score (default: file name or video title)")
    ap.add_argument("--bpm", type=float, help="force a tempo instead of detecting beats")
    ap.add_argument(
        "--time-sig",
        choices=["auto", "2/4", "3/4", "4/4", "2/2", "6/4", "3/8", "6/8", "9/8", "12/8"],
        default="auto",
        help="time signature (default: auto, chooses 2/4, 3/4, 4/4, 2/2, 3/8, 6/8 or 9/8)",
    )
    ap.add_argument(
        "--grid",
        type=int,
        default=4,
        choices=[1, 2, 3, 4, 6, 8],
        help="rhythmic grid, subdivisions per beat: 2=eighths, 3=triplets, 4=sixteenths (default)",
    )
    ap.add_argument(
        "--split",
        type=int,
        help="split hands at this MIDI note (60 = middle C) instead of letting a model assign them",
    )
    ap.add_argument(
        "--note-values",
        choices=["auto", "double", "halve"],
        default="auto",
        help="write every note twice as long or half as long, if the detected beat was off by a factor of two",
    )
    ap.add_argument("--min-velocity", type=int, default=0, help="drop quieter notes (0-127)")
    ap.add_argument(
        "--simplify",
        action="store_true",
        help="write an easy arrangement: the melody over block chords, instead of every note played",
    )
    ap.add_argument(
        "--no-lookup",
        action="store_true",
        help="don't look the piece up in the score library (python -m sheetmusicgen.lookup build) for its beats and bars",
    )
    ap.add_argument("--device", default="auto", help="torch device: auto, cpu, mps or cuda")
    ap.add_argument("--pdf-engine", choices=["auto", "verovio", "musescore"], default="auto")
    ap.add_argument("--no-pdf", action="store_true", help="skip PDF rendering")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    from .lookup import load_index
    from .pipeline import Options, find_reference, notate, safe_stem, transcribe_file
    from .youtube import download_audio, is_url

    from_url = is_url(args.audio)
    if not from_url and not Path(args.audio).exists():
        print(f"error: {args.audio} not found", file=sys.stderr)
        return 1

    opts = Options(
        title=args.title,
        bpm=args.bpm,
        time_sig=args.time_sig,
        grid=args.grid,
        split=args.split or 60,
        hands="split" if args.split else "model",
        note_values=args.note_values,
        min_velocity=args.min_velocity,
        simplify=args.simplify,
        pdf=not args.no_pdf,
        pdf_engine=args.pdf_engine,
    )
    lookup = not args.no_lookup and load_index() is not None
    total = (6 if from_url else 4) + (opts.hands == "model") + 2 * lookup
    steps = itertools.count(1)

    def progress(msg: str) -> None:
        n = next(steps)  # a step may report more than once (beat tracking outlasting transcription)
        print(f"[{n}/{max(n, total)}] {msg}")

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:  # holds the downloaded audio, if any
        try:
            if from_url:
                outdir = args.outdir or Path.cwd()
                src, video_title = download_audio(args.audio, Path(tmp), progress=progress)
                audio = src.rename(src.with_name(safe_stem(video_title) + src.suffix))
                opts.title = opts.title or video_title
            else:
                audio = Path(args.audio)
                outdir = args.outdir or audio.parent
            t = transcribe_file(audio, outdir, device=args.device, progress=progress)
            print(f"      {len(t.notes)} notes detected")
            reference = find_reference(t, args.title or opts.title or audio.stem, progress) if lookup else None
            if lookup and not reference:
                print("      not found in the score library; detecting beats and meter")
            result = notate(t, outdir, audio.stem, opts, progress=progress, reference=reference)
        except (RuntimeError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1

    if result.reference:
        print(f"      beats and bars follow {result.reference}")
    elif reference:
        print(f"      found {reference.describe()}, not used: tempo, meter or note values were forced")
    beat = {"2": "half notes", "8": "dotted quarters"}.get(result.time_sig.split("/")[1], "bpm")
    print(f"      ~{result.bpm:.0f} {beat}, {result.time_sig}, key of {result.key}")
    if result.other_meters:
        print(f"      {' or '.join(result.other_meters)} would fit nearly as well (--time-sig)")
    if result.dropped_notes:
        print(f"      left out {result.dropped_notes} notes cut off from the piece by a long silence")
    if result.pdf_engine:
        print(f"      PDF engraved with {result.pdf_engine}")
    elif result.pdf_error:
        print(f"      PDF rendering failed ({result.pdf_error}); open the MusicXML in MuseScore instead")

    print(f"Done in {time.time() - t0:.0f} s:")
    for p in result.outputs:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
