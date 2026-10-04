"""Turn a piano recording (mp3/wav/... or a YouTube link) into piano sheet music."""

from __future__ import annotations

import argparse
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
        choices=["auto", "2/4", "3/4", "4/4", "6/4", "3/8", "6/8", "9/8", "12/8"],
        default="auto",
        help="time signature (default: auto, chooses 4/4, 3/4, 3/8 or 6/8)",
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
        default=60,
        help="MIDI note where the right hand starts (default 60 = middle C)",
    )
    ap.add_argument("--min-velocity", type=int, default=0, help="drop quieter notes (0-127)")
    ap.add_argument("--device", default="auto", help="torch device: auto, cpu, mps or cuda")
    ap.add_argument("--pdf-engine", choices=["auto", "verovio", "musescore"], default="auto")
    ap.add_argument("--no-pdf", action="store_true", help="skip PDF rendering")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    from .pipeline import Options, notate, safe_stem, transcribe_file
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
        split=args.split,
        min_velocity=args.min_velocity,
        pdf=not args.no_pdf,
        pdf_engine=args.pdf_engine,
    )
    total = 6 if from_url else 4
    steps = iter(range(1, total + 1))

    def progress(msg: str) -> None:
        print(f"[{next(steps)}/{total}] {msg}")

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
            result = notate(t, outdir, audio.stem, opts, progress=progress)
        except (RuntimeError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1

    beat = "dotted quarters" if result.time_sig.endswith("/8") else "bpm"
    print(f"      ~{result.bpm:.0f} {beat}, {result.time_sig}, key of {result.key}")
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
