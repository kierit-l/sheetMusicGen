"""MusicXML -> engraved PDF via Verovio (SVG) and cairo."""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _expose_homebrew_cairo() -> None:
    # cairocffi finds libcairo through ctypes.util.find_library, which on macOS
    # only searches Homebrew's prefix if it is on DYLD_FALLBACK_LIBRARY_PATH.
    if sys.platform == "darwin":
        extra = [d for d in ("/opt/homebrew/lib", "/usr/local/lib") if os.path.isdir(d)]
        current = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(filter(None, [current, *extra]))


# Verovio writes note symbols inside text (e.g. the tempo mark) with the SMuFL
# web font, which cairo cannot load; swap them for Unicode music symbols.
_SMUFL_TO_UNICODE = {
    "\ueca2": "\U0001D15D",  # metNoteWhole
    "\ueca3": "\U0001D15E",  # metNoteHalfUp
    "\ueca5": "\u2669",  # metNoteQuarterUp
    "\ueca7": "\u266A",  # metNote8thUp
    "\uecb7": ".",  # metAugmentationDot
}
_SYMBOL_FONT = {"darwin": "Apple Symbols", "win32": "Segoe UI Symbol"}.get(sys.platform, "DejaVu Sans")
_SMUFL_TSPAN = re.compile(r'<tspan font-family="Leipzig" font-size="(\d+)px">([^<]*)</tspan>')


def _replace_smufl_text(svg: str) -> str:
    def sub(m: re.Match) -> str:
        text = "".join(_SMUFL_TO_UNICODE.get(c, "") for c in m.group(2))
        size = int(int(m.group(1)) * 0.9)
        return f'<tspan font-family="{_SYMBOL_FONT}" font-size="{size}px">{text}</tspan>'

    return _SMUFL_TSPAN.sub(sub, svg)


def _toolkit():
    import verovio

    # Verovio's resources are per thread and are only found automatically on
    # the main thread; web requests run on worker threads, so set the path.
    tk = verovio.toolkit(False)
    tk.setResourcePath(os.path.join(os.path.dirname(verovio.__file__), "data"))
    return tk


def find_musescore() -> str | None:
    for name in ("mscore", "musescore", "mscore4", "MuseScore4"):
        if path := shutil.which(name):
            return path
    for app in Path("/Applications").glob("MuseScore*.app"):
        exe = app / "Contents" / "MacOS" / "mscore"
        if exe.exists():
            return str(exe)
    return None


def render_pdf(musicxml: Path, pdf: Path, engine: str = "auto") -> str:
    """Write `pdf` from `musicxml`; returns the engine that was used."""
    if engine in ("auto", "musescore") and (mscore := find_musescore()):
        subprocess.run([mscore, "-o", str(pdf), str(musicxml)], check=True, capture_output=True)
        return "musescore"
    if engine == "musescore":
        raise RuntimeError("MuseScore not found")

    _expose_homebrew_cairo()
    import cairosvg
    from pypdf import PdfReader, PdfWriter

    tk = _toolkit()
    tk.setOptions(
        {
            "pageWidth": 2100,  # A4 in tenths of mm
            "pageHeight": 2970,
            "pageMarginTop": 100,
            "pageMarginBottom": 100,
            "pageMarginLeft": 100,
            "pageMarginRight": 100,
            "scale": 45,
            "footer": "none",
            "breaks": "auto",
        }
    )
    if not tk.loadFile(str(musicxml)):
        raise RuntimeError("Verovio could not read the MusicXML")

    writer = PdfWriter()
    for page in range(1, tk.getPageCount() + 1):
        svg = _replace_smufl_text(tk.renderToSVG(page))
        buf = io.BytesIO(cairosvg.svg2pdf(bytestring=svg.encode()))
        writer.append(PdfReader(buf))
    with open(pdf, "wb") as f:
        writer.write(f)
    return "verovio"


def render_svg_pages(musicxml: Path, width: int = 2100) -> list[str]:
    """Engrave `musicxml` as SVG pages for on-screen preview."""
    tk = _toolkit()
    tk.setOptions(
        {
            "pageWidth": width,
            "adjustPageHeight": True,
            "pageMarginTop": 60,
            "pageMarginBottom": 60,
            "pageMarginLeft": 60,
            "pageMarginRight": 60,
            "scale": 40,
            "footer": "none",
            "breaks": "auto",
            "svgViewBox": True,  # scale with the container instead of fixed pixels
        }
    )
    if not tk.loadFile(str(musicxml)):
        raise RuntimeError("Verovio could not read the MusicXML")
    return [_replace_smufl_text(tk.renderToSVG(p)) for p in range(1, tk.getPageCount() + 1)]
