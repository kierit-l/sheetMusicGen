"""Re-render each Mutopia reference with repeats unfolded (needs `brew install lilypond`).

Writes bench/cache/<id>/reference_unfolded.mid next to the as-printed reference.mid.
"""

import io
import json
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
for p in json.loads((ROOT / "pieces.json").read_text()):
    if sys.argv[1:] and not any(a in p["id"] for a in sys.argv[1:]):
        continue
    d = ROOT / "cache" / p["id"]
    out = d / "reference_unfolded.mid"
    if out.exists():
        continue
    src = d / "ly"
    shutil.rmtree(src, ignore_errors=True)
    src.mkdir(parents=True)
    ref = p["reference"]
    base = ref.removesuffix("-mids.zip").removesuffix(".mid")
    try:
        (src / "main.ly").write_bytes(urllib.request.urlopen(base + ".ly").read())
    except Exception:
        zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(base + "-lys.zip").read())).extractall(src)
    files = list(src.rglob("*.ly")) + list(src.rglob("*.ily"))
    main = [f for f in files if "\\score" in f.read_text(errors="ignore")]
    if p.get("reference_member"):  # multi-movement zip: pick the movement's file
        stem = p["reference_member"].removesuffix(".mid")
        main = [f for f in main if f.stem == stem] or main
    for f in files:
        subprocess.run(["convert-ly", "-e", str(f)], capture_output=True)
        text = f.read_text(errors="ignore")
        text = re.sub(r"\\score\s*\{", r"\\score { \\unfoldRepeats", text)
        text = text.replace("\\unfoldRepeats \\unfoldRepeats", "\\unfoldRepeats")
        f.write_text(text)
    target = main[0]
    r = subprocess.run(
        ["lilypond", "-dno-print-pages", "-dmidi-extension=mid", "-o", str(src / "out"), target.name],
        cwd=target.parent, capture_output=True, text=True,
    )
    # LilyPond names the first score's MIDI out.mid, later ones out-1.mid, ...
    mids = sorted(src.glob("out*.mid"), key=lambda m: (m.name != "out.mid", m.name))
    if not mids:
        print(f"{p['id']}: FAILED\n{r.stderr[-1500:]}")
        continue
    shutil.copy(mids[0], out)
    print(f"{p['id']}: {[m.name for m in mids]}")
