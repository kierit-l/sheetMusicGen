"""Find the recorded piece in a library of public-domain scores and take its bars from the score.

Beat level and meter are what transcription gets wrong most often, and a
published score settles both. The library is built once from the Mutopia
Project's LilyPond sources (needs `brew install lilypond`):

    git clone --depth 1 https://github.com/MutopiaProject/MutopiaProject /tmp/mutopia
    uv run python -m sheetmusicgen.lookup build /tmp/mutopia

Each solo keyboard piece is rendered to MIDI twice, as printed and with its
repeats unfolded, while a small Scheme translator logs where every bar starts
and its time signature (LilyPond's MIDI has no barlines, so pickups and meter
changes would otherwise be lost).

A recording is matched by its title (composer, catalogue numbers, words)
and by its pitch content, then confirmed by aligning the transcribed notes
to the candidate scores; only a close fit is used. The score's bars, carried
through the alignment onto the recording, become the beats and barlines.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from .transcribe import Note

INDEX_DIR = Path(os.environ.get("SHEETMUSICGEN_SCORES", Path.home() / ".cache" / "sheetmusicgen" / "scores"))
MIN_F1 = 0.5  # note F1 against the aligned score needed to trust a match
MATCH_TOL = 0.15  # seconds; onset distance for a transcribed note to match a score note
TITLE_CANDIDATES = 6  # best title matches aligned
CONTENT_CANDIDATES = 6  # best pitch-content matches aligned
SCREEN_FPS = 8  # chroma frames per second for screening candidates
ALIGN_FPS = 20  # ...and for the final alignment
GAP_SHARE = 3.5  # beat intervals this many times what the score's tempo predicts are gaps, filled with beats (tuned on --tune)

KEYBOARD = re.compile(r"piano|harpsichord|clavichord|keyboard|fortepiano|clavier|virginal|spinet", re.I)
NOT_SOLO = re.compile(
    r"voice|duet|hands|violin|flute|cello|guitar|choir|soprano|organ|orchestra|clarinet|oboe|horn|viola|"
    r"trumpet|recorder|pianos|bass|alto|tenor|satb|ensemble|quartet|trio|strings",
    re.I,
)

BAR_LOGGER = r"""
#(define (smg-bar-log ctx)
  (make-translator
    ((initialize t) (ly:message "SMG-SCORE"))
    ((start-translation-timestep t)
      (let ((pos (ly:context-property ctx 'measurePosition))
            (ts (or (ly:context-property ctx 'timeSignature #f) (ly:context-property ctx 'timeSignatureFraction #f))))
        (if (and (ly:moment? pos) (equal? pos ZERO-MOMENT) (pair? ts))
          (ly:message "SMG-BAR ~a ~a ~a" (ly:moment-main (ly:context-current-moment ctx)) (car ts) (cdr ts)))))))
"""
CONSIST = r"\context { \Score \consists #smg-bar-log }"


# ---------------------------------------------------------------- building


def _blocks(text: str, keyword: str) -> list[tuple[int, int, int]]:
    """(start of `\\keyword`, its opening brace, its closing brace) for each block, skipping comments and strings."""
    out = []
    for m in re.finditer(r"\\" + keyword + r"\s*\{", text):
        if _in_comment(text, m.start()):
            continue
        open_ = m.end() - 1
        close = _closing(text, open_)
        if close is not None:
            out.append((m.start(), open_, close))
    return out


def _in_comment(text: str, pos: int) -> bool:
    line = text[text.rfind("\n", 0, pos) + 1 : pos]
    return "%" in line.replace('\\%', "")


def _closing(text: str, open_: int) -> int | None:
    depth, i, n = 0, open_, len(text)
    while i < n:
        c = text[i]
        if c == "%":
            if text.startswith("%{", i):
                j = text.find("%}", i + 2)
                i = n if j < 0 else j + 2
                continue
            j = text.find("\n", i)
            i = n if j < 0 else j + 1
            continue
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _skip_header(text: str, pos: int) -> int:
    """Position after any `\\header { }` blocks starting at `pos`."""
    while True:
        m = re.match(r"\s*\\header\s*\{", text[pos:])
        if not m:
            return pos
        close = _closing(text, pos + m.end() - 1)
        if close is None:
            return pos
        pos = close + 1


def _instrument(files: list[Path]) -> tuple[Path, str] | None:
    for f in files:
        m = re.search(r'mutopiainstrument\s*=\s*"([^"]*)"', f.read_text(errors="ignore"))
        if m:
            return f, m.group(1)
    return None


def _fields(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in re.findall(r'\b(\w+)\s*=\s*"((?:[^"\\]|\\.)*)"', text):
        out.setdefault(k, v.replace('\\"', '"'))
    return out


def _prepare(src: Path, unfold: bool) -> None:
    """Add the bar logger to every MIDI block (adding MIDI blocks if there are none), and unfold repeats."""
    files = list(src.rglob("*.ly")) + list(src.rglob("*.ily"))
    texts = {f: f.read_text(errors="ignore") for f in files}
    has_midi = any(_blocks(t, "midi") for t in texts.values())
    for f, text in texts.items():
        scores = _blocks(text, "score")
        if not scores and not _blocks(text, "midi"):
            continue
        edits: list[tuple[int, str]] = []  # (position, text to insert)
        for _, open_, close in scores:
            if unfold:
                edits.append((_skip_header(text, open_ + 1), r" \unfoldRepeats "))
            if not has_midi:
                edits.append((close, " \\midi { " + CONSIST + " } "))
        for _, open_, _ in _blocks(text, "midi"):
            edits.append((open_ + 1, " " + CONSIST + " "))
        for pos, ins in sorted(edits, reverse=True):
            text = text[:pos] + ins + text[pos:]
        v = re.search(r'\\version\s*"[^"]*"', text)
        at = v.end() if v else 0
        f.write_text(text[:at] + "\n" + BAR_LOGGER + "\n" + text[at:])


def _render(src: Path, main: Path, out: Path, unfold: bool) -> list[tuple[Path, list[tuple[Fraction, int, int]]]] | None:
    """Render `main` (inside a copy of `src`) to MIDI; per score, the MIDI file and its bars (quarter position, time signature)."""
    work = out / ("unfolded" if unfold else "printed")
    shutil.copytree(src, work)
    _prepare(work, unfold)
    target = work / main.relative_to(src)
    try:
        r = subprocess.run(
            ["lilypond", "-dno-print-pages", "-dmidi-extension=mid", "-o", str(work / "out"), target.name],
            cwd=target.parent, capture_output=True, text=True, timeout=180,
        )
    except subprocess.TimeoutExpired:
        return None
    mids = sorted(work.glob("out*.mid"), key=lambda m: (m.name != "out.mid", len(m.name), m.name))
    scores: list[list[tuple[Fraction, int, int]]] = []
    for line in r.stderr.splitlines():
        if line.startswith("SMG-SCORE"):
            scores.append([])
        elif line.startswith("SMG-BAR") and scores:
            _, pos, num, den = line.split()
            scores[-1].append((Fraction(pos) * 4, int(num), int(den)))
    if not mids or len(mids) != len(scores):
        return None
    for bars in scores:  # the logger misses a bar starting at the very beginning
        while bars and bars[0][0] - Fraction(4 * bars[0][1], bars[0][2]) >= 0:
            q, num, den = bars[0]
            bars.insert(0, (q - Fraction(4 * num, den), num, den))
    return list(zip(mids, scores))


def _pc_profile(notes) -> list[float]:
    h = np.zeros(12)
    for n in notes:
        h[n.pitch % 12] += min(n.end - n.start, 2.0)
    return list(np.round(h / (np.linalg.norm(h) + 1e-9), 4))


def build_one(args: tuple[Path, Path, Path, str]) -> list[dict] | str:
    """Index entries for one Mutopia piece directory."""
    import warnings

    import pretty_midi

    warnings.filterwarnings("ignore", message="Tempo, Key or Time signature")
    piece_dir, main, dest, rel = args
    tmp = Path(tempfile.mkdtemp())
    try:
        src = tmp / "src"
        shutil.copytree(piece_dir, src)
        for f in list(src.rglob("*.ly")) + list(src.rglob("*.ily")):
            subprocess.run(["convert-ly", "-e", str(f)], capture_output=True, timeout=60)
        main_src = src / main.relative_to(piece_dir)
        forms = {}
        for unfold in (False, True):
            got = _render(src, main_src, tmp, unfold)
            if got is None:
                return f"{rel}: render failed ({'unfolded' if unfold else 'printed'})"
            forms["unfolded" if unfold else "printed"] = got
        if len(forms["printed"]) != len(forms["unfolded"]):
            return f"{rel}: {len(forms['printed'])} scores printed, {len(forms['unfolded'])} unfolded"

        text = main.read_text(errors="ignore")
        info = _fields(text)
        for f in sorted(piece_dir.rglob("*.ly")) + sorted(piece_dir.rglob("*.ily")):  # fields set in shared files
            for k, v in _fields(f.read_text(errors="ignore")).items():
                info.setdefault(k, v)
        pieces = [_fields(text[o:c]).get("piece") for _, o, c in _blocks(text, "score")]
        entries = []
        for i in range(len(forms["printed"])):
            entry_id = f"{rel}#{i}" if len(forms["printed"]) > 1 else rel
            stem = entry_id.replace("/", "__").replace("#", "_")
            e = {
                "id": entry_id,
                "title": info.get("mutopiatitle") or info.get("title", ""),
                "piece": pieces[i] if len(pieces) == len(forms["printed"]) and pieces[i] else "",
                "subtitle": info.get("subtitle", ""),
                "opus": info.get("mutopiaopus") or info.get("opus", ""),
                "composer": info.get("composer", ""),
                "composer_id": info.get("mutopiacomposer", ""),
                "instrument": info.get("mutopiainstrument", ""),
                "license": info.get("license", ""),
                "forms": {},
            }
            for form, rendered in forms.items():
                mid, bars = rendered[i]
                pm = pretty_midi.PrettyMIDI(str(mid))
                notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
                if len(notes) < 20 or not bars:
                    break
                path = dest / "midi" / f"{stem}.{form}.mid"
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(mid, path)
                e["forms"][form] = {
                    "midi": str(path.relative_to(dest)),
                    "bars": [[str(q), n, d] for q, n, d in bars],
                    "seconds": round(pm.get_end_time(), 2),
                    "notes": len(notes),
                }
                ks = pm.key_signature_changes
                e["key"] = pretty_midi.key_number_to_key_name(ks[0].key_number) if ks else ""
                e["profile"] = _pc_profile(notes)
            if len(e["forms"]) == 2:
                if e["forms"]["printed"]["notes"] == e["forms"]["unfolded"]["notes"]:
                    del e["forms"]["unfolded"]  # no repeats
                entries.append(e)
        return entries
    except Exception as ex:  # noqa: BLE001  (one odd source shouldn't stop the build)
        return f"{rel}: {type(ex).__name__}: {ex}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def build(mutopia: Path, dest: Path | None = None, only: list[str] | None = None) -> None:
    from multiprocessing import Pool

    dest = dest or INDEX_DIR
    ftp = mutopia / "ftp"
    jobs = []
    dirs = {f.parent.parent if f.parent.name.endswith("-lys") else f.parent for f in ftp.rglob("*.ly")}
    for d in sorted(dirs):
        files = sorted(d.glob("*.ly")) + sorted(d.glob("*-lys/**/*.ly"))
        found = _instrument(files)
        if not found or not KEYBOARD.search(found[1]) or NOT_SOLO.search(found[1]):
            continue
        rel = str(d.relative_to(ftp))
        if only and not any(o.lower() in rel.lower() for o in only):
            continue
        # files holding \score blocks: one per movement, paper-size variants (-a4, -let) once
        mains: dict[str, Path] = {}
        for f in files:
            if _blocks(f.read_text(errors="ignore"), "score"):
                mains.setdefault(re.sub(r"-(a4|let|letter)$", "", f.stem), f)
        for stem, main in mains.items():
            jobs.append((d, main, dest, rel if len(mains) == 1 else f"{rel}/{stem}"))

    dest.mkdir(parents=True, exist_ok=True)
    entries, errors = [], []
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for k, res in enumerate(pool.imap_unordered(build_one, jobs), 1):
            if isinstance(res, str):
                errors.append(res)
            else:
                entries.extend(res)
            print(f"\r{k}/{len(jobs)} pieces, {len(entries)} scores, {len(errors)} failed", end="", flush=True)
    print()
    old = []
    if only and (dest / "index.json").exists():
        ids = {e["id"] for e in entries}
        old = [e for e in json.loads((dest / "index.json").read_text())["entries"] if e["id"] not in ids]
    entries = sorted(old + entries, key=lambda e: e["id"])
    (dest / "index.json").write_text(json.dumps({"source": "Mutopia Project", "entries": entries}))
    (dest / "errors.txt").write_text("\n".join(sorted(errors)))
    print(f"wrote {len(entries)} scores to {dest / 'index.json'} ({len(errors)} pieces failed, see errors.txt)")



# ---------------------------------------------------------------- matching


@dataclass
class Reference:
    """A library score matched to a recording, its beats and bars carried onto the recording's time."""

    id: str
    title: str
    composer: str
    form: str  # "printed" or "unfolded" (repeats played out)
    f1: float  # transcribed notes vs the aligned score
    time_sig: str  # the score's main time signature, as for Options.time_sig
    beats: list[float] = field(default_factory=list)  # seconds, in the notated beat unit
    downbeats: list[float] = field(default_factory=list)  # seconds where the score's bars start
    gaps: list[tuple[float, float]] = field(default_factory=list)  # seconds; stretches the score didn't follow

    def describe(self) -> str:
        composer = re.sub(r"\s*\([^)]*\d{4}[^)]*\)", "", self.composer)  # life dates
        name = ": ".join(x for x in (composer, self.title) if x)
        how = "repeats played" if self.form == "unfolded" else "as printed"
        return f"{name} (Mutopia Project score, {how}, {self.f1:.0%} of notes aligned)"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Reference:
        return cls(**{**d, "gaps": [tuple(g) for g in d.get("gaps", [])]})


_INDEX: dict[Path, tuple[list[dict], dict[str, float], set[str]]] = {}


def _words(s: str) -> list[str]:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.findall(r"[a-z]+|\d+", s)


def _surname(e: dict) -> set[str]:
    out = set(_words(e["composer"])[-1:])
    m = re.match(r"[A-Z][a-z]+", e["composer_id"])
    if m:
        out.add(m.group(0).lower())
    return out


def _entry_words(e: dict) -> set[str]:
    path = re.sub(r"([a-z])([A-Z0-9])|([0-9])([A-Za-z])", r"\1\3 \2\4", e["id"])
    return set(_words(" ".join([e["title"], e["piece"], e["subtitle"], e["opus"], e["composer"], path]))) | _surname(e)


def load_index(path: Path | None = None) -> tuple[list[dict], dict[str, float], set[str]] | None:
    """Index entries, the inverse document frequency of each word, and the composers' surnames."""
    path = path or INDEX_DIR
    if path not in _INDEX:
        f = path / "index.json"
        if not f.exists():
            return None
        entries = json.loads(f.read_text())["entries"]
        df: dict[str, int] = {}
        for e in entries:
            e["_words"] = _entry_words(e)
            for w in e["_words"]:
                df[w] = df.get(w, 0) + 1
        idf = {w: float(np.log(len(entries) / c)) for w, c in df.items()}
        composers = set().union(*(_surname(e) for e in entries)) - {""}
        _INDEX[path] = (entries, idf, composers)
    return _INDEX[path]


def title_scores(title: str, entries: list[dict], idf: dict[str, float], composers: set[str]) -> np.ndarray:
    """How well `title` names each entry: shared words weighted by rarity, numbers double.

    A composer named in the title rules out the other composers' pieces.
    """
    words = set(_words(title))
    named = words & composers
    out = np.zeros(len(entries))
    for i, e in enumerate(entries):
        if named and not named & _surname(e):
            out[i] = -1.0
            continue
        out[i] = sum(idf.get(w, 0.0) * (2.0 if w.isdigit() else 1.0) for w in words & e["_words"])
    return out


def _chroma(onsets: np.ndarray, offsets: np.ndarray, pitches: np.ndarray, fps: float, length: float) -> np.ndarray:
    c = np.zeros((12, int(np.ceil(length * fps)) + 2))
    for a, b, p in zip((onsets * fps).astype(int), (offsets * fps).astype(int), pitches % 12):
        c[p, a : max(a + 1, b)] += 1.0
        c[p, a] += 2.0  # onsets carry most of the timing information
    return c / (np.linalg.norm(c, axis=0, keepdims=True) + 1e-3)


@dataclass
class _Score:
    onsets: np.ndarray  # seconds of the score MIDI
    offsets: np.ndarray
    pitches: np.ndarray
    pm: object  # pretty_midi.PrettyMIDI, for quarter -> seconds


def _load_score(path: Path) -> _Score:
    import warnings

    import pretty_midi

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pm = pretty_midi.PrettyMIDI(str(path))
    notes = sorted((n for i in pm.instruments if not i.is_drum for n in i.notes), key=lambda n: n.start)
    return _Score(
        np.array([n.start for n in notes]), np.array([n.end for n in notes]), np.array([n.pitch for n in notes]), pm
    )


def align(score: _Score, notes: list[Note], fps: float) -> tuple[float, object]:
    """Note F1 of `notes` against `score` aligned to them, and a function mapping score seconds to recording seconds.

    Dynamic time warping on chroma gives a coarse path; the matched notes
    (same pitch, onsets within MATCH_TOL after warping) then pin it down.
    """
    import librosa

    on = np.array([n.onset for n in notes])
    off = np.array([n.offset for n in notes])
    pitch = np.array([n.pitch for n in notes])
    # stretch the score onto the recording's span first, so the path starts out diagonal
    s0, s1, p0, p1 = score.onsets[0], score.offsets.max(), on.min(), off.max()
    k = (p1 - p0) / max(s1 - s0, 1e-6)
    s_on, s_off = p0 + (score.onsets - s0) * k, p0 + (score.offsets - s0) * k
    X = _chroma(s_on, s_off, score.pitches, fps, p1 + 1)
    Y = _chroma(on, off, pitch, fps, p1 + 1)
    _, wp = librosa.sequence.dtw(X=X, Y=Y, metric="euclidean")
    wp = wp[::-1]
    xs, first = np.unique(wp[:, 0], return_index=True)
    ys = np.array([wp[wp[:, 0] == x, 1].mean() for x in xs])
    coarse = lambda t: np.interp(np.asarray(t) * fps, xs, ys) / fps  # noqa: E731  stretched score s -> recording s

    warped = coarse(s_on)
    by_pitch: dict[int, list[int]] = {}
    for j, p in enumerate(pitch):
        by_pitch.setdefault(int(p), []).append(j)
    cands = []
    for i, (t, p) in enumerate(zip(warped, score.pitches)):
        for j in by_pitch.get(int(p), []):
            d = abs(on[j] - t)
            if d <= MATCH_TOL:
                cands.append((d, i, j))
    cands.sort()
    used_s, used_r, pairs = set(), set(), []
    for _, i, j in cands:
        if i not in used_s and j not in used_r:
            used_s.add(i)
            used_r.add(j)
            pairs.append((i, j))
    prec, rec = len(pairs) / len(notes), len(pairs) / len(score.onsets)
    f1 = 2 * prec * rec / (prec + rec) if pairs else 0.0

    # Matched onsets, one point per score onset (a chord's notes are played a little apart).
    pts: dict[float, list[float]] = {}
    for i, j in pairs:
        pts.setdefault(round(float(s_on[i]), 4), []).append(float(on[j]))
    xs_f, ys_f = [], []
    for x in sorted(pts):
        y = float(np.median(pts[x]))
        if not ys_f or (x > xs_f[-1] and y > ys_f[-1]):
            xs_f.append(x)
            ys_f.append(y)
    xs_f, ys_f = np.array(xs_f), np.array(ys_f)

    def to_recording(t):
        st = p0 + (np.asarray(t, dtype=float) - s0) * k
        out = coarse(st)
        if len(xs_f) >= 2:
            inside = (st >= xs_f[0]) & (st <= xs_f[-1])
            out = np.where(inside, np.interp(st, xs_f, ys_f), out)
        return out

    return f1, to_recording


def _time_sig(bars: list[list]) -> tuple[str, Fraction] | None:
    """The score's main time signature (by length of music), as the pipeline writes it, and its beat in quarters."""
    total: dict[tuple[int, int], Fraction] = {}
    for (q, n, d), nxt in zip(bars, bars[1:] + [None]):
        length = Fraction(nxt[0]) - Fraction(q) if nxt else Fraction(4 * n, d)
        total[(n, d)] = total.get((n, d), Fraction(0)) + length
    n, d = max(total, key=total.get)
    if d == 16 and n % 3 == 0:
        return f"{n}/8", Fraction(3, 4)  # written in 8ths: every value doubled
    if d == 8 and n % 3 == 0:
        return f"{n}/8", Fraction(3, 2)
    if d in (2, 4):
        return f"{n}/{d}", Fraction(1)  # cut time is counted in quarters too
    return None


def _structure(score: _Score, bars: list[list], to_recording):
    """Time signature, beats, bar starts and unfollowed stretches (see `_fill_gaps`) of the score, in recording seconds."""
    sig = _time_sig(bars)
    if sig is None:
        return None
    time_sig, unit = sig
    starts = [Fraction(q) for q, _, _ in bars]
    lengths = [Fraction(4 * n, d) for _, n, d in bars]
    beats_q: list[Fraction] = []
    b = starts[0] - unit
    while b >= 0:  # a pickup before the first full bar
        beats_q.insert(0, b)
        b -= unit
    for i, (a, length) in enumerate(zip(starts, lengths)):
        z = starts[i + 1] if i + 1 < len(starts) else a + length
        b = a
        while b < z - unit / 4:
            beats_q.append(b)
            b += unit

    to_sec = lambda qs: np.array([score.pm.tick_to_time(int(round(float(q) * score.pm.resolution))) for q in qs])  # noqa: E731
    beat_s = to_sec(beats_q)
    beats = to_recording(beat_s)
    downs = to_recording(to_sec(starts))
    # where the warping path stalls, several beats land on one moment: keep the first
    keep = [0]
    for i in range(1, len(beats)):
        if beats[i] > beats[keep[-1]] + 0.05:
            keep.append(i)
    filled, gaps = _fill_gaps(beats[keep], beat_s[keep])
    downs = [float(x) for x in downs if not any(a < x < b for a, b in gaps)]
    return time_sig, filled, downs, gaps


def _fill_gaps(beats: np.ndarray, score_s: np.ndarray) -> tuple[list[float], list[tuple[float, float]]]:
    """Beats at the performer's tempo where the alignment crawled, and those stretches.

    A performer who takes some repeats but not others fits neither the
    printed nor the unfolded score: the alignment crawls through the music
    played twice, spreading the score's beats far apart. Against the
    score's own tempo marks the performer's tempo is steady enough that
    such stretches stand out. The bars there are left to the accents;
    a repeat being whole bars, they mostly go on counting.
    """
    if len(beats) < 8:
        return [float(x) for x in beats], []
    d, sd = np.diff(beats), np.maximum(np.diff(score_s), 1e-3)
    stretch = d / sd
    typical = float(np.median(stretch))
    out, gaps = [float(beats[0])], []
    for i in range(len(d)):
        if stretch[i] > GAP_SHARE * typical:
            n = max(2, int(round(d[i] / (sd[i] * typical))))
            out.extend(np.linspace(beats[i], beats[i + 1], n + 1)[1:-1].tolist())
            if gaps and abs(gaps[-1][1] - beats[i]) < 1e-9:
                gaps[-1] = (gaps[-1][0], float(beats[i + 1]))
            else:
                gaps.append((float(beats[i]), float(beats[i + 1])))
        out.append(float(beats[i + 1]))
    return out, gaps


def find(notes: list[Note], title: str = "", index: Path | None = None, progress=lambda msg: None) -> Reference | None:
    """The library score `notes` play, if one fits closely (see the module docstring)."""
    index = index or INDEX_DIR
    loaded = load_index(index)
    if not loaded or len(notes) < 50:
        return None
    entries, idf, composers = loaded
    ts = title_scores(title, entries, idf, composers)
    h = np.zeros(12)
    for n in notes:
        h[n.pitch % 12] += min(n.offset - n.onset, 2.0)
    h /= np.linalg.norm(h) + 1e-9
    content = np.array([np.dot(h, e["profile"]) for e in entries])
    content[ts < 0] = -1.0  # another composer's piece
    cands = [int(i) for i in np.argsort(-ts)[:TITLE_CANDIDATES] if ts[i] > 0]
    cands += [int(i) for i in np.argsort(-content)[:CONTENT_CANDIDATES] if content[i] > 0 and i not in cands]

    progress(f"Comparing with {len(cands)} library scores")
    scored = []
    for i in cands:
        for form, f in entries[i]["forms"].items():
            score = _load_score(index / f["midi"])
            ratio = (notes[-1].onset - notes[0].onset) / max(score.onsets[-1] - score.onsets[0], 1e-6)
            if not 0.2 < ratio < 5 or not 0.3 < len(notes) / f["notes"] < 3:
                continue  # far too many or too few notes to be this score
            f1, _ = align(score, notes, SCREEN_FPS)
            scored.append((f1, i, form))
    if not scored:
        return None
    _, i, form = max(scored)
    e = entries[i]
    score = _load_score(index / e["forms"][form]["midi"])
    f1, to_recording = align(score, notes, ALIGN_FPS)
    if f1 < MIN_F1:
        return None
    built = _structure(score, e["forms"][form]["bars"], to_recording)
    if built is None:
        return None
    time_sig, beats, downs, gaps = built
    title_ = " – ".join(x for x in (e["title"], e["piece"]) if x)
    return Reference(e["id"], title_, e["composer"], form, round(f1, 3), time_sig, beats, downs, gaps)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "build":
        build(Path(sys.argv[2]), only=sys.argv[3:] or None)
    elif len(sys.argv) >= 3 and sys.argv[1] == "find":  # find <notes.json> [title]
        from .pipeline import Transcription

        ref = find(Transcription.load(Path(sys.argv[2])).notes, " ".join(sys.argv[3:]), progress=print)
        print(ref.describe() if ref else "no match", ref and ref.time_sig)
    else:
        print(__doc__)
