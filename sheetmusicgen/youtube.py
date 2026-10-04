"""Fetch the audio track of a YouTube video (or anything else yt-dlp supports)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def is_url(s: str) -> bool:
    return bool(URL_RE.match(s.strip()))


def download_audio(
    url: str,
    outdir: Path,
    max_seconds: float | None = None,
    progress: Callable[[str], None] = lambda msg: None,
) -> tuple[Path, str]:
    """Download the best audio stream of `url` into `outdir`; returns (file, video title).

    The file is left in its original container (webm/m4a); ffmpeg decodes it later.
    Raises ValueError if the video is longer than `max_seconds`, RuntimeError if it
    can't be fetched.
    """
    import yt_dlp

    outdir.mkdir(parents=True, exist_ok=True)
    params = {
        "format": "bestaudio/best",
        "outtmpl": str(outdir / "%(id)s.%(ext)s"),
        "noplaylist": True,  # a watch?v=...&list=... link means just that video
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    progress("Looking up the video")
    try:
        with yt_dlp.YoutubeDL(params) as ydl:
            info = ydl.extract_info(url.strip(), download=False)
            if info.get("_type") == "playlist":
                raise ValueError("that link is a playlist; paste the link of a single video")
            duration = info.get("duration")
            if info.get("is_live"):
                raise ValueError("live streams can't be transcribed")
            if max_seconds and duration and duration > max_seconds:
                raise ValueError(f"video is {duration / 60:.1f} min; the limit is {max_seconds / 60:.0f} min")
            progress(f"Downloading audio of “{info.get('title', url)}”")
            info = ydl.process_ie_result(info, download=True)
    except yt_dlp.utils.DownloadError as e:
        msg = re.sub(r"^ERROR:\s*", "", str(e))
        raise RuntimeError(f"could not download {url}: {msg}") from None

    path = Path(info["requested_downloads"][0]["filepath"])
    return path, info.get("title") or info.get("id") or "video"
