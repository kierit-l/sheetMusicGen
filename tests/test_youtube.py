from pathlib import Path

import pytest
import yt_dlp

from sheetmusicgen.youtube import download_audio, is_url


@pytest.mark.parametrize(
    "s, expected",
    [
        ("https://www.youtube.com/watch?v=abc", True),
        ("  http://youtu.be/abc", True),
        ("song.mp3", False),
        ("/Users/me/https.mp3", False),
    ],
)
def test_is_url(s, expected):
    assert is_url(s) is expected


class FakeYDL:
    """Stands in for yt_dlp.YoutubeDL; `info` is what the extractor would report."""

    info: dict = {}
    downloaded: list = []

    def __init__(self, params):
        self.params = params

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download):
        if self.info.get("error"):
            raise yt_dlp.utils.DownloadError("ERROR: " + self.info["error"])
        return dict(self.info)

    def process_ie_result(self, info, download):
        path = Path(self.params["outtmpl"].replace("%(id)s", info["id"]).replace("%(ext)s", "webm"))
        path.write_bytes(b"audio")
        FakeYDL.downloaded.append(path)
        return {**info, "requested_downloads": [{"filepath": str(path)}]}


@pytest.fixture
def ydl(monkeypatch):
    FakeYDL.info = {"id": "abc", "title": "Clair de lune", "duration": 300}
    FakeYDL.downloaded = []
    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    return FakeYDL


def test_download(ydl, tmp_path):
    path, title = download_audio("https://youtu.be/abc", tmp_path, max_seconds=600)
    assert path == tmp_path / "abc.webm" and path.exists()
    assert title == "Clair de lune"


def test_too_long_is_not_downloaded(ydl, tmp_path):
    with pytest.raises(ValueError, match="limit"):
        download_audio("https://youtu.be/abc", tmp_path, max_seconds=60)
    assert ydl.downloaded == []


def test_rejects_playlists_and_live(ydl, tmp_path):
    ydl.info = {"_type": "playlist", "id": "pl"}
    with pytest.raises(ValueError, match="playlist"):
        download_audio("https://youtube.com/playlist?list=pl", tmp_path)
    ydl.info = {"id": "abc", "is_live": True}
    with pytest.raises(ValueError, match="live"):
        download_audio("https://youtu.be/abc", tmp_path)


def test_download_error_becomes_runtime_error(ydl, tmp_path):
    ydl.info = {"error": "Video unavailable"}
    with pytest.raises(RuntimeError, match="could not download .*: Video unavailable"):
        download_audio("https://youtu.be/abc", tmp_path)
