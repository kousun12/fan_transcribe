from typing import NamedTuple
from pathlib import Path
import urllib.request

from logger import log


class DownloadResult(NamedTuple):
    data: bytes
    content_type: str


FAKE_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"


def download_file(url: str) -> DownloadResult:
    req = urllib.request.Request(
        url,
        data=None,
        headers={"User-Agent": FAKE_UA},
    )
    with urllib.request.urlopen(req) as response:
        return DownloadResult(
            data=response.read(),
            content_type=response.headers["content-type"],
        )


def pretty_size(num, suffix="B") -> str:
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


def cache_file(url: str, destination: Path, overwrite: bool = False) -> None:
    if destination.exists():
        if overwrite:
            log.info(f"Overwriting file at {destination}")
        else:
            log.info(f"Using cached file at {destination}")
            return

    result = download_file(url=url)
    size = pretty_size(num=len(result.data))
    log.info(f"Downloaded {size} from {url}")
    with open(destination, "wb") as f:
        f.write(result.data)
    log.info(f"Stored audio at {destination}")


def download_vid_audio(
    url: str,
    destination_path: Path,
) -> None:
    import yt_dlp

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
        "outtmpl": f"{destination_path}.%(ext)s",
    }
    log.info(f"Download video from {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(url)
    log.info(f"Saved audio from {url} to {destination_path}")
