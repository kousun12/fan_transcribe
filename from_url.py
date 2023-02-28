from typing import NamedTuple
from pathlib import Path
import urllib.request

from logger import log


class DownloadResult(NamedTuple):
    data: bytes
    content_type: str


FAKE_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36"


def download_audio_file(url: str) -> DownloadResult:
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

    result = download_audio_file(url=url)
    size = pretty_size(num=len(result.data))
    log.info(f"Downloaded {size} from {url}")
    with open(destination, "wb") as f:
        f.write(result.data)
    log.info(f"Stored audio at {destination}.")
