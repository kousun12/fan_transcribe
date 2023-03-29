import argparse
from pathlib import Path
import hashlib
import dataclasses
import sys
from typing import Union, Dict

from modal.gpu import STRING_TO_GPU_CONFIG


@dataclasses.dataclass
class TranscribeConfig:
    filename: Union[str, None]
    video_url: Union[str, None]
    url: Union[str, None]
    out: Union[str, None]
    model: str
    min_segment_len: float
    min_silence_len: float
    force: Union[bool, None] = None
    gpu: Union[str, None] = None
    initial_prompt: Union[str, None] = None

    def merge(self, other: Dict):
        return dataclasses.replace(self, **other)

    def identifier(self):
        possible = [self.filename, self.video_url, self.url]
        only_one = list(map(bool, possible)).count(True) == 1
        if not only_one:
            raise ValueError("Specify exactly one of: filename, video_url, or url")
        if self.filename:
            file = Path(self.filename)
            source = file.name
        elif self.url:
            source = self.url
        elif self.video_url:
            source = self.video_url
        else:
            raise ValueError("Must specify either filename or url")

        return source, hashlib.md5(source.encode("utf-8")).hexdigest()


@dataclasses.dataclass
class WhisperModel:
    name: str
    params: str


all_models = {
    "tiny.en": WhisperModel(name="tiny.en", params="39M"),
    "tiny": WhisperModel(name="tiny", params="39M"),
    "base.en": WhisperModel(name="base.en", params="74M"),
    "base": WhisperModel(name="base", params="74M"),
    "small.en": WhisperModel(name="small.en", params="244M"),
    "small": WhisperModel(name="small", params="244M"),
    "medium.en": WhisperModel(name="medium.en", params="769M"),
    "medium": WhisperModel(name="medium", params="769M"),
    "large": WhisperModel(name="large", params="1550M"),
    "large-v1": WhisperModel(name="large-v1", params="1550M"),
    "large-v2": WhisperModel(name="large-v2", params="1550M"),
}

DEFAULT_MODEL = all_models["base.en"]

DEFAULT_ARGS = TranscribeConfig(
    url=None,
    video_url=None,
    filename=None,
    out=None,
    model=DEFAULT_MODEL.name,
    min_segment_len=5,
    min_silence_len=2,
)

WEB_DEFAULT_ARGS = TranscribeConfig(
    url=None,
    video_url=None,
    filename=None,
    out=None,
    model=all_models["base"].name,
    min_segment_len=3.2,
    min_silence_len=1.5,
)


local_serve = sys.argv[1] == "serve" and sys.argv[2] == "api.py"
from_cli = sys.argv[0] == "fan_transcribe.py"


def cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--filename", help="a local file to transcribe")
    parser.add_argument(
        "-u",
        "--url",
        help=f"optional remote url of an audio file to transcribe",
    )
    parser.add_argument(
        "-v",
        "--video_url",
        help=f"optional remote url of a video to transcribe (supports most video streaming sites)",
    )
    parser.add_argument(
        "-o",
        "--out",
        help="optional output directory for transcription results. defaults to ./transcripts/ NB: unless you suffix this arg with .json, it will be interpreted as a directory",
    )
    parser.add_argument(
        "-m",
        "--model",
        help=f"model to use for transcription. defaults to {DEFAULT_MODEL.name}. model options: [{', '.join(all_models.keys())}]",
        default=DEFAULT_MODEL.name,
    )
    parser.add_argument(
        "-g",
        "--gpu",
        help=f"optional GPU to use for transcription. defaults to None. GPU options: [{', '.join(STRING_TO_GPU_CONFIG.keys())}]",
        default=None,
    )
    parser.add_argument(
        "-sg",
        "--min_segment_len",
        help=f"minimum segment length (in seconds) for fan out. defaults to 5.0",
        default=5.0,
        type=float,
    )
    parser.add_argument(
        "-sl",
        "--min_silence_len",
        help=f"minimum silence length (in seconds) to split on for segment generation. defaults to 2.0",
        default=2.0,
        type=float,
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="re-run a job identifier even if it's already processed",
    )
    return parser.parse_args()


if from_cli:
    args: TranscribeConfig = TranscribeConfig(**vars(cfg()))
elif local_serve:
    args = WEB_DEFAULT_ARGS
else:
    args = DEFAULT_ARGS
