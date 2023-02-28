import argparse
import dataclasses


@dataclasses.dataclass
class ScriptArgs:
    filename: str
    out: str
    model: str
    min_segment_len: float
    min_silence_len: float
    force: bool


@dataclasses.dataclass
class WhisperModel:
    name: str
    params: str


all_models = {
    "tiny.en": WhisperModel(name="tiny.en", params="39M"),
    "base.en": WhisperModel(name="base.en", params="74M"),
    "small.en": WhisperModel(name="small.en", params="244M"),
    "medium.en": WhisperModel(name="medium.en", params="769M"),
    "large": WhisperModel(name="large", params="1550M"),
}

DEFAULT_MODEL = all_models["base.en"]

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="the local file to transcribe")
parser.add_argument(
    "-o",
    "--out",
    help="optional output directory for transcription results. defaults to ./transcripts/",
)
parser.add_argument(
    "-m",
    "--model",
    help=f"model to use for transcription. defaults to {DEFAULT_MODEL.name}. model options: [{', '.join(all_models.keys())}]",
    default=DEFAULT_MODEL.name,
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
parser.add_argument("-f", "--force", action="store_true", help="re-run a job identifier even if it's already processed")
args = parser.parse_args()
