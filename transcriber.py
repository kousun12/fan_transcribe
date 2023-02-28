import time
import modal
import logging
import json
from pathlib import Path
import re
from typing import Iterator, Tuple, NamedTuple

from transcribe_args import args, all_models, WhisperModel, ScriptArgs

CACHE_DIR = "/cache"
TRANSCRIPTIONS_DIR = Path(CACHE_DIR, "transcriptions")
MODEL_DIR = Path(CACHE_DIR, "model")
RAW_AUDIO_DIR = Path("/mounts", "raw_audio")

app_image = (
    modal.Image.debian_slim()
    .pip_install(
        "openai-whisper==20230124",
        "dacite==1.8.0",
        "jiwer==2.5.1",
        "ffmpeg-python==0.2.0",
        "pandas==1.5.3",
        "loguru==0.6.0",
        "torchaudio==0.13.1",
    )
    .apt_install("ffmpeg")
)

stub = modal.Stub("fan-transcribe", image=app_image)
stub.running_jobs = modal.Dict()
volume = modal.SharedVolume().persist("fan-transcribe-volume")
silence_end_re = re.compile(r" silence_end: (?P<end>[0-9]+(\.?[0-9]*)) \| silence_duration: (?P<dur>[0-9]+(\.?[0-9]*))")


def get_logger(name, level=logging.INFO):
    logr = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(asctime)s: %(name)s  %(message)s"))
    logr.addHandler(handler)
    logr.setLevel(level)
    logr.propagate = False  # Prevent the modal client from double-logging.
    return logr


log = get_logger(__name__)


class InProgressJob(NamedTuple):
    call_id: str
    start_time: int


MAX_JOB_AGE_SECS = 10 * 60


def create_mount():
    fname = args.filename
    name = Path(fname).name if fname else ""
    return modal.Mount.from_local_file(fname, remote_path=RAW_AUDIO_DIR / name)


if stub.is_inside():
    mounts = []
    gpu = None
else:
    mounts = [create_mount()]
    gpu = args.gpu


@stub.function(mounts=mounts, image=app_image)
def split_silences(
    filename: str, min_segment_len, min_silence_len
) -> Iterator[Tuple[float, float]]:
    import ffmpeg

    remote_file = RAW_AUDIO_DIR / filename
    path = str(remote_file)
    metadata = ffmpeg.probe(path)
    duration = float(metadata["format"]["duration"])

    reader = (
        ffmpeg.input(str(path))
        .filter("silencedetect", n="-10dB", d=min_silence_len)
        .output("pipe:", format="null")
        .run_async(pipe_stderr=True)
    )

    cur_start = 0.0
    num_segments = 0

    while True:
        line = reader.stderr.readline().decode("utf-8")
        if not line:
            break
        match = silence_end_re.search(line)
        if match:
            silence_end, silence_dur = match.group("end"), match.group("dur")
            split_at = float(silence_end) - (float(silence_dur) / 2)
            if (split_at - cur_start) < min_segment_len:
                continue
            yield cur_start, split_at
            cur_start = split_at
            num_segments += 1

    # ignore things if they happen after the end
    if duration > cur_start and (duration - cur_start) > min_segment_len:
        yield cur_start, duration
        num_segments += 1
    log.info(f"Split {path} into {num_segments} segments")


@stub.function(
    mounts=mounts,
    image=app_image,
    shared_volumes={CACHE_DIR: volume},
    gpu=gpu,
    cpu=None if gpu else 2,
)
def transcribe_segment(
    start: float,
    end: float,
    filename: str,
    model: WhisperModel,
):
    import tempfile
    import time
    import ffmpeg
    import torch
    import whisper

    t0 = time.time()
    remote_file = RAW_AUDIO_DIR / filename

    with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
        (
            ffmpeg.input(str(remote_file))
            .filter("atrim", start=start, end=end)
            .output(f.name)
            .overwrite_output()
            .run(quiet=True)
        )

        use_gpu = torch.cuda.is_available()
        device = "cuda" if use_gpu else "cpu"
        model = whisper.load_model(model.name, device=device, download_root=str(MODEL_DIR))
        transcription = model.transcribe(f.name, language="en", fp16=use_gpu, temperature=0.0)  # type: ignore

    t1 = time.time()
    log.info(f"Transcribed segment [{int(start)}, {int(end)}] len={end - start:.2f}s in {t1 - t0:.2f}s on {device}")

    # convert back to global time
    for segment in transcription["segments"]:
        segment["start"] += start
        segment["end"] += start
        del segment["tokens"]
        del segment["temperature"]
        del segment["avg_logprob"]
        del segment["compression_ratio"]
        del segment["no_speech_prob"]

    return transcription, start


@stub.function(
    image=app_image,
    shared_volumes={CACHE_DIR: volume},
    timeout=60 * 10, # 10 minutes
)
def transcribe_audio(
    filename: str,
    result_path: Path,
    model: WhisperModel,
    cli_args: ScriptArgs
):
    segment_gen = split_silences.call(filename, cli_args.min_segment_len, cli_args.min_silence_len)
    full_text = ""
    output_segments = []
    for transcript, s_time in transcribe_segment.starmap(segment_gen, kwargs=dict(filename=filename, model=model)):
        full_text += transcript["text"]
        output_segments += transcript["segments"]

    transcript = {"full_text": full_text, "segments": output_segments, "model": model.name}
    with open(result_path, "w") as f:
        json.dump(transcript, f, indent=2)
    log.info(f"Wrote transcription to remote volume: {result_path}")

    return transcript


@stub.function(
    mounts=mounts,
    image=app_image,
    shared_volumes={CACHE_DIR: volume},
    timeout=60 * 10, # 10 minutes
)
def fan_transcribe(cli_args: ScriptArgs):
    import whisper

    file = Path(cli_args.filename)
    model_name = cli_args.model
    force = cli_args.force or False

    log.info(f"Processing {file.name}")
    file_stem = file.stem
    # cache the model in the shared volume
    model = all_models[model_name]

    # noinspection PyProtectedMember
    whisper._download(whisper._MODELS[model.name], str(MODEL_DIR), False)

    TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Using model '{model.name}' with {model.params} parameters.")

    result_path = TRANSCRIPTIONS_DIR / f"{file_stem}.json"
    if result_path.exists() and not force:
        log.info(f"Transcription already exists for {file.name}.")
        log.info("Skipping transcription.")
        with open(result_path, "r") as f:
            result = json.load(f)
            return result
    else:
        return transcribe_audio.call(
            filename=file.name,
            result_path=result_path,
            model=model,
            cli_args=cli_args,
        )


class FanTranscriber:
    @staticmethod
    def run():
        with stub.run():
            return fan_transcribe.call(cli_args=args)

    @staticmethod
    def trigger_job(file: Path, app: modal.App):
        now = int(time.time())
        try:
            running = app.running_jobs[file.name]
            if isinstance(running, InProgressJob) and (now - running.start_time) < MAX_JOB_AGE_SECS:
                existing_call_id = running.call_id
                log.info(f"Found existing, unexpired call ID {existing_call_id}, {file.name}")
                return {"call_id": existing_call_id}
        except KeyError:
            pass
        call = fan_transcribe.spawn(file.name)
        app.running_jobs[file.name] = InProgressJob(call_id=call.object_id, start_time=now)
