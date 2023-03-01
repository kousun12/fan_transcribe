import time
import modal
import json
from pathlib import Path
import re
from typing import Iterator, Tuple, NamedTuple, Dict, Any
from logger import log
from from_url import cache_file
import os
import requests

from transcribe_args import args, all_models, WhisperModel, TranscribeConfig, identifier

CACHE_DIR = "/cache"
TRANSCRIPTIONS_DIR = Path(CACHE_DIR, "transcriptions")
URL_DOWNLOADS_DIR = Path(CACHE_DIR, "url_downloads")
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
silence_end_re = re.compile(
    r" silence_end: (?P<end>[0-9]+(\.?[0-9]*)) \| silence_duration: (?P<dur>[0-9]+(\.?[0-9]*))"
)


class RunningJob(NamedTuple):
    model: str
    start_time: int
    source: str


MAX_JOB_AGE_SECS = 10 * 60


def create_mounts():
    fname = args.filename
    if not fname:
        return []
    name = Path(fname).name if fname else ""
    return [modal.Mount.from_local_file(fname, remote_path=RAW_AUDIO_DIR / name)]


if stub.is_inside():
    mounts = []
    gpu = None
else:
    mounts = create_mounts()
    gpu = args.gpu


@stub.function(
    mounts=mounts,
    image=app_image,
    shared_volumes={CACHE_DIR: volume},
)
def split_silences(
    filepath: str, min_segment_len, min_silence_len
) -> Iterator[Tuple[float, float]]:
    import ffmpeg

    metadata = ffmpeg.probe(filepath)
    duration = float(metadata["format"]["duration"])

    reader = (
        ffmpeg.input(filepath)
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
    log.info(f"Split {filepath} into {num_segments} segments")


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
    filepath: Path,
    model: WhisperModel,
):
    import tempfile
    import time
    import ffmpeg
    import torch
    import whisper

    t0 = time.time()

    with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
        (
            ffmpeg.input(str(filepath))
            .filter("atrim", start=start, end=end)
            .output(f.name)
            .overwrite_output()
            .run(quiet=True)
        )

        use_gpu = torch.cuda.is_available()
        device = "cuda" if use_gpu else "cpu"
        model = whisper.load_model(
            model.name, device=device, download_root=str(MODEL_DIR)
        )
        transcription = model.transcribe(
            f.name, language="en", fp16=use_gpu, temperature=0.0
        )

    t1 = time.time()
    log.info(
        f"Transcribed segment [{int(start)}, {int(end)}] len={end - start:.1f}s in {t1 - t0:.1f}s on {device}"
    )

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
    timeout=60 * 10,  # 10 minutes
)
def fan_out_work(result_path: Path, model: WhisperModel, cfg: TranscribeConfig):
    job_source, job_id = identifier(cfg)

    if cfg.url:
        filepath = URL_DOWNLOADS_DIR / job_id
    else:
        file = Path(cfg.filename)
        filepath = RAW_AUDIO_DIR / file.name

    segment_gen = split_silences.call(
        str(filepath), cfg.min_segment_len, cfg.min_silence_len
    )
    full_text = ""
    output_segments = []
    for transcript, s_time in transcribe_segment.starmap(
        segment_gen, kwargs=dict(filepath=filepath, model=model)
    ):
        full_text += transcript["text"]
        output_segments += transcript["segments"]

    transcript = {
        "full_text": full_text,
        "segments": output_segments,
        "model": model.name,
    }
    with open(result_path, "w") as f:
        json.dump(transcript, f, indent=2)
    log.info(f"Wrote transcription to remote volume: {result_path}")

    return transcript


@stub.function(
    image=app_image,
    shared_volumes={CACHE_DIR: volume},
    timeout=60 * 10,  # 10 minutes
)
def start_transcribe(cfg: TranscribeConfig, notify=None):
    import whisper
    from modal import container_app

    model_name = cfg.model
    force = cfg.force or False

    job_source, job_id = identifier(cfg)
    log.info(f"Starting job {job_id}, source: {job_source}")
    # cache the model in the shared volume
    model = all_models[model_name]

    # noinspection PyProtectedMember
    whisper._download(whisper._MODELS[model.name], str(MODEL_DIR), False)

    TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    URL_DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Using model '{model.name}' with {model.params} parameters.")

    result_path = TRANSCRIPTIONS_DIR / f"{job_id}.json"
    if result_path.exists() and not force:
        log.info(f"Transcription already exists for {job_id}, returning from cache.")
        with open(result_path, "r") as f:
            result = json.load(f)
            if notify:
                notify_webhook(result, notify)
            return result
    else:
        container_app.running_jobs[job_id] = RunningJob(
            model=model.name, start_time=int(time.time()), source=job_source
        )
        if cfg.url:
            cache_file(cfg.url, URL_DOWNLOADS_DIR / job_id)
        try:
            result = fan_out_work.call(result_path=result_path, model=model, cfg=cfg)
            if notify:
                notify_webhook(result, notify)
            return result
        except Exception as e:
            log.error(e)
        finally:
            del container_app.running_jobs[job_id]
            if cfg.url:
                log.info(f"Cleaning up cache: {URL_DOWNLOADS_DIR / job_id}")
                os.remove(URL_DOWNLOADS_DIR / job_id)


def notify_webhook(result, notify):
    # todo add a signature, signed with the secret key
    requests.post(
        notify["url"],
        json={"data": result, "metadata": notify["metadata"] or {}},
    )


class FanTranscriber:
    @staticmethod
    def run(overrides: dict = None):
        cfg = args.merge(overrides) if overrides else args
        if stub.is_inside():
            return start_transcribe.call(cfg=cfg)
        else:
            with stub.run():
                return start_transcribe.call(cfg=cfg)

    @staticmethod
    def queue(url: str, overrides: dict = None, metadata: dict = None):
        cfg = args.merge(overrides) if overrides else args
        notify = {"url": url, "metadata": metadata}
        if stub.is_inside():
            return start_transcribe.spawn(cfg=cfg, notify=notify)
        else:
            with stub.run():
                return start_transcribe.spawn(cfg=cfg, notify=notify)
