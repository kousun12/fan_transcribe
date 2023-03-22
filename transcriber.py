import time
import modal
import json
from pathlib import Path
import re
from typing import Iterator, Tuple, NamedTuple
from logger import log
from from_url import cache_file, download_vid_audio
import os
import requests
import base64
from io import BytesIO

from transcribe_args import args, all_models, WhisperModel, TranscribeConfig

CACHE_DIR = "/cache"
TRANSCRIPTIONS_DIR = Path(CACHE_DIR, "transcriptions")
URL_DOWNLOADS_DIR = Path(CACHE_DIR, "url_downloads")
MODEL_DIR = Path(CACHE_DIR, "model")
RAW_AUDIO_DIR = Path("/mounts", "raw_audio")

app_image = (
    modal.Image.debian_slim("3.10.0")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "openai-whisper==20230124",
        "dacite==1.8.0",
        "jiwer==2.5.1",
        "ffmpeg-python==0.2.0",
        "pandas==1.5.3",
        "loguru==0.6.0",
        "torchaudio==0.13.1",
        "openai",
        "git+https://github.com/yt-dlp/yt-dlp.git@master",
    )
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
    if min_segment_len == 0:
        log.info(f"No split {filepath}")
        yield 0, duration
        return
    if duration < min_segment_len:
        min_segment_len = duration
    if duration < min_silence_len:
        min_silence_len = duration

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
        transcriber = whisper.load_model(
            model.name, device=device, download_root=str(MODEL_DIR)
        )
        transcription = transcriber.transcribe(
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
    timeout=60 * 12,
)
def fan_out_work(
    result_path: Path,
    model: WhisperModel,
    cfg: TranscribeConfig,
    file_dir: Path = RAW_AUDIO_DIR,
):
    job_source, job_id = cfg.identifier()

    if cfg.url:
        filepath = URL_DOWNLOADS_DIR / job_id
    elif cfg.video_url:
        filepath = URL_DOWNLOADS_DIR / f"{job_id}.mp3"
    else:
        file = Path(cfg.filename)
        filepath = file_dir / file.name

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
    secrets=[
        modal.Secret.from_name("openai-secret-key"),
        modal.Secret.from_name("openai-org-id"),
    ]
)
def summarize_transcript(text: str):
    import openai

    openai.organization = os.environ["OPENAI_ORGANIZATION_KEY"]
    chunk_size = 14000
    summaries = []
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i : i + chunk_size])
    is_multi = len(chunks) > 1
    for idx, chunk in enumerate(chunks):
        c = (
            f"Summarize the following conversation:\n\n{chunk}"
            if not is_multi
            else f"Summarize the following conversation (part {idx + 1} of {len(chunks)}):\n\n{chunk}"
        )
        messages = [
            {
                "role": "system",
                "content": f"You are a helpful assistant that summarizes {'multi-part ' if is_multi else ''}conversations.",
            },
            {"role": "user", "content": c},
        ]
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.5,
                frequency_penalty=1.0,
                n=1,
            )
            summaries.append(response["choices"][0]["message"]["content"].strip())
        except Exception as e:
            log.info(f"Error: {e}")

    if len(summaries) >= 5:
        summary_text = "\n".join(summaries)
        messages = [
            {
                "role": "system",
                "content": f"You are a helpful assistant that summarizes a conversation.",
            },
            {
                "role": "user",
                "content": f"Condense this conversation summary into bullet points:\n\n{summary_text}",
            },
        ]
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.5,
                frequency_penalty=1.0,
                n=1,
            )
            bullet = response["choices"][0]["message"]["content"].strip()
            summaries.insert(
                0, f"##### Overview:\n\n{bullet}\n\n##### Extended summary:"
            )
        except Exception as e:
            log.info(f"Error: {e}")

    return "\n\n".join(summaries)


@stub.function(
    secrets=[
        modal.Secret.from_name("openai-secret-key"),
        modal.Secret.from_name("openai-org-id"),
    ]
)
def llm_respond(text: str):
    import openai

    openai.organization = os.environ["OPENAI_ORGANIZATION_KEY"]
    messages = [
        {
            "role": "system",
            "content": """Do not use "As an AI language model" in your responses.""",
        },
        {
            "role": "user",
            "content": text,
        },
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.85,
            n=1,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return "I don't know"


@stub.function(
    image=app_image,
    shared_volumes={CACHE_DIR: volume},
    timeout=60 * 12,
)
def start_transcribe(
    cfg: TranscribeConfig,
    notify=None,
    summarize=False,
    byte_string=None,
):
    import whisper
    from modal import container_app

    model_name = cfg.model
    force = cfg.force or False

    job_source, job_id = cfg.identifier()
    log.info(f"Starting job {job_id}, source: {job_source}, args: {cfg}")
    # cache the model in the shared volume
    model = all_models[model_name]

    # noinspection PyProtectedMember
    whisper._download(whisper._MODELS[model.name], str(MODEL_DIR), False)

    TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    URL_DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    if byte_string:
        b = BytesIO(base64.b64decode(byte_string.encode("ISO-8859-1")))
        with open(URL_DOWNLOADS_DIR / cfg.filename, "wb") as file:
            file.write(b.getbuffer())
        log.info(f"Saved bytes to {URL_DOWNLOADS_DIR / cfg.filename}")

    log.info(f"Using model '{model.name}' with {model.params} parameters.")

    result_path = TRANSCRIPTIONS_DIR / f"{job_id}.json"
    use_llm = bool(byte_string)
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
        elif cfg.video_url:
            download_vid_audio(cfg.video_url, URL_DOWNLOADS_DIR / job_id)
        try:
            result = fan_out_work.call(
                result_path=result_path,
                model=model,
                cfg=cfg,
                file_dir=URL_DOWNLOADS_DIR if byte_string else RAW_AUDIO_DIR,
            )
            if summarize:
                summary = summarize_transcript.call(result["full_text"])
                result["summary"] = summary
            if use_llm:
                llm_response = llm_respond.call(result["full_text"])
                result["llm_response"] = llm_response
            if notify:
                notify_webhook(result, notify)
            return result
        except Exception as e:
            log.error(e)
        finally:
            del container_app.running_jobs[job_id]
            if byte_string:
                log.info(f"Cleaning up cache: {URL_DOWNLOADS_DIR / cfg.filename}")
                os.remove(URL_DOWNLOADS_DIR / cfg.filename)
            if cfg.url or cfg.video_url:
                filepath = URL_DOWNLOADS_DIR / (
                    f"{job_id}{'.mp3' if cfg.video_url else ''}"
                )
                log.info(f"Cleaning up cache: {filepath}")
                os.remove(filepath)


def notify_webhook(result, notify):
    # todo add a signature, signed with the secret key
    meta = notify["metadata"] or {}
    log.info(f"Sending notification to {notify['url']}, meta: {meta}")
    requests.post(notify["url"], json={"data": result, "metadata": meta})


class FanTranscriber:
    @staticmethod
    def run(overrides: dict = None, byte_string: str = None):
        log.info(f"Starting fan-out transcriber with overrides: {overrides}")
        cfg = args.merge(overrides) if overrides else args
        if stub.is_inside():
            return start_transcribe.call(cfg=cfg, byte_string=byte_string)
        else:
            with stub.run():
                return start_transcribe.call(cfg=cfg, byte_string=byte_string)

    @staticmethod
    def queue(url: str, cfg: TranscribeConfig, metadata: dict = None, summarize=False):
        notify = {"url": url, "metadata": metadata or {}}
        if stub.is_inside():
            return start_transcribe.spawn(cfg=cfg, notify=notify, summarize=summarize)
        else:
            with stub.run():
                return start_transcribe.spawn(
                    cfg=cfg, notify=notify, summarize=summarize
                )
