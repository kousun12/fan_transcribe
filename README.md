# fan_transcribe
Fan out audio transcription tasks via [OpenAI Whisper](https://github.com/openai/whisper) and [Modal Labs](https://modal.com/docs/guide).

OpenAI Whisper is a tool for running audio transcriptions, but it's not especially fast to run on a local machine. This project is a wrapper around Whisper that allows you to fan out transcription tasks to a cluster of machines provisioned by Modal.

The key ideas here are that you can first use `ffmpeg` to break up an audio file by gaps of silence, then you can transcribe each of those audio chunks in parallel on a managed set of containers. 

Modal's workflow allows you to describe system images, define remote functions, mount shared volumes and execute code across managed runtimes, all from a local python context. Notably, it makes it easy to associate a function invocation to a managed container, allowing us to fan out a bunch of similar compute tasks by simply mapping sets of arguments over a function reference.

## Usage

You should have a Modal account and have the `modal` CLI installed and set up with a token.

After that, you can use the main `fan_transcribe.py` module:

```bash
$ python fan_transcribe.py -h                                                

usage: fan_transcribe.py [-h] [-o OUT] [-m MODEL] [-m_seg MIN_SEGMENT_LEN] [-m_silence MIN_SILENCE_LEN] [-f] filename

positional arguments:
  filename              the local file to transcribe

options:
  -h, --help            show this help message and exit
  -o OUT, --out OUT     optional output directory for transcription results. defaults to ./transcripts/
  -m MODEL, --model MODEL
                        model to use for transcription. defaults to base.en. model options: [tiny.en, base.en, small.en, medium.en, large]
  -sg MIN_SEGMENT_LEN, --min_segment_len MIN_SEGMENT_LEN
                        minimum segment length (in seconds) for fan out. defaults to 5.0
  -sl MIN_SILENCE_LEN, --min_silence_len MIN_SILENCE_LEN
                        minimum silence length (in seconds) to split on for segment generation. defaults to 2.0
  -f, --force           re-run a job identifier even if it's already processed
```


Example usage:

```bash
$ python fan_transcribe.py test.mp3
$ python fan_transcribe.py ~/docs/test.mp3 -m base.en -o ./xcribe/
```
