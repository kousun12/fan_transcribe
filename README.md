# fan_transcribe
Fan out audio transcription tasks via [OpenAI Whisper](https://github.com/openai/whisper) and [Modal Labs](https://modal.com/docs/guide).

OpenAI Whisper is a tool for running audio transcriptions, but it's not especially fast to run on a local machine. 

This project provides a command line wrapper around Whisper that allows you to fan out transcription tasks to a cluster of machines provisioned by Modal. Additionally, it also provides an easy way to expose that entrypoint as a web service.

The key idea here is that you can first use `ffmpeg` to break up an audio file by gaps of silence, then you can transcribe each of those chunks in parallel on a managed set of containers. 

Modal's workflow allows you to describe system images, define remote functions, mount shared volumes and execute code across managed runtimes, all from a local python context. Notably, it makes it easy to associate a function invocation to a managed container, allowing us to fan out a bunch of similar compute tasks in duplicate runtimes by simply mapping a set of arguments over a function reference.

## Benchmarks

The following are benchmark results from running the `medium.en` model on a ~1h30m audio file. 

| Environment    | Runtime | Segments | Concurrency     | Cost  | 
|----------------|---------|----------|-----------------|-------|
| M1 Pro laptop  | 9:40:54 | 254      | 1               | n/a   |
| Modal CPU      | 0:03:35 | 254      | ~258 Containers | $2.61 |
| Modal t4 GPU   | 0:07:55 | 254      | ~14 Containers  | $1.24 |
| Modal a10g GPU | 0:03:49 | 254      | ~22 Containers  | $1.49 |
| Modal CPU      | 0:03:28 | 508      | ~345 Containers | $4.09 |

## Usage

You should have a Modal account and have the `modal` CLI installed and set up with a token.

After that, you can use the main `fan_transcribe.py` module from a shell:

```bash
$ python fan_transcribe.py -h                                                

usage: fan_transcribe.py [-h] [-i FILENAME] [-u URL] [-o OUT] [-m MODEL] [-g GPU] [-sg MIN_SEGMENT_LEN] [-sl MIN_SILENCE_LEN] [-f]

options:
  -h, --help            show this help message and exit
  -i FILENAME, --filename FILENAME
                        a local file to transcribe
  -u URL, --url URL     optional remote url of an audio file to transcribe
  -o OUT, --out OUT     optional output directory for transcription results. defaults to ./transcripts/ NB: unless you suffix this arg with .json, it will be
                        interpreted as a directory
  -m MODEL, --model MODEL
                        model to use for transcription. defaults to base.en. model options: [tiny.en, tiny, base.en, base, small.en, small, medium.en, medium, large,
                        large-v1, large-v2]
  -g GPU, --gpu GPU     optional GPU to use for transcription. defaults to None. GPU options: [t4, a100, a100-20g, a10g, any]
  -sg MIN_SEGMENT_LEN, --min_segment_len MIN_SEGMENT_LEN
                        minimum segment length (in seconds) for fan out. defaults to 5.0
  -sl MIN_SILENCE_LEN, --min_silence_len MIN_SILENCE_LEN
                        minimum silence length (in seconds) to split on for segment generation. defaults to 2.0
  -f, --force           re-run a job identifier even if it's already processed
```

Example CLI usage:

```bash
$ python fan_transcribe.py -i test.mp3
$ python fan_transcribe.py -u https://example.com/test.mp3

$ python fan_transcribe.py -i ~/docs/test.mp3 -m base.en -o ./xcribe/
$ python fan_transcribe.py -i test.mp3 --min_segment_len 30 --min_silence_len 2
$ python fan_transcribe.py -i test.mp3 --gpu t4
$ python fan_transcribe.py -i already-processed.mp3 --min_silence_len 3 -f
```

Alternatively, you can use modal's deployment commands to expose a web endpoint:

```bash
$ modal deploy api.py # or modal serve api.py

# then you can make requests to the endpoint:

$ curl -X "POST" "https://<your-endpoint-url>.modal.run" \
     -H 'Content-Type: application/json; charset=utf-8' \
     -d $'{ "url": "https://example.com/test.mp3" }'

# sample result:

{
  "full_text":"Sample transcription",
  "segments":[{"id":0,"seek":0,"start":0.0,"end":3.74,"text":"Sample transcription"}],
  "model":"base.en"
} 
```

