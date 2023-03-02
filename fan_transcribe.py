import time
import json
import os
from pathlib import Path
from transcriber import FanTranscriber
from transcribe_args import args

WORKING_DIR = Path(__file__).parent
DEFAULT_OUT = WORKING_DIR / "transcripts"

if __name__ == "__main__":
    t0 = time.perf_counter()
    job_source, job_id = args.identifier()

    result = FanTranscriber.run()

    if args.out:
        parsed_path = Path(args.out)
        is_file = args.out.endswith(".json")
        write_to = parsed_path if is_file else parsed_path / f"{job_id}.json"
    elif args.url:
        write_to = DEFAULT_OUT / f"{job_id}.json"
    else:
        local_file = Path(args.filename)
        write_to = DEFAULT_OUT / f"{local_file.stem}.json"

    os.makedirs(os.path.dirname(write_to), exist_ok=True)

    with open(write_to, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote transcription to local filesystem: {write_to}")
    t1 = time.perf_counter()
    print(f"Total run time: {t1 - t0:.1f} seconds")
