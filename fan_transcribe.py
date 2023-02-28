import time
import json
import os
from pathlib import Path
from transcriber import FanTranscriber, args

WORKING_DIR = Path(__file__).parent

if __name__ == "__main__":
    t0 = time.perf_counter()
    local_file = Path(args.filename)
    result = FanTranscriber.run()
    out_path = Path(args.out) if args.out else WORKING_DIR / "transcripts"
    write_to = out_path / f"{local_file.stem}.json"
    os.makedirs(os.path.dirname(write_to), exist_ok=True)
    with open(write_to, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote transcription to local filesystem: {write_to}")
    t1 = time.perf_counter()
    print(f"Total run time: {t1 - t0:.2f} seconds")
