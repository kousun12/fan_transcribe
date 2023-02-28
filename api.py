from pydantic import BaseModel
from transcriber import stub, CACHE_DIR, volume, FanTranscriber


class APIArgs(BaseModel):
    url: str


@stub.webhook(
    method="POST",
    shared_volumes={CACHE_DIR: volume},
    keep_warm=True,
)
def transcribe(api_args: APIArgs):
    results = FanTranscriber.run({"url": api_args.url})
    return results
