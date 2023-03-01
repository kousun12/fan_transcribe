import modal
import os
from pydantic import BaseModel
from transcriber import stub, CACHE_DIR, volume, FanTranscriber
from fastapi import Header
from typing import Union


class APIArgs(BaseModel):
    url: str
    callback_url: Union[str, None] = None
    callback_metadata: Union[dict, None] = None


@stub.webhook(
    method="POST",
    shared_volumes={CACHE_DIR: volume},
    keep_warm=True,
    secret=modal.Secret.from_name("api-secret-key"),
)
def transcribe(api_args: APIArgs, x_modal_secret: str = Header(default=None)):
    secret = os.environ["API_SECRET_KEY"]
    if secret and x_modal_secret != secret:
        return {"error": "Not authorized"}
    if api_args.callback_url:
        results = FanTranscriber.queue(
            api_args.callback_url,
            overrides={"url": api_args.url},
            metadata=api_args.callback_metadata,
        )
        return {"call_id": results.object_id}

    results = FanTranscriber.run({"url": api_args.url})
    return results
