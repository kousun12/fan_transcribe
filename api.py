import modal
import os
from pydantic import BaseModel
from transcriber import stub, CACHE_DIR, volume, FanTranscriber
from fastapi import Header
from typing import Union
import requests


class APIArgs(BaseModel):
    url: str
    callback_url: Union[str, None]


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
    results = FanTranscriber.run({"url": api_args.url})

    if api_args.callback_url:
        requests.post(api_args.callback_url, json={"data": results})
    return results
