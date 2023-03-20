import modal
import os
from pydantic import BaseModel

import time
from transcribe_args import WEB_DEFAULT_ARGS
from transcriber import stub, CACHE_DIR, volume, FanTranscriber
from fastapi import Header
from typing import Union
from logger import log


class APIArgs(BaseModel):
    url: str
    summarize: Union[bool, None] = None
    callback_url: Union[str, None] = None
    byte_string: Union[str, None] = None
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
            cfg=WEB_DEFAULT_ARGS.merge({"url": api_args.url}),
            metadata=api_args.callback_metadata,
            summarize=bool(api_args.summarize),
        )
        return {"call_id": results.object_id}

    log.info(f"Processing {api_args.url}")
    results = FanTranscriber.run(
        {
            "url": api_args.url,
            "filename": f"bytes-{int(time.time())}.mp3"
            if api_args.byte_string
            else None,
        },
        byte_string=api_args.byte_string,
    )
    return results
