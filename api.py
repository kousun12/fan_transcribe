import modal
import os
from pydantic import BaseModel

import time
from transcribe_args import WEB_DEFAULT_ARGS
from transcriber import stub, FanTranscriber, llm_respond
from fastapi import Header
from typing import Union
from logger import log


class APIArgs(BaseModel):
    url: Union[str, None] = None
    summarize: Union[bool, None] = None
    callback_url: Union[str, None] = None
    byte_string: Union[str, None] = None
    llm_text: Union[str, None] = None
    callback_metadata: Union[dict, None] = None
    initial_prompt: Union[str, None] = None


@stub.function(secret=modal.Secret.from_name("api-secret-key"), keep_warm=1)
@stub.web_endpoint(method="POST")
def transcribe(api_args: APIArgs, x_modal_secret: str = Header(default=None)):
    log.info(f"Processing {api_args.url}")
    secret = os.environ["API_SECRET_KEY"]
    if secret and x_modal_secret != secret:
        return {"error": "Not authorized"}

    overrides = {}
    if api_args.llm_text:
        answer = llm_respond.call(api_args.llm_text)
        return {"llm_response": answer, "full_text": api_args.llm_text}
    if api_args.url:
        overrides["url"] = api_args.url
    if api_args.byte_string:
        overrides["filename"] = f"bytes-{int(time.time())}.mp3"
        overrides["min_segment_len"] = 0
        overrides["model"] = "tiny.en"

    if api_args.callback_url:
        results = FanTranscriber.queue(
            api_args.callback_url,
            cfg=WEB_DEFAULT_ARGS.merge(overrides),
            metadata=api_args.callback_metadata,
            summarize=bool(api_args.summarize),
        )
        return {"call_id": results.object_id}

    results = FanTranscriber.run(
        overrides,
        byte_string=api_args.byte_string,
    )
    return results
