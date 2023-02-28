from pydantic import BaseModel
from transcriber import stub, CACHE_DIR, volume


class APIArgs(BaseModel):
    name: str
    qty: int = 42


@stub.webhook(
    method="POST",
    shared_volumes={CACHE_DIR: volume},
)
def transcribe(args: APIArgs):
    return {"foo": "bar"}
