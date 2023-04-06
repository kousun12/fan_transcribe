import modal
import os
from pydantic import BaseModel
from transcriber import stub
from fastapi import Header
from logger import log


class LLMArgs(BaseModel):
    prompt: str
    context: str


@stub.webhook(
    label="llm",
    method="POST",
    keep_warm=True,
    secrets=[
        modal.Secret.from_name("api-secret-key"),
        modal.Secret.from_name("openai-secret-key"),
        modal.Secret.from_name("openai-org-id"),
    ],
)
def llm(args: LLMArgs, x_modal_secret: str = Header(default=None)):
    secret = os.environ["API_SECRET_KEY"]
    if secret and x_modal_secret != secret:
        return {"error": "Not authorized"}

    log.info("Running LLM response")
    import openai

    openai.organization = os.environ["OPENAI_ORGANIZATION_KEY"]
    system_msg = f"""You will be given a prompt and maybe some context. Always respond in a json format. The object should have two keys: "data" and "type". "data" is your response, and "type" is either "summarize" or "respond" depending on whether or not this is a summarization task or some other question/ask.

example response format:
{{ "data": YOUR_RESPONSE, "type": SUMMARIZE_OR_RESPOND }}
"""
    user_msg = f"""context:

{args.context or 'No context'}

context:

{args.prompt}"""
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.5,
            n=1,
        )
        return {"data": response["choices"][0]["message"]["content"].strip()}
    except Exception as e:
        print(e)
        return {"data": "I don't know"}
