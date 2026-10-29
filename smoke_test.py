"""FaultLens - smoke test for Meta Model API.

Checks two things:
  1. A plain text call to Muse Spark works (key, billing, model ID are right).
  2. An image call works (pass a photo path as the first argument).

Usage:
  pip install openai python-dotenv
  python smoke_test.py                 # text only
  python smoke_test.py unit_photo.jpg  # text + image
"""

import base64
import mimetypes
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads MODEL_API_KEY from a .env file next to this script

MODEL = "muse-spark-1.3"

client = OpenAI(
    base_url="https://api.meta.ai/v1",
    api_key=os.environ["MODEL_API_KEY"],  # the SDK does not read this name on its own
)


def text_check() -> None:
    resp = client.responses.create(
        model=MODEL,
        input="In one sentence: what are the three most common reasons an air conditioner stops cooling?",
    )
    print("TEXT CALL OK")
    print(resp.output_text)
    print("Tokens used:", resp.usage.input_tokens, "in /", resp.usage.output_tokens, "out\n")


def image_check(path: str) -> None:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    resp = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "This is a photo of cooling equipment. List the visible parts "
                            "(fans, coils, panels, breakers, labels) and anything that looks "
                            "damaged, dirty or abnormal. Say 'not visible' rather than guessing."
                        ),
                    },
                    {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"},
                ],
            }
        ],
    )
    print("IMAGE CALL OK")
    print(resp.output_text)
    print("Tokens used:", resp.usage.input_tokens, "in /", resp.usage.output_tokens, "out")


if __name__ == "__main__":
    text_check()
    if len(sys.argv) > 1:
        image_check(sys.argv[1])
