"""FaultLens on Modal: the public demo.

Deploy from the project folder (your .env must hold the model keys):

    modal deploy modal_app.py

What this does:
- Builds an image with the code, the reference library and the reports, and trains the
  pattern model inside the image (about 2-3 minutes, once per deploy).
- Sends your .env to Modal as a secret. The .env file itself is never copied into the image.
- Runs one container, so a job's state and the visitor limits live in one place.
- Turns on PUBLIC_DEMO: paid model calls are limited per visitor and per day, and the
  full test-set run is switched off (the saved run is shown instead).
"""

from pathlib import Path

import modal

HERE = Path(__file__).parent
APP_DIR = "/root/app"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(HERE / "requirements.txt"))
    .add_local_dir(
        str(HERE), APP_DIR, copy=True,
        ignore=[".env", ".env.*", ".venv", "venv", ".git", "**/__pycache__", "**/*.pyc",
                "data/runs", "models/*.joblib", "*.zip", "manuals", "raw_footage"],
    )
    # Train the pattern model here, so it matches this image's library versions.
    # The published report (models/pattern_report.json) is kept as it was measured.
    .run_commands(
        f"cd {APP_DIR} && cp models/pattern_report.json /tmp/pattern_report.json "
        "&& python -m faultlens.pattern && cp /tmp/pattern_report.json models/pattern_report.json"
    )
    .env({"PUBLIC_DEMO": "1"})
)

app = modal.App("faultlens")


@app.function(
    image=image,
    secrets=[modal.Secret.from_dotenv(__file__)],
    max_containers=1,          # one container: game jobs and visitor limits are kept in memory
    scaledown_window=1800,     # stay warm 30 minutes after the last visitor
    timeout=600,
)
@modal.concurrent(max_inputs=40)
@modal.asgi_app()
def web():
    import os
    import sys

    os.chdir(APP_DIR)
    sys.path.insert(0, APP_DIR)
    from web.server import app as fastapi_app

    return fastapi_app
