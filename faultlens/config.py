"""Central settings. Everything comes from environment variables (.env)."""

import os

from dotenv import load_dotenv

load_dotenv()

META_BASE_URL = os.getenv("META_BASE_URL", "https://api.meta.ai/v1")
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "")

# Main reasoning agent. Free hackathon credits are billed like the contributor tier.
AGENT_MODEL = os.getenv("AGENT_MODEL", "muse-spark-1.3")

# Part finding and tracking.
SAM_MODEL = os.getenv("SAM_MODEL", "sam-3.1")

# Speech to text (recorded WAV files).
VOICE_MODEL = os.getenv("VOICE_MODEL", "muse-voice-transcribe-1.0")

# Independent checker. Leave CHECKER_BASE_URL empty to use Muse Spark on Meta's API.
# To use Muse Spark Open on a partner (Baseten, Modal, Fireworks), set all three.
CHECKER_BASE_URL = os.getenv("CHECKER_BASE_URL", "")
CHECKER_API_KEY = os.getenv("CHECKER_API_KEY", "")
CHECKER_MODEL = os.getenv("CHECKER_MODEL", "")
if not CHECKER_API_KEY or CHECKER_API_KEY.startswith("your-"):
    # No real partner key yet: run the checker on Meta's API instead.
    CHECKER_BASE_URL = CHECKER_API_KEY = CHECKER_MODEL = ""

# Fireworks: Muse Spark Open, used to compare Meta's open model in the vision benchmark
# and as a backup host for the checker.
FIREWORKS_BASE_URL = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
FIREWORKS_MODEL = os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/muse-spark-open")

# Which equipment profile to load from faultlens/knowledge/<name>.yaml
EQUIPMENT_PROFILE = os.getenv("EQUIPMENT_PROFILE", "split_ac")

# Offline mode: no API calls, canned answers. For building the UI and tests
# while the API account is unavailable. Never use for results you report.
MOCK = os.getenv("FAULTLENS_MOCK", "0") == "1"

# Reasoning effort for the step-by-step coach (low, medium, high). Medium keeps turns quick.
COACH_REASONING = os.getenv("COACH_REASONING", "medium")

# Agent loop limits.
MAX_AGENT_TURNS = int(os.getenv("MAX_AGENT_TURNS", "8"))
KEYFRAMES_PER_VIDEO = int(os.getenv("KEYFRAMES_PER_VIDEO", "4"))

# Set by `python -m faultlens.checker` once a sound format is proven to reach the checker model.
CHECKER_AUDIO_FORMAT = os.getenv("CHECKER_AUDIO_FORMAT", "").strip()

# Public demo (the Modal deployment). Paid model calls are limited per visitor and per day, and the
# full test-set run is switched off (the saved run is shown instead).
PUBLIC_DEMO = os.getenv("PUBLIC_DEMO", "0") == "1"
VISITOR_CALLS_PER_HOUR = int(os.getenv("VISITOR_CALLS_PER_HOUR", "60"))
DAILY_CALLS = int(os.getenv("DAILY_CALLS", "800"))
