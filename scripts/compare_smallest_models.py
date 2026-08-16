"""Call Smallest.ai Pulse and Pulse Pro with the same audio file.

Also probes whether the documented `language` query parameter has any effect on
Pulse Pro's output script, and prints the full response body so a wrong-script
result can be quoted verbatim as evidence.
"""

import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voicerefine_eval.config import load_dotenv

load_dotenv()

# Change this line to test another prepared Svarah recording.
AUDIO_FILE = ROOT / "data" / "prepared" / "svarah_test_0048.wav"

# (model, language) pairs. `None` omits the language parameter entirely, which
# shows whether the parameter has any effect on the returned script.
CALLS = [
    ("pulse-pro", "en"),
    ("pulse-pro", None),
    ("pulse-pro", "hi"),
    ("pulse", "en"),
]

API_URL = "https://api.smallest.ai/waves/v1/stt/"
API_KEY = os.getenv("SMALLEST_API_KEY") or os.getenv("SMALLESTAI_API_KEY")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/octet-stream",
}
audio = AUDIO_FILE.read_bytes()

print(f"AUDIO: {AUDIO_FILE.name} ({len(audio)} bytes)")

for model, language in CALLS:
    params = {"model": model}
    if language is not None:
        params["language"] = language

    response = requests.post(API_URL, params=params, headers=headers, data=audio)

    print(f"\n=== model={model} language={language} ===")
    print(f"REQUEST URL: {response.url}")
    print(f"HTTP STATUS: {response.status_code}")

    # A rejected language is evidence too: it shows the parameter is validated
    # rather than silently ignored, so an accepted `en` really did select English.
    try:
        body = response.json()
    except ValueError:
        print(f"NON-JSON BODY: {response.text[:1000]}")
        continue

    if response.ok:
        print(f"TRANSCRIPTION: {body.get('transcription')!r}")
    print("FULL RESPONSE:")
    print(json.dumps(body, ensure_ascii=False, indent=2)[:4000])
