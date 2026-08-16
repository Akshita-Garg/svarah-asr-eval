"""Call Smallest.ai Pulse and Pulse Pro with the same audio file."""

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

API_URL = "https://api.smallest.ai/waves/v1/stt/"
API_KEY = os.getenv("SMALLEST_API_KEY") or os.getenv("SMALLESTAI_API_KEY")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/octet-stream",
}
audio = AUDIO_FILE.read_bytes()

pulse_response = requests.post(
    API_URL,
    params={"model": "pulse", "language": "en"},
    headers=headers,
    data=audio,
)
pulse_response.raise_for_status()

pulse_pro_response = requests.post(
    API_URL,
    params={"model": "pulse-pro", "language": "en"},
    headers=headers,
    data=audio,
)
pulse_pro_response.raise_for_status()

print("\nPULSE OUTPUT:")
print(pulse_response.json()["transcription"])

print("\nPULSE PRO OUTPUT:")
print(pulse_pro_response.json()["transcription"])
