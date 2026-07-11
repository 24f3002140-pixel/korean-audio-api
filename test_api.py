import base64
import json
from pathlib import Path

import requests


AUDIO_FILE = "sample.wav.ogg"
API_URL = "http://127.0.0.1:8000/process-audio"
AUDIO_ID = "q0"


def main() -> None:
    audio_path = Path(AUDIO_FILE)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"{AUDIO_FILE} was not found.\n"
            "Place the audio file inside the same project folder."
        )

    audio_base64 = base64.b64encode(
        audio_path.read_bytes()
    ).decode("utf-8")

    response = requests.post(
        API_URL,
        json={
            "audio_id": AUDIO_ID,
            "audio_base64": audio_base64,
        },
        timeout=300,
    )

    print("Status code:", response.status_code)

    try:
        print(
            json.dumps(
                response.json(),
                indent=2,
                ensure_ascii=False,
            )
        )
    except ValueError:
        print(response.text)


if __name__ == "__main__":
    main()