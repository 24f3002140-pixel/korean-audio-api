import base64
import binascii
import json
import os
import re
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# =========================================================
# Configuration
# =========================================================

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.1-flash-lite",
)

MAX_AUDIO_SIZE_MB = 20


# =========================================================
# FastAPI application
# =========================================================

app = FastAPI(
    title="Korean Audio Dataset Statistics API",
    version="1.0.0",
)


# Allows the IITM grader and Cloudflare Worker to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Request schema
# =========================================================

class AudioRequest(BaseModel):
    audio_id: str = Field(
        ...,
        min_length=1,
        description="Identifier supplied by the grader",
    )

    audio_base64: str = Field(
        ...,
        min_length=1,
        description="Base64-encoded audio",
    )


# =========================================================
# Required response keys
# =========================================================

REQUIRED_KEYS = [
    "rows",
    "columns",
    "mean",
    "std",
    "variance",
    "min",
    "max",
    "median",
    "mode",
    "range",
    "allowed_values",
    "value_range",
    "correlation",
]


def empty_result() -> dict[str, Any]:
    """
    Return the required response structure with empty defaults.
    """

    return {
        "rows": 0,
        "columns": [],
        "mean": {},
        "std": {},
        "variance": {},
        "min": {},
        "max": {},
        "median": {},
        "mode": {},
        "range": {},
        "allowed_values": {},
        "value_range": {},
        "correlation": [],
    }


# =========================================================
# Audio helpers
# =========================================================

def remove_data_url_prefix(
    audio_base64: str,
) -> tuple[str, str | None]:
    """
    Support both plain base64 and data URLs such as:

    data:audio/wav;base64,UklGR...
    """

    value = audio_base64.strip()
    mime_type = None

    if value.startswith("data:"):
        if "," not in value:
            raise ValueError("Invalid audio data URL.")

        header, value = value.split(",", 1)

        match = re.match(
            r"data:([^;]+);base64",
            header,
            flags=re.IGNORECASE,
        )

        if match:
            mime_type = match.group(1).lower()

    value = re.sub(r"\s+", "", value)

    return value, mime_type


def detect_audio_mime_type(
    audio_bytes: bytes,
    supplied_mime_type: str | None,
) -> str:
    """
    Detect the audio format from its file signature.
    """

    if supplied_mime_type:
        allowed_supplied_types = {
            "audio/wav",
            "audio/x-wav",
            "audio/wave",
            "audio/mpeg",
            "audio/mp3",
            "audio/ogg",
            "audio/flac",
            "audio/x-flac",
            "audio/mp4",
            "audio/m4a",
            "audio/aac",
            "audio/webm",
        }

        if supplied_mime_type in allowed_supplied_types:
            return supplied_mime_type

    # WAV: starts with RIFF and normally contains WAVE.
    if (
        len(audio_bytes) >= 12
        and audio_bytes[:4] == b"RIFF"
        and audio_bytes[8:12] == b"WAVE"
    ):
        return "audio/wav"

    # MP3 with ID3 metadata.
    if audio_bytes.startswith(b"ID3"):
        return "audio/mpeg"

    # MP3 frame sync.
    if (
        len(audio_bytes) >= 2
        and audio_bytes[0] == 0xFF
        and (audio_bytes[1] & 0xE0) == 0xE0
    ):
        return "audio/mpeg"

    # OGG.
    if audio_bytes.startswith(b"OggS"):
        return "audio/ogg"

    # FLAC.
    if audio_bytes.startswith(b"fLaC"):
        return "audio/flac"

    # MP4/M4A commonly contains ftyp near the beginning.
    if (
        len(audio_bytes) >= 12
        and audio_bytes[4:8] == b"ftyp"
    ):
        return "audio/mp4"

    # WebM/Matroska EBML signature.
    if audio_bytes.startswith(
        b"\x1a\x45\xdf\xa3"
    ):
        return "audio/webm"

    # AAC ADTS frame sync.
    if (
        len(audio_bytes) >= 2
        and audio_bytes[0] == 0xFF
        and (audio_bytes[1] & 0xF6) == 0xF0
    ):
        return "audio/aac"

    # The grader is most likely to use WAV or MP3.
    # WAV is used as a final fallback.
    return "audio/wav"


def decode_audio(
    audio_base64: str,
) -> tuple[bytes, str]:
    """
    Decode base64 audio and determine its MIME type.
    """

    cleaned_base64, supplied_mime_type = (
        remove_data_url_prefix(audio_base64)
    )

    try:
        audio_bytes = base64.b64decode(
            cleaned_base64,
            validate=True,
        )
    except (binascii.Error, ValueError) as error:
        raise ValueError(
            "audio_base64 is not valid base64."
        ) from error

    if not audio_bytes:
        raise ValueError(
            "The decoded audio is empty."
        )

    maximum_bytes = MAX_AUDIO_SIZE_MB * 1024 * 1024

    if len(audio_bytes) > maximum_bytes:
        raise ValueError(
            f"The audio is larger than "
            f"{MAX_AUDIO_SIZE_MB} MB."
        )

    mime_type = detect_audio_mime_type(
        audio_bytes,
        supplied_mime_type,
    )

    return audio_bytes, mime_type


# =========================================================
# JSON cleanup helpers
# =========================================================

def extract_json_object(text: str) -> dict[str, Any]:
    """
    Convert Gemini's response into a Python dictionary.

    Handles:
    - plain JSON
    - ```json ... ``` blocks
    - accidental text surrounding the JSON
    """

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

    try:
        parsed = json.loads(cleaned)

        if not isinstance(parsed, dict):
            raise ValueError(
                "The model response is not a JSON object."
            )

        return parsed

    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            "The model did not return valid JSON."
        )

    extracted = cleaned[start:end + 1]

    try:
        parsed = json.loads(extracted)
    except json.JSONDecodeError as error:
        raise ValueError(
            "The model returned malformed JSON."
        ) from error

    if not isinstance(parsed, dict):
        raise ValueError(
            "The model response is not a JSON object."
        )

    return parsed


def convert_numeric_string(
    value: Any,
) -> Any:
    """
    Convert strings that clearly represent numbers into JSON numbers.

    Other strings are preserved.
    """

    if not isinstance(value, str):
        return value

    cleaned = value.strip()

    if not cleaned:
        return value

    cleaned_without_commas = cleaned.replace(",", "")

    if re.fullmatch(
        r"-?\d+",
        cleaned_without_commas,
    ):
        try:
            return int(cleaned_without_commas)
        except ValueError:
            return value

    if re.fullmatch(
        r"-?(?:\d+\.\d+|\d+\.|\.\d+)"
        r"(?:[eE][+-]?\d+)?",
        cleaned_without_commas,
    ):
        try:
            return float(cleaned_without_commas)
        except ValueError:
            return value

    return value


def recursively_clean_values(
    value: Any,
) -> Any:
    """
    Recursively clean values while preserving the exact structure.
    """

    if isinstance(value, dict):
        return {
            str(key): recursively_clean_values(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            recursively_clean_values(item)
            for item in value
        ]

    return convert_numeric_string(value)


def normalize_result(
    model_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Return exactly the eleven required keys with appropriate defaults.
    """

    result = empty_result()

    rows = model_result.get("rows")

    if isinstance(rows, bool):
        rows = 0
    elif isinstance(rows, float) and rows.is_integer():
        rows = int(rows)
    elif isinstance(rows, str):
        cleaned_rows = rows.replace(",", "").strip()

        if re.fullmatch(r"\d+", cleaned_rows):
            rows = int(cleaned_rows)
        else:
            rows = 0
    elif not isinstance(rows, int):
        rows = 0

    result["rows"] = rows

    columns = model_result.get("columns")

    if isinstance(columns, list):
        result["columns"] = [
            str(column)
            for column in columns
        ]
    else:
        result["columns"] = []

    dictionary_fields = [
        "mean",
        "std",
        "variance",
        "min",
        "max",
        "median",
        "mode",
        "range",
        "allowed_values",
        "value_range",
    ]

    for field_name in dictionary_fields:
        field_value = model_result.get(field_name)

        if isinstance(field_value, dict):
            result[field_name] = (
                recursively_clean_values(field_value)
            )
        else:
            result[field_name] = {}

    correlation = model_result.get("correlation")

    if isinstance(correlation, list):
        result["correlation"] = (
            recursively_clean_values(correlation)
        )
    else:
        result["correlation"] = []

    # Rebuild explicitly to guarantee exact key ordering
    # and prevent any extra model-generated keys.
    return {
        "rows": result["rows"],
        "columns": result["columns"],
        "mean": result["mean"],
        "std": result["std"],
        "variance": result["variance"],
        "min": result["min"],
        "max": result["max"],
        "median": result["median"],
        "mode": result["mode"],
        "range": result["range"],
        "allowed_values": result["allowed_values"],
        "value_range": result["value_range"],
        "correlation": result["correlation"],
    }


# =========================================================
# Prompt
# =========================================================

def create_prompt(audio_id: str) -> str:
    """
    Instructions for extracting the required JSON from Korean audio.
    """

    return f"""
You are a highly accurate Korean-language audio data extraction system.

The attached audio is identified as:

{audio_id}

Listen to the entire audio carefully.

The Korean speech describes a dataset and the exact statistics,
constraints, metadata, or relationships that must be returned.

The audio may state:

- the number of rows
- column names
- means
- standard deviations
- variances
- minimum values
- maximum values
- medians
- modes
- ranges
- allowed categorical values
- permitted numeric value ranges
- correlations between columns

Your job is to understand the Korean audio and return the exact
information described in it.

Return exactly one JSON object with exactly these eleven keys:

{{
  "rows": 0,
  "columns": [],
  "mean": {{}},
  "std": {{}},
  "variance": {{}},
  "min": {{}},
  "max": {{}},
  "median": {{}},
  "mode": {{}},
  "range": {{}},
  "allowed_values": {{}},
  "value_range": {{}},
  "correlation": []
}}

STRICT RULES:

1. Listen to all Korean speech before answering.

2. Translate and interpret Korean correctly, but do not return
   a transcription or translation.

3. Return only the required JSON object.

4. Return exactly the eleven keys shown above.

5. Do not include extra keys.

6. Do not omit any key.

7. Preserve column names exactly as spoken or specified.

8. rows must be a JSON integer.

9. columns must be a JSON array of strings in the order specified.

10. Statistical values must be JSON numbers, not strings.

11. Do not round values unless the audio explicitly gives rounded values.

12. Preserve negative numbers and decimal precision.

13. mean, std, variance, min, max, median, mode, and range
    must be JSON objects keyed by column name.

14. allowed_values must map each applicable column name to its
    specified list of permitted values.

15. value_range must preserve the exact range representation
    requested in the audio.

16. correlation must preserve the exact list structure and values
    described in the audio.

17. If the audio says a field is empty, return the correct empty
    object or empty list.

18. Distinguish standard deviation from variance.

19. Distinguish minimum/maximum statistics from allowed value ranges.

20. Do not calculate unrelated values or invent missing information.

21. Return valid JSON only. Do not use Markdown code fences,
    explanations, Korean commentary, or English commentary.
""".strip()


# =========================================================
# Routes
# =========================================================

@app.get("/")
def home() -> dict[str, str]:
    return {
        "status": "online",
        "endpoint": "POST /process-audio",
        "model": MODEL_NAME,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
    }


@app.post("/")
def process_audio_root(
    request: AudioRequest,
) -> JSONResponse:
    """
    Accept requests at the base URL as well.

    This is useful because the assignment asks users to submit
    an API endpoint URL and may send POST directly to that URL.
    """

    return handle_audio_request(request)


@app.post("/process-audio")
def process_audio(
    request: AudioRequest,
) -> JSONResponse:
    return handle_audio_request(request)


@app.post("/analyze")
def analyze_audio(
    request: AudioRequest,
) -> JSONResponse:
    """
    Additional compatible endpoint in case the grader expects
    an explicit audio-analysis path.
    """

    return handle_audio_request(request)


def handle_audio_request(
    request: AudioRequest,
) -> JSONResponse:
    """
    Decode the audio, send it to Gemini, and return the required JSON.
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured.",
        )

    try:
        audio_bytes, mime_type = decode_audio(
            request.audio_base64
        )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    prompt = create_prompt(
        request.audio_id.strip()
    )

    try:
        client = genai.Client(
            api_key=api_key
        )

        audio_part = types.Part.from_bytes(
            data=audio_bytes,
            mime_type=mime_type,
        )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                audio_part,
                prompt,
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
            ),
        )

        if not response.text:
            raise HTTPException(
                status_code=502,
                detail="Gemini returned an empty response.",
            )

        try:
            model_result = extract_json_object(
                response.text
            )
        except ValueError as error:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"{str(error)} "
                    f"Response: {response.text[:500]}"
                ),
            ) from error

        normalized_result = normalize_result(
            model_result
        )

        return JSONResponse(
            status_code=200,
            content=normalized_result,
        )

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini API error: {str(error)}",
        ) from error