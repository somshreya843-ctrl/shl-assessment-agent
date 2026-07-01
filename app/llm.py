import json
import os
import re

from dotenv import load_dotenv
load_dotenv()

from typing import List, Dict, Any
from groq import Groq

MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        _client = Groq(api_key=api_key)
    return _client


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def call_control_model(
    system: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 600,
) -> Dict[str, Any]:

    client = _get_client()

    groq_messages = [{"role": "system", "content": system}] + messages

    response = client.chat.completions.create(
        model=MODEL,
        messages=groq_messages,
        temperature=0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    cleaned = _strip_fences(raw)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"Model did not return valid JSON.\n\nOutput:\n{raw}")