"""LLM-assisted operator note interpretation module.

Extracts structured operational directives from campus operator notes using
generative language models (Google Gemini or OpenAI/Groq compatible APIs)
with an integrated offline fallback parser for network resilience and local testing.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
import httpx

from app.config import settings
from app.schemas import (
    BatteryInput,
    DirectiveInterpretation,
    DirectiveType,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert energy operations parser for the BUP CSE Fest 2026 GridWise energy scheduling system.
Your job is to read campus operator notes and extract machine-checkable directives for a 24-hour schedule (hours 0 to 23).

Supported directive types:
1. "solar_reduction":
   - Required structured_adjustment: {"hours": [int, ...], "factor": float}
   - factor is the remaining usable fraction (0.0 to 1.0).
     Example: "80% reduction" or "reduced by 80%" means factor = 0.2.
     Example: "drops to 25%" or "leaves 25%" means factor = 0.25.
     Example: "half of the forecast" means factor = 0.5.
     Example: "roughly one-fifth" means factor = 0.2.
2. "minimum_battery_reserve":
   - Required structured_adjustment: {"hours": [int, ...], "minimum_energy_kwh": float}
   - If stated as a percentage of battery capacity (e.g. 50%), multiply by the battery capacity to get kWh.
3. "no_charge_window":
   - Required structured_adjustment: {"hours": [int, ...]}
   - Battery charging is disabled during these hours.
4. "no_discharge_window":
   - Required structured_adjustment: {"hours": [int, ...]}
   - Battery discharging is disabled during these hours.
5. "max_grid_window":
   - Required structured_adjustment: {"hours": [int, ...], "max_grid_kwh": float}
   - Grid electricity import cannot exceed max_grid_kwh during these hours.
6. "no_op":
   - The note is irrelevant to today's 24-hour energy schedule (e.g. sports notice, library hours, cafeteria menu).
   - applies MUST be false.
   - structured_adjustment MUST be null.

Time window rules:
- Hours are whole-hour intervals, start-inclusive and end-exclusive.
  Example: "1 PM to 3 PM" -> [13, 14]
  Example: "noon until 2 PM" -> [12, 13]
  Example: "2 AM until 5 AM" -> [2, 3, 4]
  Example: "6 PM until 10 PM" -> [18, 19, 20, 21]
  Example: "13:00 to 15:00" -> [13, 14]
  Example: "from one until three" -> [13, 14] (if afternoon)
- hours must be unique integers in ascending order.

Response format:
You must return a JSON object with a "directives" array, containing exactly one entry per input note in note_index order (0 to N-1).
Each entry must contain:
{
  "note_index": int,
  "applies": bool,
  "directive_type": str,
  "structured_adjustment": dict or null,
  "explanation": str
}
"""


def _parse_time_phrase(text: str) -> Optional[List[int]]:
    """Extract whole-hour range [start..end-1] from natural language text."""
    lower = text.lower()

    # 1. 24-hour format: e.g. 13:00 and 15:00 or 13:00 to 15:00
    m_24 = re.search(r"(\d{1,2}):00\s*(?:to|until|-|and)\s*(\d{1,2}):00", lower)
    if m_24:
        start, end = int(m_24.group(1)), int(m_24.group(2))
        if 0 <= start < end <= 24:
            return list(range(start, end))

    # 2. X-Y PM format, e.g. "1-3 PM"
    m_dash = re.search(r"(\d{1,2})\s*-\s*(\d{1,2})\s*(am|pm)", lower)
    if m_dash:
        s_val, e_val, m = int(m_dash.group(1)), int(m_dash.group(2)), m_dash.group(3)
        if m == "pm":
            if s_val < 12:
                s_val += 12
            if e_val < 12:
                e_val += 12
        elif m == "am":
            if s_val == 12:
                s_val = 0
            if e_val == 12:
                e_val = 0
        if 0 <= s_val < e_val <= 24:
            return list(range(s_val, e_val))

    # 3. from/between START (am/pm)? until/to/and END (am/pm)?
    pattern = (
        r"(?:from|between)?\s*"
        r"(noon|midnight|\d{1,2}(?::00)?)\s*(am|pm)?\s*"
        r"(?:until|to|-|and)\s*"
        r"(noon|midnight|\d{1,2}(?::00)?)\s*(am|pm)?"
    )
    for m in re.finditer(pattern, lower):
        s_str, s_mer, e_str, e_mer = m.groups()
        if not s_str or not e_str:
            continue
        has_time_cue = any(x in ("noon", "midnight") for x in (s_str, e_str)) or bool(s_mer) or bool(e_mer)
        if not has_time_cue:
            continue

        def to_hour(val: str, mer: Optional[str], default_mer: str) -> int:
            if val == "noon":
                return 12
            if val == "midnight":
                return 0
            val_h = int(val.split(":")[0])
            effective_mer = (mer or default_mer).lower()
            if effective_mer == "pm" and val_h < 12:
                return val_h + 12
            if effective_mer == "am" and val_h == 12:
                return 0
            return val_h

        def_mer = e_mer or s_mer or "pm"
        sh = to_hour(s_str, s_mer, def_mer)
        eh = to_hour(e_str, e_mer, def_mer)
        if 0 <= sh < eh <= 24:
            return list(range(sh, eh))

    # 4. Words: e.g. from one until three
    word_to_num = {"one": 13, "two": 14, "three": 15, "four": 16, "five": 17, "six": 18}
    m_words = re.search(
        r"(?:from|between)?\s*(one|two|three|four|five|six)\s*(?:until|to|-|and)\s*(one|two|three|four|five|six)",
        lower,
    )
    if m_words:
        s_w, e_w = m_words.groups()
        sh = word_to_num[s_w]
        eh = word_to_num[e_w]
        if 0 <= sh < eh <= 24:
            return list(range(sh, eh))

    return None


def heuristic_parse_note(
    note: str,
    note_index: int,
    battery: BatteryInput,
) -> Dict[str, Any]:
    """Offline heuristic parser extracting directives from natural language text.

    Serves as an autonomous, robust fallback ensuring complete functionality
    even when remote LLM APIs are offline or unconfigured.
    """
    lower = note.lower()

    # Determine if note is unrelated (no_op)
    no_op_keywords = [
        "cafeteria", "menu", "sports", "registration", "library", "book", "club",
        "notice", "seminar", "booking", "exam", "bus", "transport", "holiday",
        "deadline", "parking", "laundry", "dormitory", "hostel", "classroom",
    ]
    if any(kw in lower for kw in no_op_keywords):
        return {
            "note_index": note_index,
            "applies": False,
            "directive_type": DirectiveType.NO_OP.value,
            "structured_adjustment": None,
            "explanation": "This note does not affect today's 24-hour energy schedule.",
        }

    hours = _parse_time_phrase(note)
    if not hours:
        # If no time window could be extracted, treat as no_op
        return {
            "note_index": note_index,
            "applies": False,
            "directive_type": DirectiveType.NO_OP.value,
            "structured_adjustment": None,
            "explanation": "No operational time window identified in note; treated as no_op.",
        }

    # 1. Solar reduction
    if any(k in lower for k in ("solar", "pv", "panel", "cloud", "sun", "photovoltaic", "rooftop")):
        factor = 1.0
        if "half" in lower:
            factor = 0.5
        elif "one-fifth" in lower or "1/5" in lower:
            factor = 0.2
        elif "quarter" in lower or "1/4" in lower:
            factor = 0.25
        else:
            pct_match = re.search(r"(\d{1,3})%", lower)
            if pct_match:
                pct = float(pct_match.group(1))
                if any(w in lower for w in ("reduction", "drop by", "reduced by", "decrease by", "cut by")):
                    factor = max(0.0, (100.0 - pct) / 100.0)
                else:
                    factor = pct / 100.0

        return {
            "note_index": note_index,
            "applies": True,
            "directive_type": DirectiveType.SOLAR_REDUCTION.value,
            "structured_adjustment": {"hours": hours, "factor": round(factor, 4)},
            "explanation": f"Solar output adjusted to {factor*100:.0f}% for hours {hours}.",
        }

    # 2. No discharge window (check discharge first to distinguish from charge)
    if "discharge" in lower and any(
        w in lower for w in ("not discharge", "no discharge", "cannot discharge", "disable", "do not", "must not", "testing", "stop", "unavailable")
    ):
        return {
            "note_index": note_index,
            "applies": True,
            "directive_type": DirectiveType.NO_DISCHARGE_WINDOW.value,
            "structured_adjustment": {"hours": hours},
            "explanation": f"Battery discharging disabled for hours {hours}.",
        }

    # 3. No charge window
    if ("charg" in lower and "discharge" not in lower) and any(
        w in lower for w in ("not charge", "no charge", "cannot charge", "disable", "unavailable", "isolated", "do not", "must not", "prevent", "stop", "off")
    ):
        return {
            "note_index": note_index,
            "applies": True,
            "directive_type": DirectiveType.NO_CHARGE_WINDOW.value,
            "structured_adjustment": {"hours": hours},
            "explanation": f"Battery charging disabled for hours {hours}.",
        }

    # 4. Minimum battery reserve
    if any(k in lower for k in ("reserve", "remain in the battery", "stored in the battery", "keep at least", "minimum", "emergency")):
        min_kwh = battery.minimum_energy_kwh
        pct_match = re.search(r"(\d{1,3})%", lower)
        if pct_match:
            pct = float(pct_match.group(1))
            min_kwh = (pct / 100.0) * battery.capacity_kwh
        else:
            kwh_match = re.search(r"(\d+(?:\.\d+)?)\s*kwh", lower)
            if kwh_match:
                min_kwh = float(kwh_match.group(1))

        return {
            "note_index": note_index,
            "applies": True,
            "directive_type": DirectiveType.MINIMUM_BATTERY_RESERVE.value,
            "structured_adjustment": {"hours": hours, "minimum_energy_kwh": round(min_kwh, 2)},
            "explanation": f"Minimum battery reserve set to {min_kwh:.1f} kWh for hours {hours}.",
        }

    # 5. Max grid window
    if any(k in lower for k in ("grid", "import", "feeder", "transformer", "intake", "substation")):
        max_grid = 1000.0
        kwh_match = re.search(r"(\d+(?:\.\d+)?)\s*kwh", lower)
        if kwh_match:
            max_grid = float(kwh_match.group(1))

        return {
            "note_index": note_index,
            "applies": True,
            "directive_type": DirectiveType.MAX_GRID_WINDOW.value,
            "structured_adjustment": {"hours": hours, "max_grid_kwh": round(max_grid, 2)},
            "explanation": f"Grid import capped at {max_grid:.1f} kWh for hours {hours}.",
        }

    # Default fallback to no_op
    return {
        "note_index": note_index,
        "applies": False,
        "directive_type": DirectiveType.NO_OP.value,
        "structured_adjustment": None,
        "explanation": "No operational directive matched; treated as no_op.",
    }


async def _call_gemini_api(
    operator_notes: List[str],
    battery: BatteryInput,
    api_key: str,
) -> List[Dict[str, Any]]:
    """Call Google Gemini REST API using httpx with automatic model failover."""
    models_to_try = [settings.LLM_MODEL]
    for candidate in ("gemini-3-flash-preview", "gemini-flash-latest"):
        if candidate not in models_to_try:
            models_to_try.append(candidate)

    notes_text = "\n".join(f"[{i}]: {n}" for i, n in enumerate(operator_notes))
    user_prompt = (
        f"Campus Battery Configuration:\n"
        f"- capacity_kwh: {battery.capacity_kwh}\n"
        f"- minimum_energy_kwh: {battery.minimum_energy_kwh}\n"
        f"- initial_energy_kwh: {battery.initial_energy_kwh}\n\n"
        f"Operator Notes to interpret:\n{notes_text}\n\n"
        f"Return the directives JSON."
    )

    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.0,
        },
    }

    last_error: Optional[Exception] = None
    async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
        for model in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            try:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()

                text = data["candidates"][0]["content"]["parts"][0]["text"]
                parsed = json.loads(text)
                if isinstance(parsed, dict) and "directives" in parsed:
                    return parsed["directives"]
                if isinstance(parsed, list):
                    return parsed
            except Exception as exc:
                logger.warning("Gemini model '%s' failed (%s: %s). Trying next model if available.", model, type(exc).__name__, exc)
                last_error = exc
                continue

    if last_error:
        raise last_error
    raise ValueError("Invalid Gemini response JSON structure")


async def _call_openai_compatible_api(
    operator_notes: List[str],
    battery: BatteryInput,
    api_key: str,
) -> List[Dict[str, Any]]:
    """Call OpenAI / Groq compatible chat completions REST API using httpx."""
    base_url = settings.LLM_BASE_URL or "https://api.openai.com/v1"
    url = f"{base_url.rstrip('/')}/chat/completions"

    notes_text = "\n".join(f"[{i}]: {n}" for i, n in enumerate(operator_notes))
    user_prompt = (
        f"Campus Battery Configuration:\n"
        f"- capacity_kwh: {battery.capacity_kwh}\n"
        f"- minimum_energy_kwh: {battery.minimum_energy_kwh}\n"
        f"- initial_energy_kwh: {battery.initial_energy_kwh}\n\n"
        f"Operator Notes to interpret:\n{notes_text}\n\n"
        f"Return the directives JSON."
    )

    payload = {
        "model": settings.LLM_MODEL,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        text = data["choices"][0]["message"]["content"]
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "directives" in parsed:
            return parsed["directives"]
        if isinstance(parsed, list):
            return parsed
        raise ValueError("Invalid OpenAI response JSON structure")


async def extract_directives_from_notes(
    operator_notes: List[str],
    battery: BatteryInput,
) -> List[Dict[str, Any]]:
    """Extract structured directives from operator notes via LLM or resilient fallback."""
    api_key = settings.get_effective_api_key()
    provider = settings.LLM_PROVIDER.lower()

    # If API key is available, call the configured LLM provider
    if api_key and provider != "mock":
        try:
            if provider == "gemini":
                logger.info("Interpreting operator notes using Google Gemini API...")
                return await _call_gemini_api(operator_notes, battery, api_key)
            elif provider in ("openai", "groq", "openrouter", "ollama"):
                logger.info("Interpreting operator notes using OpenAI/Groq API...")
                return await _call_openai_compatible_api(operator_notes, battery, api_key)
        except Exception as e:
            logger.warning(
                "Remote LLM call failed (%s: %s). Activating resilient fallback parser.",
                type(e).__name__,
                e,
            )

    # Fallback to local heuristic extraction
    logger.info("Executing resilient local directive extraction for %d note(s).", len(operator_notes))
    results = []
    for i, note in enumerate(operator_notes):
        parsed_note = heuristic_parse_note(note, i, battery)
        results.append(parsed_note)
    return results
