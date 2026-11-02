"""Google Gemini backend (works with a free Gemini API key from aistudio.google.com).

The pipeline keeps its conversation in Anthropic's message format. This module
converts it for Gemini on each call and returns an object that looks like an
Anthropic message holding one text block with the JSON transcription, so the
rest of the pipeline is shared. Gemini is asked for JSON matching the same
schema the Claude tool uses (structured output), at high media resolution so
small symbols stay legible.

Free-tier quotas: per-minute limits are waited out automatically; when the
daily quota is used up, QuotaExhausted is raised so the run can pause and be
resumed later without re-reading finished systems.
"""
from __future__ import annotations

import base64
import os
import re
import threading
import time
from types import SimpleNamespace

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_NOTE = (
    "\n\n# Output format in this setup\n"
    "There is no tool to call here. Wherever these instructions or messages say to call a submit_... tool, "
    "reply instead with exactly the JSON object that tool would receive (the response format enforces it). "
    "Never answer with anything except that JSON object."
)
_LEVELS = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}


class QuotaExhausted(RuntimeError):
    """The provider refuses further work for now (daily free quota used up, or no credit left)."""


def is_gemini(model: str | None) -> bool:
    return bool(model) and str(model).lower().startswith("gemini")


class _Block:
    def __init__(self, text: str, raw=None):
        self.type = "text"
        self.text = text
        self._raw = raw

    def model_dump(self, exclude_none: bool = True) -> dict:
        d = {"type": "text", "text": self.text}
        if self._raw is not None:
            d["_gemini_content"] = self._raw  # original Content, kept so thought signatures go back unchanged
        return d


def _retry_delay(details) -> float | None:
    try:
        for d in (details.get("error", details) or {}).get("details", []) or []:
            if str(d.get("@type", "")).endswith("RetryInfo") and d.get("retryDelay"):
                return float(str(d["retryDelay"]).rstrip("s"))
    except Exception:
        pass
    return None


def _is_daily(err) -> bool:
    text = str(getattr(err, "details", "")) + " " + str(getattr(err, "message", ""))
    return bool(re.search(r"PerDay|per day|daily", text, re.I))


class GeminiBackend:
    def __init__(self, api_key: str | None, model: str = DEFAULT_GEMINI_MODEL, effort: str | None = "high",
                 thinking: bool = True, log=print):
        from google import genai
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("No Gemini API key. Get a free one at https://aistudio.google.com/apikey")
        base_url = os.environ.get("SCORESCRIBE_GEMINI_BASE_URL")  # for tests
        http = {"base_url": base_url} if base_url else None
        self.client = genai.Client(api_key=key, http_options=http)
        self.model = model
        self.level = _LEVELS.get(effort or "high", "high") if thinking else None
        self.media_high = True
        self.log = log
        self.stopped: QuotaExhausted | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------ conversion
    @staticmethod
    def _contents(messages: list[dict]):
        from google.genai import types
        out = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            blocks = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
            raw = next((b.get("_gemini_content") for b in blocks if isinstance(b, dict) and b.get("_gemini_content")), None)
            if raw is not None:
                out.append(raw)
                continue
            parts = []
            for b in blocks:
                t = b.get("type")
                if t == "image":
                    src = b["source"]
                    parts.append(types.Part.from_bytes(data=base64.b64decode(src["data"]), mime_type=src["media_type"]))
                elif t == "text" and b.get("text"):
                    parts.append(types.Part(text=b["text"]))
                elif t == "tool_result":
                    c = b.get("content")
                    txt = c if isinstance(c, str) else " ".join(x.get("text", "") for x in c or [] if isinstance(x, dict))
                    if txt:
                        parts.append(types.Part(text=txt))
                elif t == "tool_use":
                    import json
                    parts.append(types.Part(text=json.dumps(b.get("input", {}), ensure_ascii=False)))
                # thinking blocks from Claude are dropped
            if parts:
                out.append(types.Content(role=role, parts=parts))
        return out

    def _config(self, system: str, schema: dict):
        from google.genai import types
        cfg = dict(system_instruction=system + GEMINI_NOTE, response_mime_type="application/json",
                   response_json_schema=schema, max_output_tokens=32768, temperature=1.0)
        if self.level:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_level=self.level)
        if self.media_high:
            cfg["media_resolution"] = types.MediaResolution.MEDIA_RESOLUTION_HIGH
        return types.GenerateContentConfig(**cfg)

    # ------------------------------------------------------------ calls
    def create(self, messages: list[dict], system: str, schema: dict):
        from google.genai import errors
        contents = self._contents(messages)
        attempt = 0
        while True:
            if self.stopped:
                raise self.stopped
            try:
                resp = self.client.models.generate_content(model=self.model, contents=contents,
                                                           config=self._config(system, schema))
                break
            except errors.ClientError as e:
                msg = str(getattr(e, "message", "") or e)
                if e.code == 429:
                    if _is_daily(e):
                        with self._lock:
                            self.stopped = self.stopped or QuotaExhausted(
                                "The Gemini free-tier daily limit for this model is used up. Everything finished "
                                "so far is saved; resume after the quota resets (midnight Pacific time).")
                        raise self.stopped
                    attempt += 1
                    if attempt > 12:
                        raise
                    wait = _retry_delay(e.details) or min(60, 10 * attempt)
                    self.log(f"  Gemini rate limit reached, waiting {wait:.0f}s ...")
                    time.sleep(wait + 1)
                    continue
                if e.code == 400 and (self.level or self.media_high) and re.search(
                        r"thinking|media_resolution|mediaResolution", msg, re.I):
                    self.log(f"  {self.model} rejected a setting ({msg[:120]}); retrying without it")
                    if re.search(r"media", msg, re.I):
                        self.media_high = False
                    else:
                        self.level = None
                    continue
                if e.code in (401, 403):
                    raise RuntimeError(f"Gemini refused the API key ({e.code}): {msg[:200]}") from e
                raise
            except errors.ServerError:
                attempt += 1
                if attempt > 6:
                    raise
                wait = min(60, 5 * 2 ** (attempt - 1))
                self.log(f"  Gemini is busy, retrying in {wait}s ...")
                time.sleep(wait)
        text = resp.text if getattr(resp, "candidates", None) else None
        if not text:
            reason = None
            try:
                reason = resp.candidates[0].finish_reason
            except Exception:
                reason = getattr(getattr(resp, "prompt_feedback", None), "block_reason", None)
            raise RuntimeError(f"Gemini returned no transcription (finish reason: {reason})")
        raw = None
        try:
            raw = resp.candidates[0].content
        except Exception:
            pass
        um = getattr(resp, "usage_metadata", None)
        cached = getattr(um, "cached_content_token_count", 0) or 0
        usage = SimpleNamespace(
            input_tokens=max(0, (getattr(um, "prompt_token_count", 0) or 0) - cached),
            output_tokens=(getattr(um, "candidates_token_count", 0) or 0) + (getattr(um, "thoughts_token_count", 0) or 0),
            cache_read_input_tokens=cached, cache_creation_input_tokens=0)
        return SimpleNamespace(content=[_Block(text, raw)], usage=usage)
