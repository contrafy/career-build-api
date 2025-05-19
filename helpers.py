import json
import os
from io import BytesIO
from typing import Any, Dict, Mapping
import re
import uuid

import openai
from groq import Groq

import requests
import PyPDF2
from dotenv import load_dotenv

from models import (
    InternshipFilters,
    JobFilters,
    YcFilters,
    LLMGeneratedFilters,
)

# --------------------------------------------------------------------------- #
#  ENV / CONSTANTS
# --------------------------------------------------------------------------- #
load_dotenv()
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Check for RapidAPI key and at least one LLM key
if not RAPIDAPI_KEY:
    raise RuntimeError("RAPIDAPI_KEY missing in environment/.env")
if not (OPENAI_API_KEY or GROQ_API_KEY):
    raise RuntimeError("OPENAI_API_KEY (or GROQ_API_KEY) missing in environment/.env")

# Initialize OpenAI and Groq clients
openai.api_key = OPENAI_API_KEY
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),  # This is the default and can be omitted
)

COMMON_HEADERS = {"x-rapidapi-key": RAPIDAPI_KEY}

# ─── global in‑mem store (top of file, after COMMON_HEADERS) ───────────────
# RESUME_CACHE: dict[str, str] = {}     # resume_id → plaintext résumé


# --------------------------------------------------------------------------- #
#  RapidAPI helpers
# --------------------------------------------------------------------------- #

def _sanitize_params(params: Mapping[str, Any]) -> Dict[str, str]:
    clean: Dict[str, str] = {}
    for k, v in params.items():
        if k == "resume_id":
            continue
        if v is None or v == "":
            continue
        if isinstance(v, bool):
            clean[k] = "true" if v else "false"
        else:
            clean[k] = str(v)
    return clean

# --------------------------------------------------------------------------- #
# ❶  small util: pull the list out of any RapidAPI payload shape
# --------------------------------------------------------------------------- #
def _extract_jobs_list(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for k in ("internships", "jobs", "yc_jobs", "results", "data"):
            if k in payload and isinstance(payload[k], list):
                return payload[k]
    return []                                         # fallback

# _DELIMS = re.compile(r"(\(|\)|\||<->|!)")

# ─── helper to pop resume_id + convert to text ─────────────────────────────
"""
def _pull_resume(params: Mapping[str, Any]) -> str | None:
    
    Remove resume_id from the params dict (so it never reaches RapidAPI)
    and return the cached plaintext (if we have it).
    
    if isinstance(params, dict):
        resume_id = params.pop("resume_id", None)
        print("DEBUG params incoming:", params)
        if resume_id:
            return RESUME_CACHE.get(resume_id)
    return None
"""
# unused for now
def _quote_advanced_terms(expr: str) -> str:
    """
    ('foo bar' | baz | C++)  ->  ('foo bar' | 'baz' | 'C++')
    Parentheses and operators remain in their original positions.
    """
    out: list[str] = []
    for chunk in _DELIMS.split(expr):              # yields delimiters *and* gaps
        if chunk in {"(", ")", "|", "<->", "!"}:
            out.append(chunk)                      # keep delimiters as‑is
        else:
            stripped = chunk.strip()
            if not stripped:                       # pure whitespace ⇒ preserve
                out.append(chunk)
            elif stripped[0] in {"'", '"'}:        # already quoted
                out.append(stripped)
            else:                                  # wrap bare term
                out.append(f"'{stripped}'")
    return "".join(out)                            # join w/out inserting extras

def _call_api(url: str, host: str, params: Mapping[str, Any]) -> dict:
    headers = {**COMMON_HEADERS, "x-rapidapi-host": host}
    query = _sanitize_params(params)
    resp = requests.get(url, headers=headers, params=query, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_internships(params: Mapping[str, Any], resume_text: str | None = None) -> dict:
    payload = _call_api(
        "https://internships-api.p.rapidapi.com/active-jb-7d",
        "internships-api.p.rapidapi.com",
        params,
    )
    _rate_jobs_against_resume(_extract_jobs_list(payload), resume_text)
    return payload


def fetch_jobs(params: Mapping[str, Any], resume_text: str | None = None) -> dict:
    payload = _call_api(
        "https://active-jobs-db.p.rapidapi.com/active-ats-7d",
        "active-jobs-db.p.rapidapi.com",
        params,
    )
    _rate_jobs_against_resume(_extract_jobs_list(payload), resume_text)
    return payload


def fetch_yc_jobs(params: Mapping[str, Any], resume_text: str | None = None) -> dict:
    payload = _call_api(
        "https://free-y-combinator-jobs-api.p.rapidapi.com/active-jb-7d",
        "free-y-combinator-jobs-api.p.rapidapi.com",
        params,
    )
    _rate_jobs_against_resume(_extract_jobs_list(payload), resume_text)
    return payload


# --------------------------------------------------------------------------- #
#  Convert PDF documents (resume/CV) to plaintext for LLM ingestion
# --------------------------------------------------------------------------- #
def _pdf_to_text(pdf_bytes: bytes) -> str:
    """Return plaintext extracted from a PDF."""
    reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages)


# --------------------------------------------------------------------------- #
#  LLM prompts
# --------------------------------------------------------------------------- #
_FILTER_DOC = """
You are an **API filter generator**.  
Analyse the résumé and output **only** a JSON object using the keys below.

Valid JSON keys  
    advanced_title_filter    ← a single, gigantic OR-clause  
    location_filter

──────────────────────────────────────────────────────────────────────────────
advanced_title_filter  (STRING)
──────────────────────────────────────────────────────────────────────────────
• Build **one** parenthesised clause that contains **only**:  
  → the pipe operator `|` (OR)  
  → the keywords / phrases you extracted from the résumé  
  *No other operators* (`&`, `!`, `<->`, `:*`, etc.) may appear.

• **Wrap every term in single quotes**, even single-word terms, e.g.  
  `'Python' | 'Machine Learning' | 'C++'`

• Example of a valid output:  
  `('Software Engineer' | 'DevOps' | 'C++' | 'Kubernetes' | 'Cloud Computing')`

• Your goal is to **maximise breadth** while staying relevant to the candidate's
  skills.  Split multi-word concepts into their shortest useful forms and
  include both singular & broader synonyms where appropriate.

──────────────────────────────────────────────────────────────────────────────
location_filter  (STRING)
──────────────────────────────────────────────────────────────────────────────
• Use full names only (“United States”, “New York”, “United Kingdom”).  
• Combine multiple locations with **OR**, e.g.  
  `United States OR Canada OR Netherlands`  
• Prefer state-level granularity unless the résumé clearly specifies
  a city. Avoid countries unless the resume or other context mentions openness to multiple countries.

──────────────────────────────────────────────────────────────────────────────
OUTPUT  (STRICT)
──────────────────────────────────────────────────────────────────────────────
```json
{
  "advanced_title_filter": "(...)",
  "location_filter": "..."
}
```

Include a key only if you have a value for it.
No comments, no additional keys, no markdown fences outside the JSON block.
"""

def _build_resume_prompt(resume_text: str) -> str:
    return (
    _FILTER_DOC
    + "\n---\nRESUME:\n"
    + resume_text[:7000] # protect token budget
    + "\n---\nGenerate JSON now:"
    )

# ---------------------------------------------------------------------------
# Public: generate filters with OpenAI
# ---------------------------------------------------------------------------

def generate_filters_from_resume(pdf_bytes: bytes) -> LLMGeneratedFilters:
    # Convert PDF resume to text for LLM ingestion
    resume_text = _pdf_to_text(pdf_bytes)

    # cache and emit a UUID so future search requests can reference it
    # resume_id = str(uuid.uuid4())
    # RESUME_CACHE[resume_id] = resume_text

    # Use Groq to generate filters
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful job hunting assistant, the goal is to maximize the breadth of jobs that the user can and should apply to, "
                                          "while also giving them the jobs they are most likely to desire and do well at from the information available to you."},
            {"role": "user", "content": _build_resume_prompt(resume_text)},
        ],
        temperature=0.2,
    )
    content = response.choices[0].message.content.strip()
    if not content:
        raise ValueError("Empty response from LLM")

    # Ensure pure JSON
    json_str = content.split("```json")[-1].split("```")[0] if "```" in content else content
    raw_filters = json.loads(json_str)

    # Wrap terms in single quotes to appease the postgres gods
    """
    atf = raw_filters.get("advanced_title_filter")
    if isinstance(atf, str):
        raw_filters["advanced_title_filter"] = _quote_advanced_terms(atf)
    """

    # Fan‑out into each API model
    internship_f = InternshipFilters(**raw_filters).dict(exclude_none=True)
    job_f        = JobFilters(**raw_filters).dict(exclude_none=True)
    yc_f         = YcFilters(**raw_filters).dict(exclude_none=True)

    result = LLMGeneratedFilters(
        internships=internship_f or None,
        jobs=job_f or None,
        yc_jobs=yc_f or None,
    ).dict(exclude_none=True)
    result["resumeText"] = resume_text
    return result

# --------------------------------------------------------------------------- #
# ❷  call an LLM to score every job 0.0‑10.0 and print the result
# --------------------------------------------------------------------------- #
def _rate_jobs_against_resume(jobs: list[dict], resume_text: str | None = None):
    if not jobs:                                     # nothing to do
        return

    # keep only fields that help the LLM
    shortlist = [
        {
            "id": j.get("id"),
            "date_posted": j.get("date_posted"),
            "title": j.get("title"),
            "organization": j.get("organization"),
            "description_text": j.get("description_text"),
        }
        for j in jobs
    ]

    system_msg = (
        "You are a career-match assistant.\n"
        "Rate each job 0.0-10.0 (exactly one decimal place) for how well it fits the "
        "candidate's résumé.  Return ONLY a JSON object whose keys are the "
        "job IDs and whose values are the ratings.  No other text."
    )

    user_parts = []
    if resume_text:
        print("resume_text:", resume_text[:1000])
        user_parts.append("Résumé:\n" + resume_text[:8000])
    user_parts.append("Job listings JSON:\n" + json.dumps(shortlist, ensure_ascii=False))
    user_msg = "\n\n".join(user_parts)

    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",    "content": user_msg},
            ],
            temperature=0.5,
        )
        raw = resp.choices[0].message.content.strip()
        json_str = raw.split("```json")[-1].split("```")[0] if "```" in raw else raw
        ratings = json.loads(json_str)
        print("Job-fit ratings:", ratings)           # <-- for now just log
    except Exception as e:
        print("rating LLM call failed:", e)