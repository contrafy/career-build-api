import json
import os
from io import BytesIO
from typing import Any, Dict, Mapping
import re

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
ADZUNA_APP_ID  = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")
if not (ADZUNA_APP_ID and ADZUNA_APP_KEY):
    raise RuntimeError("ADZUNA_APP_ID / ADZUNA_APP_KEY missing in environment/.env")

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


# --------------------------------------------------------------------------- #
#  RapidAPI helpers
# --------------------------------------------------------------------------- #
def _sanitize_params(params: Mapping[str, Any]) -> Dict[str, str]:
    clean: Dict[str, str] = {}
    for k, v in params.items():
        if v is None or v == "":
            continue
        if isinstance(v, bool):
            clean[k] = "true" if v else "false"
        else:
            clean[k] = str(v)
    return clean

_DELIMS = re.compile(r"(\(|\)|\||<->|!)")          # we keep these untouched

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

def _call_adzuna(params: Mapping[str, Any]) -> dict:
    """
    Invoke Adzuna’s `/v1/api/jobs/{country}/search/{page}` endpoint.

    Required path params:
        country – ISO-3166 code (default “us”)
        page    – 1-based page index

    All other search options go in the query-string.
    """
    # ----------------------------- path parts ------------------------------
    country = str(params.pop("country", "us")).lower()
    page    = int(params.pop("page",    1))                # defaults to 1

    url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"

    # ----------------------------- query params ---------------------------
    query             = _sanitize_params(params)
    query["app_id"]   = ADZUNA_APP_ID
    query["app_key"]  = ADZUNA_APP_KEY

    # NB: Adzuna returns 403 if a User-Agent is not present.
    headers = {"User-Agent": "career-builder/1.0"}

    resp = requests.get(url, headers=headers, params=query, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_internships(params: Mapping[str, Any]) -> dict:
    return _call_api(
        "https://internships-api.p.rapidapi.com/active-jb-7d",
        "internships-api.p.rapidapi.com",
        params,
    )


def fetch_jobs(params: Mapping[str, Any]) -> dict:
    return _call_api(
        "https://active-jobs-db.p.rapidapi.com/active-ats-7d",
        "active-jobs-db.p.rapidapi.com",
        params,
    )


def fetch_yc_jobs(params: Mapping[str, Any]) -> dict:
    return _call_api(
        "https://free-y-combinator-jobs-api.p.rapidapi.com/active-jb-7d",
        "free-y-combinator-jobs-api.p.rapidapi.com",
        params,
    )

def fetch_adzuna_jobs(filters: Mapping[str, Any]) -> dict:
    p: dict[str, Any] = {}

    raw_title = filters.get("advanced_title_filter") or filters.get("title_filter") or ""
    if raw_title:
        cleaned = re.sub(r"[()']", "", str(raw_title)).replace("|", " ").strip()
        if cleaned:
            p["title_only"] = cleaned

    raw_loc = (filters.get("location_filter") or "").strip()
    if raw_loc:
        p["where"] = raw_loc.split(" OR ", 1)[0].strip()   # first keyword

    try:
        p["distance"] = int(filters.get("distance", 50))   # override or default
    except ValueError:
        p["distance"] = 50

    limit  = int(filters.get("limit", 50))
    offset = int(filters.get("offset", 0))
    p["results_per_page"] = limit
    p["page"] = offset // limit + 1

    return _call_adzuna(p)

# --------------------------------------------------------------------------- #
#  Résumé → plaintext
# --------------------------------------------------------------------------- #
def _pdf_to_text(pdf_bytes: bytes) -> str:
    """Return plaintext extracted from a PDF."""
    reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages)


# --------------------------------------------------------------------------- #
#  LLM prompt engineering
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
• Prefer state or country-level granularity unless the résumé clearly specifies
  a city. However, ensure to place the more granular location(s) first in the query to maintain relevance, 
  eg. return 'Michigan OR United States' instead of 'United States OR Michigan'.

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

    return LLMGeneratedFilters(
        internships=internship_f or None,
        jobs=job_f or None,
        yc_jobs=yc_f or None,
    )
