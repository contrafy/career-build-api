import json
import os
from io import BytesIO
from typing import Any, Dict, Mapping

import openai
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

if not RAPIDAPI_KEY:
    raise RuntimeError("RAPIDAPI_KEY missing in environment/.env")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing in environment/.env")

openai.api_key = OPENAI_API_KEY
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


def _call_api(url: str, host: str, params: Mapping[str, Any]) -> dict:
    headers = {**COMMON_HEADERS, "x-rapidapi-host": host}
    query = _sanitize_params(params)
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
You are an API filter generator. Your job is to analyse résumés
and output **ONLY** a JSON object with the most specific filters you can derive,
using the keys listed below. If a filter cannot be inferred with high
confidence, leave it out completely. Never guess.

Valid JSON keys (duplicate keys across APIs appear only once):
    title_filter
    advanced_title_filter
    location_filter
    description_filter
    description_type
    remote                (true | false)
    agency                (true | false)
    include_ai            (true | false)
    ai_work_arrangement_filter
    limit                 (10‑100)
    offset
    organization_filter
    source
    ai_employment_type_filter
    ai_has_salary         (true | false)
    ai_experience_level_filter
    ai_visa_sponsorship_filter
    include_li            (true | false)
    li_organization_slug_filter
    li_organization_slug_exclusion_filter
    li_industry_filter
    li_organization_specialties_filter
    li_organization_description_filter
    date_filter

Output format **strictly**:
```json
{
  "<key>": <value>,          # include only when value is known
  "...": "..."
}
```
No explanations, no additional keys.
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
    resume_text = _pdf_to_text(pdf_bytes)
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": _build_resume_prompt(resume_text)},
        ],
        temperature=0.1,
    )

    content = response.choices[0].message.content.strip()
    # Ensure pure JSON
    json_str = content.split("```json")[-1].split("```")[0] if "```" in content else content
    raw_filters = json.loads(json_str)

    # Fan‑out into each API model
    internship_f = InternshipFilters(**raw_filters).dict(exclude_none=True)
    job_f        = JobFilters(**raw_filters).dict(exclude_none=True)
    yc_f         = YcFilters(**raw_filters).dict(exclude_none=True)

    return LLMGeneratedFilters(
        internships=internship_f or None,
        jobs=job_f or None,
        yc_jobs=yc_f or None,
    )
