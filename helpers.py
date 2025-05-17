import json
import os
from io import BytesIO
from typing import Any, Dict, Mapping

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
using the keys listed below.

Valid JSON keys:
    advanced_title_filter         
    location_filter       

API documentation for these query parameters:

advanced_title_filter
String

Advanced Title filter which enables more features like parenthesis, 'AND', and prefix searching.

Phrares (two words or more) always need to be single quoted or use the operator <->

Instead of using natural language like 'OR' you need to use operators like:

    & (AND)
    | (OR)
    ! (NOT)
    <-> (FOLLOWED BY)
    ' ' (FOLLOWED BY alternative, does not work with 6. Prefix Wildcard)
    :* (Prefix Wildcard)

For example:

(AI | 'Machine Learning' | 'Robotics') & ! Marketing

Will return all jobs with ai, or machine learning, or robotics in the title except titles with marketing

Project <-> Manag:*

Will return jobs like Project Manager or Project Management

Your goal when crafting a advanced_title_filter based on a resume is to MAXIMIZE the breadth of jobs that the API returns while still remaining specific to their skills and experience.
This means you should ensure, for example with a CS heavy resume, that (software engineering OR software) and optionally based on the resume (frontend OR backend etc.) are added in an all encompassing way using OR's and splitting keywords,
while still ensuring that 'engineer' is not independently used as a keyword to avoid irrelevant non-CS engineering jobs such as mechanical engineering. Complexity in this query is fine if needed in order to avoid irrelevant jobs or roles unlikely to be a good fit (no need to include backend if the resume is unapologetically geared towards frontend dev for example).

Location_filter
String

Filter on location. Please do not search on abbreviations like US, UK, NYC. Instead, search on full names like United States, New York, United Kingdom.

You may filter on more than one location in a single API call using the OR parameter. For example: Dubai OR Netherlands OR Belgium

With this one, you should avoid limiting to a specific city and default to the state if known, or country if not. If options are specified on the resume, use them.

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
    # Convert PDF resume to text for LLM ingestion
    resume_text = _pdf_to_text(pdf_bytes)

    # Use Groq to generate filters
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful job hunting assistant, the goal is to maximize the breadth of jobs that the user can and should apply to, "
                                          "while also giving them the jobs they are most likely to desire and do well at from the information available to you."},
            {"role": "user", "content": _build_resume_prompt(resume_text)},
        ],
        temperature=0.5,
    )
    content = response.choices[0].message.content.strip()
    if not content:
        raise ValueError("Empty response from LLM")

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
