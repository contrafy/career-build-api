import json
import os
from io import BytesIO
from typing import Any, Dict, Mapping
import re
import ai

import requests
import PyPDF2

from models import LLMGeneratedFilters

# --------------------------------------------------------------------------- #
#  RapidAPI helpers
# --------------------------------------------------------------------------- #

def _sanitize_params(params: Mapping[str, Any]) -> Dict[str, str]:
    ALLOWED_KEYS = {"advanced_title_filter", "location_filter", "limit", "title_only", "where", "distance", "page", "results_per_page", "country"}

    clean: Dict[str, str] = {}

    # ── 1. basic filtering / stringification ───────────────────────────────
    if not params.get("limit"):
        clean["limit"] = "15" 

    for k, v in params.items():
        if k not in ALLOWED_KEYS:
            continue
        if v is None or v == "":
            continue
        if isinstance(v, bool):
            clean[k] = "true" if v else "false"
        else:
            clean[k] = str(v)

    # ── 2. quote multi-word terms in advanced_title_filter ─────────────────
    raw = clean.get("advanced_title_filter")
    if raw:
        # split on the '|' operator, trim whitespace around each token
        tokens = [t.strip() for t in raw.split("|")]
        quoted: list[str] = []
        for tok in tokens:
            if not tok:
                continue
            # already quoted? leave it; otherwise quote if it contains spaces
            if " " in tok and not (tok.startswith("'") or tok.startswith('"')):
                quoted.append(f"'{tok}'")
            else:
                quoted.append(tok)
        clean["advanced_title_filter"] = "|".join(quoted)

    return clean

def _extract_jobs_list(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for k in ("internships", "jobs", "yc_jobs", "results", "data"):
            if k in payload and isinstance(payload[k], list):
                return payload[k]
    return []                                         # fallback

def _map_adzuna(item: dict) -> dict:
    """
    Convert ONE raw Adzuna record → JobListing interface used in JobCard.tsx
    """
    loc = item.get("location") or {}
    comp = item.get("company") or {}

    return {
        "id":           str(item.get("id")),                 # JobCard.id is str
        "title":        item.get("title"),
        "organization": comp.get("display_name"),
        "locations_derived": [
            loc.get("display_name")
        ] if loc.get("display_name") else [],
        "location_type": None,                               # Adzuna has no flag
        "url":          item.get("redirect_url"),
        "date_posted":  item.get("created"),                 # e.g. "2024-12-01T17:34:00Z"
        "date_created": item.get("created"),
        # "rating" left out – LLM will add later
    }

def _call_api(url: str, host: str, params: Mapping[str, Any]) -> dict:
    headers = {**ai.COMMON_HEADERS, "x-rapidapi-host": host}
    
    query = _sanitize_params(params)
    print("\nQuery about to be sent: ", query, "\n")
    resp = requests.get(url, headers=headers, params=query, timeout=15)
    print("\nResponse from external API: ", resp, "\n")
    resp.raise_for_status()
    return resp.json()

_CAMEL_TO_SNAKE = {
    "title": "title_filter",
    "advancedTitle": "advanced_title_filter",
    "description": "description_filter",
    "location": "location_filter",
}
def _normalise_keys(d: dict[str, Any]) -> dict[str, Any]:
    return {
        _CAMEL_TO_SNAKE.get(k, k): v            # map if known, else keep
        for k, v in d.items()
    }

def _call_adzuna(params: Mapping[str, Any]) -> dict:
    """
    Invoke Adzuna's `/v1/api/jobs/{country}/search/{page}` endpoint.

    Required path params:
        country - ISO-3166 code (default “us”)
        page    - 1-based page index

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

def fetch_internships(params: Mapping[str, Any], resume_pdf: bytes | None = None) -> dict:
    params = _normalise_keys(dict(params))
    payload = _call_api(
        "https://internships-api.p.rapidapi.com/active-jb-7d",
        "internships-api.p.rapidapi.com",
        params,
    )
    resume_txt = _pdf_to_text(resume_pdf) if resume_pdf else None
    _rate_jobs_against_resume(_extract_jobs_list(payload), resume_txt)
    return payload


def fetch_jobs(params: Mapping[str, Any], resume_pdf: bytes | None = None) -> dict:
    params = _normalise_keys(dict(params))
    payload = _call_api(
        "https://active-jobs-db.p.rapidapi.com/active-ats-7d",
        "active-jobs-db.p.rapidapi.com",
        params,
    )
    resume_txt = _pdf_to_text(resume_pdf) if resume_pdf else None
    _rate_jobs_against_resume(_extract_jobs_list(payload), resume_txt)
    return payload


def fetch_yc_jobs(params: Mapping[str, Any], resume_pdf: bytes | None = None) -> dict:
    params = _normalise_keys(dict(params))
    payload = _call_api(
        "https://free-y-combinator-jobs-api.p.rapidapi.com/active-jb-7d",
        "free-y-combinator-jobs-api.p.rapidapi.com",
        params,
    )
    resume_txt = _pdf_to_text(resume_pdf) if resume_pdf else None
    _rate_jobs_against_resume(_extract_jobs_list(payload), resume_txt)
    return payload

def fetch_adzuna_jobs(filters: Mapping[str, Any]) -> dict:
    p: dict[str, Any] = {}

    raw_title = filters.get("advanced_title_filter") or filters.get("title_filter") or ""
    if raw_title:
        cleaned = re.sub(r"[()']", "", str(raw_title)).replace("|", " ").strip()
        if cleaned:
            p["title_only"] = cleaned

    raw_loc = (filters.get("location_filter") or "").strip()
    if raw_loc:
        p["where"] = raw_loc.split(" OR ", 1)[0].strip()

    try:
        p["distance"] = int(filters.get("distance", 50))
    except ValueError:
        p["distance"] = 50

    limit = int(filters.get("limit", 50))    
    p["results_per_page"] = limit            
    p["page"] = 1                            

    raw = _call_adzuna(p)
    mapped = [_map_adzuna(r) for r in raw.get("results", [])]
    return {"results": mapped}


# --------------------------------------------------------------------------- #
#  Convert PDF documents (resume/CV) to plaintext for LLM ingestion
# --------------------------------------------------------------------------- #
def _pdf_to_text(pdf_bytes: bytes) -> str:
    """Return plaintext extracted from a PDF."""
    reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages)

# --------------------------------------------------------------------------- #
# ❷  call an LLM to score every job 0.0‑10.0 and print the result
# --------------------------------------------------------------------------- #
def _rate_jobs_against_resume(jobs: list[dict], resume_text: str | None = None):
    if not jobs:                                     # nothing to do
        print("\nno jobs?\n")
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
        # print("resume_text:", resume_text[:8000])
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

        # ---------- attach ratings to each listing ----------
        for j in jobs:
            jid = j.get("id")
            if jid and jid in ratings:
                j["rating"] = float(ratings[jid])

        print("Job-fit ratings:", ratings)           # <-- for now just log
    except Exception as e:
        print("rating LLM call failed:", e)