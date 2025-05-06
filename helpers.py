import os
from typing import Any, Dict, Mapping

import requests
from dotenv import load_dotenv

load_dotenv()
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
if not RAPIDAPI_KEY:
    raise RuntimeError("RAPIDAPI_KEY missing in environment/.env")

COMMON_HEADERS = {"x-rapidapi-key": RAPIDAPI_KEY}


# ---------- internal helpers -------------------------------------------------


def _sanitize_params(params: Mapping[str, Any]) -> Dict[str, str]:
    """
    • Drop keys with empty value
    • Convert bool → 'true'/'false'
    • Convert everything else to str
    """
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


# ---------- public wrappers ---------------------------------------------------


def fetch_internships(params: Mapping[str, Any]) -> dict:
    """
    RapidAPI: internships‑api  (active‑jb‑7d)
    Pass any documented param through.
    """
    return _call_api(
        "https://internships-api.p.rapidapi.com/active-jb-7d",
        "internships-api.p.rapidapi.com",
        params,
    )


def fetch_jobs(params: Mapping[str, Any]) -> dict:
    """
    RapidAPI: active‑jobs‑db  (active‑ats‑7d)
    """
    return _call_api(
        "https://active-jobs-db.p.rapidapi.com/active-ats-7d",
        "active-jobs-db.p.rapidapi.com",
        params,
    )


def fetch_yc_jobs(params: Mapping[str, Any]) -> dict:
    """
    RapidAPI: free‑y‑combinator‑jobs‑api  (active‑jb‑7d)
    """
    return _call_api(
        "https://free-y-combinator-jobs-api.p.rapidapi.com/active-jb-7d",
        "free-y-combinator-jobs-api.p.rapidapi.com",
        params,
    )
