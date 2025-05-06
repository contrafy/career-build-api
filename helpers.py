import os
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

load_dotenv()
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

if not RAPIDAPI_KEY:
    raise RuntimeError("RAPIDAPI_KEY missing in environment or .env file")

COMMON_HEADERS = {"x-rapidapi-key": RAPIDAPI_KEY}


def fetch_internships(
    *,
    title_filter: Optional[str] = None,
    location_filter: Optional[str] = None,
    description_filter: Optional[str] = None,
    description_type: str = "text",
) -> dict[str, Any]:
    url = "https://internships-api.p.rapidapi.com/active-jb-7d"
    headers = {**COMMON_HEADERS, "x-rapidapi-host": "internships-api.p.rapidapi.com"}

    params: Dict[str, str] = {"description_type": description_type}
    if title_filter:
        params["title_filter"] = title_filter
    if location_filter:
        params["location_filter"] = location_filter
    if description_filter:
        params["description_filter"] = description_filter

    resp = requests.get(url, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_jobs(
    *,
    title_filter: Optional[str] = None,
    location_filter: Optional[str] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    description_type: str = "text",
) -> dict[str, Any]:
    url = "https://active-jobs-db.p.rapidapi.com/active-ats-7d"
    headers = {**COMMON_HEADERS, "x-rapidapi-host": "active-jobs-db.p.rapidapi.com"}

    params: Dict[str, str] = {"description_type": description_type}
    if title_filter:
        params["title_filter"] = title_filter
    if location_filter:
        params["location_filter"] = location_filter
    if limit is not None:
        params["limit"] = str(limit)
    if offset is not None:
        params["offset"] = str(offset)

    resp = requests.get(url, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()
