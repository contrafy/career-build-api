from typing import Optional

from fastapi import APIRouter, HTTPException, Query

import helpers

router = APIRouter()


@router.get("/fetch_internships")
def fetch_internships(
    title: Optional[str] = Query(None, alias="title"),
    location: Optional[str] = Query(None, alias="location"),
    description: Optional[str] = Query(None, alias="description"),
):
    """
    Forward query params to RapidAPI internships endpoint.
    All params are optional.
    """
    try:
        return helpers.fetch_internships(
            title_filter=title,
            location_filter=location,
            description_filter=description,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/fetch_jobs")
def fetch_jobs(
    title: Optional[str] = Query(None, alias="title"),
    location: Optional[str] = Query(None, alias="location"),
    limit: Optional[int] = None,
    offset: Optional[int] = None,
):
    """
    Forward query params to RapidAPI active‑jobs endpoint.
    All params are optional.
    """
    try:
        return helpers.fetch_jobs(
            title_filter=title,
            location_filter=location,
            limit=limit,
            offset=offset,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
