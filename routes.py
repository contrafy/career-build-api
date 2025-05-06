from fastapi import APIRouter, HTTPException, Request

import helpers

router = APIRouter()


@router.get("/fetch_internships")
async def fetch_internships(request: Request):
    """
    Pass *all* query‑string keys straight through to the internships endpoint.
    Every parameter in the RapidAPI docs is accepted.
    """
    try:
        return helpers.fetch_internships(dict(request.query_params))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/fetch_jobs")
async def fetch_jobs(request: Request):
    """
    Forward any query params to the active‑jobs endpoint (ATS feeds).
    Accepts every param listed in the docs.
    """
    try:
        return helpers.fetch_jobs(dict(request.query_params))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/fetch_yc_jobs")
async def fetch_yc_jobs(request: Request):
    """
    Forward any query params to the YC‑Jobs endpoint.
    """
    try:
        return helpers.fetch_yc_jobs(dict(request.query_params))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
