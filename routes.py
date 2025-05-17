from fastapi import APIRouter, HTTPException, Request, UploadFile, File

import helpers
from models import LLMGeneratedFilters

router = APIRouter()


@router.get("/fetch_internships")
async def fetch_internships(request: Request):
    try:
        return helpers.fetch_internships(dict(request.query_params))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/fetch_jobs")
async def fetch_jobs(request: Request):
    try:
        return helpers.fetch_jobs(dict(request.query_params))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/fetch_yc_jobs")
async def fetch_yc_jobs(request: Request):
    try:
        return helpers.fetch_yc_jobs(dict(request.query_params))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/test_llm_resume_parsing", response_model=LLMGeneratedFilters)
async def test_llm_resume_parsing(resume: UploadFile = File(...)):
    """
    Upload a PDF résumé, receive JSON filters derived by GPT‑4.
    """
    if resume.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF resumes are supported")

    try:
        pdf_bytes = await resume.read()
        return helpers.generate_filters_from_resume(pdf_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))