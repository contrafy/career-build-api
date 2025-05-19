from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from pydantic import BaseModel

import helpers
from models import LLMGeneratedFilters, JobFilters

router = APIRouter()

class SearchRequest(BaseModel):
    filters: JobFilters
    resumeText: str | None = None   # plain résumé text (can be null)

@router.post("/fetch_internships")
async def fetch_internships(req: SearchRequest):
    try:
        print(req)
        data = helpers.fetch_internships(req.filters.as_query(), req.resumeText)
        return data
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/fetch_jobs")
async def fetch_jobs(req: SearchRequest):
    try:
        print(req)
        data = helpers.fetch_jobs(req.filters.as_query(), req.resumeText)
        return data
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/fetch_yc_jobs")
async def fetch_yc_jobs(req: SearchRequest):
    try:
        print(req)
        return helpers.fetch_yc_jobs(req.filters.as_query(), req.resumeText)
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
        filters = helpers.generate_filters_from_resume(pdf_bytes)
        if not filters:
            raise HTTPException(status_code=400, detail="No filters generated from the résumé")
        
        print("Generated filters:", filters)

        return filters
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))