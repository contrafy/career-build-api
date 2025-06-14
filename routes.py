from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from pydantic import BaseModel

import helpers
import ai

import json
from models import LLMGeneratedFilters, JobFilters
import re
from typing import Any, Mapping

router = APIRouter()

class SearchRequest(BaseModel):
    filters: JobFilters
    resumeText: str | None = None   # plain résumé text (can be null)

@router.post("/fetch_internships")
async def fetch_internships(
    filters: str = Form(...),
    resume: UploadFile | None = File(None)
):
    try:
        filters_obj = json.loads(filters)
        print("\nFrontend Sent: ", filters_obj)
        pdf_bytes = await resume.read() if resume else None
        return helpers.fetch_internships(filters_obj, pdf_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/fetch_jobs")
async def fetch_jobs(
    filters: str = Form(...),
    resume: UploadFile | None = File(None)
):
    try:
        filters_obj = json.loads(filters)
        print("\nFrontend Sent: ", filters_obj)
        pdf_bytes = await resume.read() if resume else None
        return helpers.fetch_jobs(filters_obj, pdf_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/fetch_yc_jobs")
async def fetch_yc_jobs(
    filters: str = Form(...),
    resume: UploadFile | None = File(None)
):
    try:
        filters_obj = json.loads(filters)
        print("\nFrontend Sent: ", filters_obj)
        pdf_bytes = await resume.read() if resume else None
        return helpers.fetch_yc_jobs(filters_obj, pdf_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    
@router.get("/fetch_adzuna_jobs")
async def fetch_adzuna_jobs_route(request: Request):
    try:
        print(request.query_params)
        return helpers.fetch_adzuna_jobs(dict(request.query_params))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))



@router.post("/test_llm_resume_parsing", response_model=LLMGeneratedFilters)
async def test_llm_resume_parsing(resume: UploadFile = File(...)):
    """
    Upload a PDF résumé, receive JSON filters derived by GPT-4.
    """
    if resume.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF resumes are supported")

    try:
        pdf_bytes = await resume.read()
        filters = ai.generate_filters_from_resume(pdf_bytes)
        if not filters:
            raise HTTPException(status_code=400, detail="No filters generated from the résumé")
        
        print("Generated filters:", filters)

        return filters
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))