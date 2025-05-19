from typing import Optional, Dict

from pydantic import BaseModel, Field


# ---------- Job / Internship / YC filter models ------------------------------

class _BaseFilters(BaseModel):
    # common to all three APIs
    title_filter:        Optional[str] = None
    advanced_title_filter: Optional[str] = None
    location_filter:     Optional[str] = None
    description_filter:  Optional[str] = None
    description_type:    Optional[str] = None
    remote:              Optional[bool] = Field(default=None, description="true/false")
    offset:              Optional[int]  = None
    date_filter:         Optional[str]  = None

    # carries the client’s résumé cache key (never forwarded upstream)
    # resume_id:           Optional[str] = None

    def as_query(self) -> Dict[str, str | int]:
        """Return only the fields we want to pass to RapidAPI."""
        d = self.dict(exclude_none=True)
        d.pop("resume_id", None)                  # strip it right here
        return d


class InternshipFilters(_BaseFilters):
    agency:               Optional[bool] = None
    include_ai:           Optional[bool] = None
    ai_work_arrangement_filter: Optional[str] = None


class JobFilters(_BaseFilters):
    limit:                Optional[int]  = None
    organization_filter:  Optional[str]  = None
    source:               Optional[str]  = None
    include_ai:           Optional[bool] = None
    ai_employment_type_filter:  Optional[str] = None
    ai_work_arrangement_filter: Optional[str] = None
    ai_has_salary:            Optional[bool] = None
    ai_experience_level_filter: Optional[str] = None
    ai_visa_sponsorship_filter: Optional[bool] = None
    include_li:               Optional[bool] = None
    li_organization_slug_filter: Optional[str] = None
    li_organization_slug_exclusion_filter: Optional[str] = None
    li_industry_filter:       Optional[str] = None
    li_organization_specialties_filter: Optional[str] = None
    li_organization_description_filter: Optional[str] = None


class YcFilters(_BaseFilters):
    pass  # currently no additional unique params


# ---------- LLM response model -----------------------------------------------

class LLMGeneratedFilters(BaseModel):
    """
    Container that may hold subsets for internships / jobs / yc.
    Keys absent = param not detected / not relevant.
    """
    internships: Optional[InternshipFilters] = None
    jobs:        Optional[JobFilters]        = None
    yc_jobs:     Optional[YcFilters]         = None
