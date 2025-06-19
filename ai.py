from dotenv import load_dotenv
import os, json

import helpers
from models import LLMGeneratedFilters

import openai
from groq import Groq

load_dotenv()
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ADZUNA_APP_ID  = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")
if not (ADZUNA_APP_ID and ADZUNA_APP_KEY):
    raise RuntimeError("ADZUNA_APP_ID / ADZUNA_APP_KEY missing in environment/.env")

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
#  LLM prompts
# --------------------------------------------------------------------------- #
_FILTER_DOC = """
You are an **Job API filter generator**.  
Analyse the résumé and output **only** a JSON object using the keys below.

Valid JSON keys  
    advanced_title_filter    ← a single comma-separated string of relevant titles/skills  
    location_filter          ← a single comma-separated string of relevant locations

──────────────────────────────────────────────────────────────────────────────
advanced_title_filter  (STRING - comma-separated)
──────────────────────────────────────────────────────────────────────────────
• Extract relevant titles, skills, and technologies from the résumé.
• Return them as **one comma-separated list**, e.g.  
  `Systems Administrator, Linux Administrator, DevOps Engineer, C++, Python`

• DO NOT wrap terms in quotes.
• Maximize breadth while staying relevant to the candidate's skills.
• Include both specific and general forms (e.g., "Kubernetes", "Cloud Engineer").

──────────────────────────────────────────────────────────────────────────────
location_filter  (STRING - comma-separated)
──────────────────────────────────────────────────────────────────────────────
• Use full names only (“United States”, “New York”, “United Kingdom”).  
• Combine multiple locations with **commas**, e.g.  
  `United States, Canada, Netherlands`
• Only go down to COUNTRY level granularity based on where the resume says they wish to be or are currently located explicitly

──────────────────────────────────────────────────────────────────────────────
OUTPUT  (STRICT)
──────────────────────────────────────────────────────────────────────────────
{"advanced_title_filter": "...", "location_filter": "..."}
"""

def _build_resume_prompt(resume_text: str) -> str:
    return (
    _FILTER_DOC
    + "\n---\nRESUME:\n"
    + resume_text[:7000] # protect token budget
    + "\n---\nGenerate JSON now:"
    )

def generate_filters_from_resume(pdf_bytes: bytes) -> LLMGeneratedFilters:
    # Convert PDF resume to text for LLM ingestion
    resume_text = helpers._pdf_to_text(pdf_bytes)


    # Use Groq to generate filters
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful job hunting assistant, the goal is to maximize the breadth of jobs that the user can and should apply to, "
                                          "while also giving them the jobs they are most likely to desire and do well at from the information available to you."},
            {"role": "user", "content": _build_resume_prompt(resume_text)},
        ],
        temperature=0.2,
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content.strip()
    print("LLM response:", content)  # <-- for debugging
    if not content:
        raise ValueError("Empty response from LLM")

    # Ensure pure JSON
    json_str = content.split("```json")[-1].split("```")[0] if "```" in content else content

    # Groq occasionally emits un‑escaped \n / \r inside string literals.
    # json.loads(strict=False) tolerates those; if it still fails, strip them.
    try:
        raw_filters = json.loads(json_str, strict=False)
    except json.JSONDecodeError:
        cleaned = json_str.replace("\r", " ").replace("\n", " ")
        raw_filters = json.loads(cleaned, strict=False)


    # Validate and return only the two flat filters needed by the UI
    return LLMGeneratedFilters(**raw_filters).dict(exclude_none=True)

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
        "Rate each job 0.0-10.0 (exactly one decimal place, use whole values sparingly) for how well it fits the "
        "candidate's résumé.  Return ONLY a JSON object whose keys are the "
        "job IDs and whose values are the ratings.  No other text. When rating how well a certain job fits, ensure to place a heavy emphasis on making sure the amount of experience required is a match or close match to the experience that you"
        " can gather from the resume, for instance someone with 1-2 years of experience would likely be a poor (<5) fit for a Senior level role, and vice versa for someone with 10-12 years of relevant experience against an entry level job listing."
        " Be sure to also consider the relevance to their resume and specific skills that appear in both the resume and the posting/description."
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