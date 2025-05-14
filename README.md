# career-build-api

FastAPI wrapper for querying job and internship data from multiple RapidAPI endpoints.

## Endpoints

- `GET /fetch_internships` – Internships API
- `GET /fetch_jobs` – Active jobs (ATS feeds)
- `GET /fetch_yc_jobs` – YC startup jobs
- `GET /` – Basic health check

Query parameters are passed directly through to the respective APIs. See the RapidAPI docs for accepted keys.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your RAPIDAPI_KEY
uvicorn main:app --reload