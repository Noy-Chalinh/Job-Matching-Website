"""
Thin client for the CamHR job API.

Pure HTTP functions only — no file I/O and no Django dependency, so this
module can be unit-tested or reused on its own. Storage lives in
jobs.management.commands.scrape_camhr, which upserts what these functions
return into the RawJob table (Supabase/Postgres).
"""
import requests

BASE_URL = "https://api.camhr.com/v1.0.0"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}


def get_total_pages(page_size=10):
    url = f"{BASE_URL}/jobs/simple/page-query?page=1&size={page_size}&locationId=0&jobTitleOrCompany="
    res = requests.get(url, headers=HEADERS)
    res.raise_for_status()
    return res.json()["data"]["totalPage"]


def get_job_ids(page, page_size=10):
    url = f"{BASE_URL}/jobs/simple/page-query?page={page}&size={page_size}&locationId=0&jobTitleOrCompany="
    res = requests.get(url, headers=HEADERS)
    res.raise_for_status()
    return [job["id"] for job in res.json()["data"]["result"]]


def get_job_raw(job_id):
    """Fetch one job's raw payload. Returns None on a non-200 response."""
    url = f"{BASE_URL}/jobs/{job_id}"
    res = requests.get(url, headers=HEADERS)

    if res.status_code != 200:
        return None

    data = res.json().get("data", {})

    return {
        "pubdate": data.get("pubdate"),
        "expdate": data.get("expdate"),
        "title": data.get("title"),
        "hirelings": data.get("hirelings"),
        "workyears": data.get("workyears"),
        "address": data.get("address"),
        "requirement": data.get("requirement"),
        "description": data.get("description"),
        "jobLangs": data.get("jobLangs"),
        "employer": data.get("employer"),
    }
