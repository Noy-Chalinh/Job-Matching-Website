"""
Pure normalization logic for a single CamHR raw job payload.

No file I/O and no Django dependency, so this can be reused by the
`normalize_camhr` management command (DB -> DB) and tested standalone.
"""


def extract_languages(job_langs):
    """Extract language requirements from a jobLangs field."""
    results = []
    for jl in job_langs or []:
        try:
            results.append({
                "name": jl["languageId"]["label"].lower(),
                "level": jl["languageLevelId"]["label"].lower()
            })
        except (KeyError, TypeError):
            continue
    return results


def extract_location(raw):
    """Extract location from a raw job payload."""
    try:
        employer = raw.get("employer", {})
        loc = employer.get("locationId", {}).get("label")
        return loc.lower() if loc else None
    except (AttributeError, TypeError):
        return None


def normalize_job(job_id, raw, skills_extractor, education_extractor):
    """Turn one raw CamHR payload into the normalized job dict.

    `raw` is the dict stored in RawJob.raw_data (or the legacy
    camhr_raw_*.json "raw" field) — the fields returned by
    camhr_client.get_job_raw.
    """
    combined_text = " ".join([
        raw.get("requirement") or "",
        raw.get("description") or ""
    ])

    skills = skills_extractor.extract(combined_text)
    education = education_extractor.extract(raw.get("requirement") or "")
    languages = extract_languages(raw.get("jobLangs"))
    location = extract_location(raw)
    company = raw.get("employer", {}).get("company")
    industry = raw.get("employer", {}).get("industrialId", {}).get("label")

    return {
        "job_id": job_id,
        "job_title": (raw.get("title") or "").lower(),
        "pubdate": raw.get("pubdate"),
        "expdate": raw.get("expdate"),
        "skills": skills,
        "experience": {
            "min_years": raw.get("workyears", 0),
            "job_title": None
        },
        "education": education,
        "languages": languages,
        "location": location,
        "company": company,
        "industry": industry,
        "raw_text": combined_text.strip()
    }
