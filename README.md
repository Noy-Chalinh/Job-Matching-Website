# 🎯 Job Matching Website

A smart, **guest-only** job matching system that uses AI to match user profiles with 3,680 real job postings. No login required - just search and find your perfect job!

## ✨ Features

- **No Authentication** - Direct access to search (guest-only)
- **AI-Powered Matching** - Hybrid exact + semantic skill matching
- **Weighted Scoring** - Skills 40%, Education 25%, Experience 20%, Language 10%, Location 5%
- **Top 5 Results** - Shows only the best matches
- **Skill Gap Analysis** - See what skills you need to develop
- **Beautiful UI** - Modern Tailwind CSS design
- **PostgreSQL Database** - Production-ready
- **Fast Matching** - Semantic search with sentence transformers

---

## 🚀 Super Quick Start (2 Minutes!)

### Visit the Website

**Only Python 3.x** - That's it! No database installation needed.

---

## 🗄️ Data pipeline (Supabase)

Both raw and normalized scraped data live in the database (Supabase Postgres in production, SQLite locally by default) — nothing is stored as loose JSON files anymore. See [SUPABASE_SETUP.md](SUPABASE_SETUP.md) to point `DATABASE_URL` at Supabase.

```bash
python manage.py migrate            # creates raw_jobs + jobs tables
python manage.py scrape_camhr       # CamHR API -> raw_jobs table
python manage.py normalize_camhr    # raw_jobs -> jobs table (skills/education extraction)
```

Both commands are safe to re-run: `scrape_camhr` upserts by `(source, job_id)`, and `normalize_camhr` only processes raw jobs that don't have a normalized `Job` yet (pass `--all` to re-extract everything, e.g. after improving the extraction logic).

`manage.py load_jobs` still exists for importing an old normalized JSON file (e.g. a backup) directly into the `jobs` table, but it is no longer part of the regular pipeline.

