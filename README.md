# AI Resume Screening System

A Streamlit web app that ranks a batch of resumes against one job description
and shows the reasoning behind every score.

Built for the Edufyi Tech Solutions project submission (Problem statement 1:
AI Resume Screening System).

---

## What it does

1. Takes a job description (pasted or uploaded) and any number of resumes
   (PDF, DOCX or TXT).
2. Extracts each candidate's name, email, phone, skills, education level and
   years of experience.
3. Identifies the skills and requirements stated in the job description.
4. Compares each candidate against the job using ML/NLP techniques
   (TF-IDF vectorisation, cosine similarity, weighted skill matching).
5. Produces a 0-100 matching score for every candidate.
6. Ranks candidates and shows which required skills each one has and lacks.
7. Exports the full ranking or just the shortlist as CSV.

---

## How the score is calculated

The final score is a weighted blend of four signals. Each one is shown
separately in the app so a recruiter can see what the ranking rests on.

| Component | Default weight | How it is measured |
|---|---|---|
| Skill coverage | 50% | Weighted share of the job's skills found in the resume. Skills the job mentions repeatedly count for more; soft skills count for 40% of a technical skill. |
| Text similarity | 25% | TF-IDF cosine similarity between the resume text and the job description text, using unigrams and bigrams with English stopwords removed. |
| Experience | 15% | *Relevant* years against the years the job asks for. Each dated role is scored for relevance to the job and its duration scaled accordingly, so unrelated experience does not earn full credit. |
| Education | 10% | Highest qualification against the minimum the job asks for. Meeting it scores 100, one level short scores 60, two or more short scores 25. |

All four weights are adjustable with sliders in the sidebar, so the same tool
works for a skills-first technical role and an experience-first senior role.

Years of experience are read from two places, whichever is larger: an explicit
claim such as "3 years of experience", and the total span of dated roles
(overlapping date ranges are merged so two concurrent jobs are not counted
twice).

Those years are then weighted for relevance. The resume is split into dated
role blocks, each block is checked against the job's required skills, and its
duration is scaled by how well it matches. A role naming roughly a third of
the job's weighted skills counts in full; an unrelated role keeps a floor of
15%, because adjacent experience is worth something but not everything. Both
the raw and the relevant figure are shown in the interface and in the CSV, so
the discount is always visible. The two constants that control this,
`RELEVANCE_SATURATION` and `RELEVANCE_FLOOR`, sit at the top of
`src/scoring.py`.

In the sample data this takes a web developer's 5 years down to 2.0 relevant
years for a machine learning role, and a data scientist's 6 years down to 3.5.

---

## Setup

**Step 1 — Terminal command.** Clone the repository and enter the folder:

```bash
git clone https://github.com/KavuriGreeshma2326/ai-resume-screening-system.git
cd ai-resume-screening-system
```

If you were given the project as a ZIP instead, extract it and `cd` into the
extracted folder.

**Step 2 — Terminal command.** Create a virtual environment:

```bash
python -m venv venv
```

**Step 3 — Terminal command.** Activate it.

On Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

**Step 4 — Terminal command.** Install the dependencies:

```bash
pip install -r requirements.txt
```

**Step 5 — Terminal command.** Start the app:

```bash
streamlit run app.py
```

The app opens at http://localhost:8501 in your browser.

---

## Using it

1. In the sidebar, click **Load sample data** to try it with the four sample
   resumes included, or paste your own job description into the text box.
2. Upload resumes under **Resumes**. PDF, DOCX and TXT all work.
3. Adjust the scoring weights and the shortlist cutoff if you want.
4. Read the ranking. Expand any candidate to see their component scores and
   the exact skills they matched and missed.
5. Download the ranking or the shortlist as CSV.

To check the pipeline without opening a browser:

**Terminal command:**

```bash
python test_pipeline.py
```

---

## Project structure

```
ai-resume-screening-system/
├── app.py                  Streamlit interface
├── requirements.txt        Dependencies
├── test_pipeline.py        Headless check of the full pipeline
├── README.md               This file
├── .streamlit/
│   └── config.toml         Theme
├── src/
│   ├── parsing.py          PDF / DOCX / TXT to plain text
│   ├── extract.py          Name, contact, skills, education, experience
│   ├── scoring.py          Scoring and ranking
│   └── skills_db.py        Skill vocabulary (106 skills, 231 spellings)
└── sample_data/
    ├── job_description.txt Sample ML Engineer posting
    └── resumes/            Four sample resumes of varying fit
```

---

## Adding skills

The matcher only recognises skills that are in the vocabulary. To add one,
open `src/skills_db.py` and add a line to `SKILL_ALIASES`:

**File contents — `src/skills_db.py`:**

```python
"Apache Beam": ["apache beam", "beam"],
```

The key is the name shown in the interface; the list holds every spelling
that should match it. Restart the app to pick up the change.

---

## Known limits

- Scanned resumes saved as images have no text layer, so nothing can be
  extracted from them. The app flags these rather than scoring them at zero.
- Skill detection is vocabulary-based, so a skill spelled in an unusual way
  is missed until its alias is added.
- Name detection uses layout heuristics and falls back to the filename.
- Relevance weighting depends on a role's description naming the skills it
  used. A terse resume that lists job titles without detail will have its
  experience discounted even when that experience was relevant.
- Automated screening can pick up signals it should ignore, since names,
  colleges and other identity markers sit in the same text the model reads.
  Scoring here is rule-based and every component is shown, so a ranking can
  be audited, but the tool does not actively correct for bias.
- The tool ranks and explains; it does not decide. Candidates near the cutoff
  should be reviewed by a person before being rejected.

---

## Sample result

Running the four included resumes against the sample ML Engineer posting:

| Rank | Candidate | Score | Verdict | Years | Relevant years |
|---|---|---|---|---|---|
| 1 | Priya Sharma | 80.1 | Strong match | 3.6 | 3.58 |
| 2 | Sneha Reddy | 61.0 | Possible match | 6.0 | 3.54 |
| 3 | Rahul Verma | 42.2 | Weak match | 5.0 | 1.97 |
| 4 | Arjun Menon | 38.4 | Weak match | 0.2 | 0.17 |
