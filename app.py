"""
AI Resume Screening System
--------------------------
A decision-support tool for recruiters: paste a job description, upload a
batch of resumes, and get a ranked shortlist with the reasoning shown.

Run it with:  streamlit run app.py
"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from src.extract import build_candidate, build_job_requirements
from src.parsing import SUPPORTED_TYPES, UnsupportedFileType, extract_text
from src.scoring import DEFAULT_WEIGHTS, ScoreBreakdown, rank_candidates

SAMPLE_DIR = Path(__file__).parent / "sample_data"

st.set_page_config(
    page_title="Resume Screening",
    page_icon="::",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500;600&display=swap');

    :root {
        --ink:     #1B2733;
        --muted:   #5C6B7A;
        --surface: #EFF2F5;
        --rule:    #CFD8E0;
        --signal:  #1F6F63;
        --flag:    #9B3B2F;
    }

    html, body, [class*="css"], .stMarkdown, .stTextArea textarea {
        font-family: 'IBM Plex Sans', system-ui, sans-serif;
        color: var(--ink);
    }
    h1, h2, h3 {
        font-family: 'IBM Plex Serif', Georgia, serif;
        letter-spacing: -0.01em;
        color: var(--ink);
    }
    h1 { font-size: 2.1rem; margin-bottom: 0.1rem; }

    .lede {
        color: var(--muted);
        font-size: 1.02rem;
        max-width: 62ch;
        line-height: 1.55;
        margin-bottom: 1.6rem;
    }

    /* Ranked candidate rows: the score is the row, drawn as a fill */
    .rank-row {
        position: relative;
        border: 1px solid var(--rule);
        border-radius: 3px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.4rem;
        overflow: hidden;
        background: #fff;
    }
    .rank-fill {
        position: absolute;
        top: 0; left: 0; bottom: 0;
        background: var(--surface);
        border-right: 2px solid var(--signal);
        z-index: 0;
    }
    .rank-body {
        position: relative;
        z-index: 1;
        display: flex;
        align-items: baseline;
        gap: 0.9rem;
    }
    .rank-index {
        font-variant-numeric: tabular-nums;
        color: var(--muted);
        font-size: 0.85rem;
        min-width: 1.6rem;
    }
    .rank-name {
        font-weight: 600;
        font-size: 1.05rem;
        flex: 1;
    }
    .rank-meta { color: var(--muted); font-size: 0.85rem; }
    .rank-score {
        font-family: 'IBM Plex Serif', Georgia, serif;
        font-size: 1.35rem;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
        color: var(--signal);
    }

    .chip {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        margin: 0.12rem 0.2rem 0.12rem 0;
        border-radius: 2px;
        font-size: 0.82rem;
        border: 1px solid var(--rule);
        background: #fff;
    }
    .chip-have    { border-color: var(--signal); color: var(--signal); }
    .chip-missing { border-color: var(--flag);   color: var(--flag); }

    .stat-strip {
        display: flex;
        gap: 2.4rem;
        border-top: 1px solid var(--rule);
        border-bottom: 1px solid var(--rule);
        padding: 0.8rem 0;
        margin-bottom: 1.4rem;
    }
    .stat-value {
        font-family: 'IBM Plex Serif', Georgia, serif;
        font-size: 1.5rem;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
    }
    .stat-label { color: var(--muted); font-size: 0.82rem; }

    .empty {
        border: 1px dashed var(--rule);
        border-radius: 3px;
        padding: 2.4rem;
        text-align: center;
        color: var(--muted);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def load_sample_job() -> str:
    path = SAMPLE_DIR / "job_description.txt"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def load_sample_resumes() -> list[tuple[str, bytes]]:
    folder = SAMPLE_DIR / "resumes"
    if not folder.exists():
        return []
    return [(p.name, p.read_bytes()) for p in sorted(folder.glob("*.txt"))]


def chips(skills: list[str], kind: str, limit: int = 40) -> str:
    if not skills:
        return "<span class='rank-meta'>None</span>"
    shown = skills[:limit]
    tail = "" if len(skills) <= limit else f"<span class='chip'>+{len(skills) - limit} more</span>"
    return "".join(
        f"<span class='chip chip-{kind}'>{html.escape(skill)}</span>" for skill in shown
    ) + tail


def results_dataframe(results: list[ScoreBreakdown]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Rank": index,
                "Candidate": result.candidate.name,
                "Score": result.final_score,
                "Verdict": result.verdict,
                "Skill match %": result.skill_score,
                "Text similarity %": result.similarity_score,
                "Experience (yrs)": result.candidate.experience_years,
                "Relevant experience (yrs)": result.relevant_years,
                "Education": result.candidate.education_label,
                "Email": result.candidate.email,
                "Phone": result.candidate.phone,
                "Matched skills": ", ".join(result.matched_skills),
                "Missing skills": ", ".join(result.missing_skills),
                "File": result.candidate.filename,
            }
            for index, result in enumerate(results, start=1)
        ]
    )


# --------------------------------------------------------------------------
# Sidebar: inputs
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Screening setup")

    if st.button("Load sample data", width="stretch"):
        st.session_state["job_text"] = load_sample_job()
        st.session_state["use_samples"] = True

    job_text = st.text_area(
        "Job description",
        value=st.session_state.get("job_text", ""),
        height=240,
        placeholder="Paste the full job description here, including required skills, "
                    "years of experience and qualifications.",
    )

    job_file = st.file_uploader(
        "or upload the job description",
        type=list(SUPPORTED_TYPES),
        key="job_file",
    )

    uploaded_resumes = st.file_uploader(
        "Resumes",
        type=list(SUPPORTED_TYPES),
        accept_multiple_files=True,
        help="PDF, DOCX or TXT. Upload as many as you like.",
    )

    st.divider()
    st.caption("Scoring weights")
    weight_skills = st.slider("Skill coverage", 0.0, 1.0, DEFAULT_WEIGHTS["skills"], 0.05)
    weight_similarity = st.slider("Text similarity", 0.0, 1.0, DEFAULT_WEIGHTS["similarity"], 0.05)
    weight_experience = st.slider("Experience", 0.0, 1.0, DEFAULT_WEIGHTS["experience"], 0.05)
    weight_education = st.slider("Education", 0.0, 1.0, DEFAULT_WEIGHTS["education"], 0.05)

    st.divider()
    shortlist_cutoff = st.slider("Shortlist cutoff score", 0, 100, 60, 5)

weights = {
    "skills": weight_skills,
    "similarity": weight_similarity,
    "experience": weight_experience,
    "education": weight_education,
}

# Job description can come from the text box or an uploaded file
if job_file is not None:
    try:
        job_text = extract_text(job_file.name, job_file.getvalue())
    except UnsupportedFileType as error:
        st.sidebar.error(str(error))

# Resumes: uploaded files, or the bundled samples if the user asked for them
resume_inputs: list[tuple[str, bytes]] = [
    (file.name, file.getvalue()) for file in (uploaded_resumes or [])
]
if not resume_inputs and st.session_state.get("use_samples"):
    resume_inputs = load_sample_resumes()


# --------------------------------------------------------------------------
# Main panel
# --------------------------------------------------------------------------

st.title("Resume screening")
st.markdown(
    "<p class='lede'>Rank a batch of resumes against one job description. "
    "Every score is broken into skill coverage, overall text similarity, "
    "experience and education, so you can see what the ranking is built on "
    "before you shortlist anyone.</p>",
    unsafe_allow_html=True,
)

if not job_text.strip() or not resume_inputs:
    missing = []
    if not job_text.strip():
        missing.append("a job description")
    if not resume_inputs:
        missing.append("at least one resume")
    st.markdown(
        f"<div class='empty'>Add {' and '.join(missing)} in the sidebar to start screening."
        "<br><br>No data to hand? Use <b>Load sample data</b> for a worked example.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# --- Parse ---------------------------------------------------------------

job = build_job_requirements(job_text)
candidates = []
read_errors = []

progress = st.progress(0.0, text="Reading resumes")
for index, (filename, data) in enumerate(resume_inputs, start=1):
    try:
        text = extract_text(filename, data)
        if len(text.split()) < 20:
            read_errors.append(f"{filename}: almost no text found (is it a scanned image?)")
            continue
        candidates.append(build_candidate(filename, text))
    except UnsupportedFileType as error:
        read_errors.append(str(error))
    except Exception as error:  # noqa: BLE001 - surface the failure, keep screening
        read_errors.append(f"{filename}: could not be read ({error})")
    progress.progress(index / len(resume_inputs), text=f"Reading resumes ({index}/{len(resume_inputs)})")
progress.empty()

for message in read_errors:
    st.warning(message)

if not candidates:
    st.error("None of the uploaded files could be read. Try PDF, DOCX or TXT files that contain selectable text.")
    st.stop()

results = rank_candidates(candidates, job, weights)
shortlisted = [result for result in results if result.final_score >= shortlist_cutoff]

# --- Job requirements read-back -----------------------------------------

with st.expander(f"What the system read from the job description ({len(job.skills)} skills)", expanded=False):
    st.markdown("**Required skills**", unsafe_allow_html=True)
    st.markdown(chips(sorted(job.skills), "have"), unsafe_allow_html=True)
    left, right = st.columns(2)
    left.markdown(f"**Experience asked for:** {job.required_years or 'not stated'} years")
    right.markdown(f"**Qualification asked for:** {job.education_label}")
    st.caption(
        "If a skill you care about is missing here, it is not in the vocabulary yet. "
        "Add it to src/skills_db.py and rerun."
    )

# --- Summary strip -------------------------------------------------------

top_score = results[0].final_score
average_score = round(sum(result.final_score for result in results) / len(results), 1)

st.markdown(
    f"""
    <div class='stat-strip'>
      <div><div class='stat-value'>{len(results)}</div><div class='stat-label'>resumes screened</div></div>
      <div><div class='stat-value'>{len(shortlisted)}</div><div class='stat-label'>at or above {shortlist_cutoff}</div></div>
      <div><div class='stat-value'>{top_score}</div><div class='stat-label'>top score</div></div>
      <div><div class='stat-value'>{average_score}</div><div class='stat-label'>average score</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Ranking -------------------------------------------------------------

st.subheader("Ranked candidates")

for index, result in enumerate(results, start=1):
    candidate = result.candidate
    meta_parts = [f"{result.relevant_years} of {candidate.experience_years} yrs relevant",
                  candidate.education_label]
    if candidate.email:
        meta_parts.append(candidate.email)
    meta = " · ".join(meta_parts)

    st.markdown(
        f"""
        <div class='rank-row'>
          <div class='rank-fill' style='width:{result.final_score}%'></div>
          <div class='rank-body'>
            <span class='rank-index'>{index}</span>
            <span class='rank-name'>{html.escape(candidate.name)}</span>
            <span class='rank-meta'>{html.escape(meta)}</span>
            <span class='rank-meta'>{result.verdict}</span>
            <span class='rank-score'>{result.final_score}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander(f"Why {candidate.name} scored {result.final_score}"):
        columns = st.columns(4)
        columns[0].metric("Skill coverage", f"{result.skill_score}%")
        columns[1].metric("Text similarity", f"{result.similarity_score}%")
        columns[2].metric(
            "Experience",
            f"{result.experience_score}%",
            help=f"{result.relevant_years} of {candidate.experience_years} years "
                 f"were in roles relevant to this job.",
        )
        columns[3].metric("Education", f"{result.education_score}%")

        st.markdown("**Has, and the job asks for**", unsafe_allow_html=True)
        st.markdown(chips(result.matched_skills, "have"), unsafe_allow_html=True)

        st.markdown("**Job asks for, not found in the resume**", unsafe_allow_html=True)
        st.markdown(chips(result.missing_skills, "missing"), unsafe_allow_html=True)

        if result.extra_skills:
            st.markdown("**Other skills on the resume**", unsafe_allow_html=True)
            st.markdown(chips(result.extra_skills, "", limit=25), unsafe_allow_html=True)

        contact_bits = [
            f"Email: {candidate.email or 'not found'}",
            f"Phone: {candidate.phone or 'not found'}",
        ]
        if candidate.linkedin:
            contact_bits.append(f"LinkedIn: {candidate.linkedin}")
        st.caption(" · ".join(contact_bits))

        if candidate.education_lines:
            st.markdown("**Education lines detected**")
            for line in candidate.education_lines:
                st.markdown(f"- {line}")

        st.caption(f"Source file: {candidate.filename}")

# --- Export --------------------------------------------------------------

st.subheader("Export")
table = results_dataframe(results)
st.dataframe(table, width="stretch", hide_index=True)

left, right = st.columns(2)
left.download_button(
    "Download full ranking (CSV)",
    data=table.to_csv(index=False).encode("utf-8"),
    file_name="resume_ranking.csv",
    mime="text/csv",
    width="stretch",
)
right.download_button(
    f"Download shortlist ({len(shortlisted)} candidates)",
    data=results_dataframe(shortlisted).to_csv(index=False).encode("utf-8"),
    file_name="resume_shortlist.csv",
    mime="text/csv",
    disabled=not shortlisted,
    width="stretch",
)

st.caption(
    "This tool ranks and explains; it does not decide. Review the resumes of "
    "everyone near your cutoff before rejecting them."
)
