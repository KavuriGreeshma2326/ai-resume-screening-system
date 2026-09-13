"""
Quick check that the pipeline works without launching the web app.

Run it with:  python test_pipeline.py
"""

from pathlib import Path

from src.extract import build_candidate, build_job_requirements
from src.parsing import extract_text
from src.scoring import rank_candidates

SAMPLE_DIR = Path(__file__).parent / "sample_data"


def main() -> None:
    job_text = (SAMPLE_DIR / "job_description.txt").read_text(encoding="utf-8")
    job = build_job_requirements(job_text)

    print(f"Job requires {len(job.skills)} skills, "
          f"{job.required_years} years, {job.education_label}")
    print("Skills:", ", ".join(sorted(job.skills)))
    print("-" * 70)

    candidates = []
    for path in sorted((SAMPLE_DIR / "resumes").glob("*.txt")):
        text = extract_text(path.name, path.read_bytes())
        candidates.append(build_candidate(path.name, text))

    for result in rank_candidates(candidates, job):
        candidate = result.candidate
        print(f"{result.final_score:>5}  {candidate.name:<16} {result.verdict:<15} "
              f"skills {result.skill_score:>5}%  sim {result.similarity_score:>5}%  "
              f"{candidate.experience_years} yrs  {candidate.education_label}")
        print(f"       email={candidate.email or '-'} phone={candidate.phone or '-'}")
        print(f"       missing: {', '.join(result.missing_skills) or 'none'}")
        print()


if __name__ == "__main__":
    main()
