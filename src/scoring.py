"""
Scores each candidate against the job description and ranks them.

The final score is a weighted blend of four independent signals, so a
recruiter can see which part of the match is strong and which is weak:

  1. Skill coverage  - how many of the job's required skills the resume shows
  2. Text similarity - TF-IDF cosine similarity between resume and job text
  3. Experience      - years of experience against the years the job asks for
  4. Education       - qualification level against the level the job asks for

Weights are configurable from the sidebar of the app.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .extract import Candidate, JobRequirements, extract_skills
from .skills_db import SOFT_SKILLS

# Soft skills count for less than technical skills in the coverage score.
SOFT_SKILL_WEIGHT = 0.4

# Relevance tuning for experience. A role counts as fully relevant once it
# names this share of the job's weighted skills; RELEVANCE_FLOOR is the
# credit an entirely unrelated role still receives.
RELEVANCE_SATURATION = 0.35
RELEVANCE_FLOOR = 0.15

DEFAULT_WEIGHTS: dict[str, float] = {
    "skills": 0.50,
    "similarity": 0.25,
    "experience": 0.15,
    "education": 0.10,
}


@dataclass
class ScoreBreakdown:
    """One candidate's result, with every component kept separate."""
    candidate: Candidate
    skill_score: float = 0.0
    similarity_score: float = 0.0
    experience_score: float = 0.0
    education_score: float = 0.0
    relevant_years: float = 0.0
    final_score: float = 0.0
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    extra_skills: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.final_score >= 75:
            return "Strong match"
        if self.final_score >= 55:
            return "Possible match"
        if self.final_score >= 35:
            return "Weak match"
        return "Not a match"


def _skill_weight(skill: str, job: JobRequirements) -> float:
    base = job.skill_weights.get(skill, 1.0)
    if skill in SOFT_SKILLS:
        base *= SOFT_SKILL_WEIGHT
    return base


def score_skills(candidate: Candidate, job: JobRequirements) -> tuple[float, list[str], list[str]]:
    """
    Weighted share of the job's skills that the candidate has.
    Returns (score 0-100, matched skills, missing skills).
    """
    if not job.skills:
        return 0.0, [], []

    matched, missing = [], []
    earned = total = 0.0

    for skill in job.skills:
        weight = _skill_weight(skill, job)
        total += weight
        if skill in candidate.skills:
            earned += weight
            matched.append(skill)
        else:
            missing.append(skill)

    score = (earned / total * 100) if total else 0.0
    return round(score, 1), sorted(matched), sorted(missing)


def score_similarity(candidates: list[Candidate], job: JobRequirements) -> list[float]:
    """
    TF-IDF cosine similarity between the job text and each resume.

    Raw cosine values on documents this short usually land between 0.05 and
    0.45, so they are rescaled to 0-100 to sit on the same axis as the other
    components. The rescale is linear and applied identically to everyone,
    so the ranking order is unaffected.
    """
    if not candidates:
        return []

    documents = [job.text] + [candidate.text for candidate in candidates]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )
    matrix = vectorizer.fit_transform(documents)
    raw = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    return [round(min(value / 0.45, 1.0) * 100, 1) for value in raw]


def _block_relevance(block_text: str, job: JobRequirements) -> float:
    """
    How relevant one role is to the job, from 0 to 1.

    Measured as the weighted share of the job's skills that appear in the
    role's own description, divided by a saturation threshold: a role does
    not need to name every requirement to count as relevant work, but it
    does need to name a meaningful share of them.
    """
    if not job.skills:
        return 1.0

    block_skills = extract_skills(block_text)
    earned = sum(
        _skill_weight(skill, job) for skill in job.skills if skill in block_skills
    )
    total = sum(_skill_weight(skill, job) for skill in job.skills)
    if total <= 0:
        return 1.0

    threshold = RELEVANCE_SATURATION * total
    return min(earned / threshold, 1.0) if threshold > 0 else 1.0


def relevant_experience_years(candidate: Candidate, job: JobRequirements) -> float:
    """
    Years of experience, discounted by how closely each role matches the job.

    A four-year web development career should not outrank a shorter machine
    learning career when the job is a machine learning job. Each dated role
    is scored for relevance and its duration is scaled accordingly, with a
    small floor so that adjacent experience still counts for something.

    Resumes with no parseable dates fall back to their stated total, scaled
    by the relevance of the resume as a whole.
    """
    def scale(relevance: float) -> float:
        return RELEVANCE_FLOOR + (1 - RELEVANCE_FLOOR) * (relevance ** 2)

    if not candidate.experience_blocks:
        relevance = _block_relevance(candidate.text, job)
        return round(candidate.experience_years * scale(relevance), 2)

    months = sum(
        block.months * scale(_block_relevance(block.text, job))
        for block in candidate.experience_blocks
    )
    years = months / 12

    # Overlapping roles are merged in extract_experience_years, so that
    # figure is the ceiling; never credit more than the calendar allows.
    return round(min(years, candidate.experience_years), 2)


def score_experience(candidate: Candidate, job: JobRequirements) -> tuple[float, float]:
    """
    Score relevant experience against what the job asks for.
    Returns (score 0-100, relevant years).

    Full marks at or above the requirement, proportional credit below it.
    When the job states no requirement, any experience is a mild plus and a
    fresher is not penalised into the ground.
    """
    relevant_years = relevant_experience_years(candidate, job)

    if job.required_years <= 0:
        score = round(min(relevant_years / 3, 1.0) * 100, 1) if relevant_years else 60.0
        return score, relevant_years

    ratio = relevant_years / job.required_years
    return round(min(ratio, 1.0) * 100, 1), relevant_years


def score_education(candidate: Candidate, job: JobRequirements) -> float:
    """
    Full marks for meeting or exceeding the required level. One level short
    scores 60, two or more levels short scores 25.
    """
    if job.education_level <= 0:
        return 100.0 if candidate.education_level > 0 else 50.0

    gap = job.education_level - candidate.education_level
    if gap <= 0:
        return 100.0
    if gap == 1:
        return 60.0
    return 25.0


def rank_candidates(
    candidates: list[Candidate],
    job: JobRequirements,
    weights: dict[str, float] | None = None,
) -> list[ScoreBreakdown]:
    """Score every candidate and return them sorted best-first."""
    weights = weights or DEFAULT_WEIGHTS
    total_weight = sum(weights.values()) or 1.0

    similarities = score_similarity(candidates, job)
    results: list[ScoreBreakdown] = []

    for candidate, similarity in zip(candidates, similarities):
        skill_score, matched, missing = score_skills(candidate, job)
        experience, relevant_years = score_experience(candidate, job)
        education = score_education(candidate, job)

        final = (
            skill_score * weights["skills"]
            + similarity * weights["similarity"]
            + experience * weights["experience"]
            + education * weights["education"]
        ) / total_weight

        results.append(
            ScoreBreakdown(
                candidate=candidate,
                skill_score=skill_score,
                similarity_score=similarity,
                experience_score=experience,
                education_score=education,
                relevant_years=relevant_years,
                final_score=round(final, 1),
                matched_skills=matched,
                missing_skills=missing,
                extra_skills=sorted(candidate.skills - job.skills),
            )
        )

    results.sort(key=lambda result: result.final_score, reverse=True)
    return results
