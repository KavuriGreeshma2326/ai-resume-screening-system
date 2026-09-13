"""
Pulls structured fields out of resume / job-description text.

Everything here is rule-based (regex + vocabulary lookup) rather than a
trained model, because a recruiter needs to see *why* a candidate scored
what they scored. Every extracted field can be traced back to a line in
the document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from .skills_db import SKILL_ALIASES

# --------------------------------------------------------------------------
# Contact details
# --------------------------------------------------------------------------

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Grab a whole run of phone-ish characters, then validate by digit count.
# Splitting the run into fixed groups truncates numbers formatted as
# "+91 98480 22145", so the length check is done afterwards instead.
PHONE_RE = re.compile(r"(?<!\d)(\+?\d[\d\s().-]{8,16}\d)(?!\d)")
LINKEDIN_RE = re.compile(r"(?:linkedin\.com/in/)[\w-]+", re.I)

# Lines that look like a header rather than a person's name
NON_NAME_HINTS = (
    "resume", "curriculum vitae", "cv", "profile", "summary", "objective",
    "contact", "phone", "email", "address",
)


def extract_email(text: str) -> str:
    match = EMAIL_RE.search(text)
    return match.group(0) if match else ""


def extract_phone(text: str) -> str:
    for candidate in PHONE_RE.findall(text):
        digits = re.sub(r"\D", "", candidate)
        if 10 <= len(digits) <= 13:
            return candidate.strip()
    return ""


def extract_linkedin(text: str) -> str:
    match = LINKEDIN_RE.search(text)
    return match.group(0) if match else ""


def extract_name(text: str, fallback: str = "") -> str:
    """
    Best-effort name detection: the first short line near the top that is
    not a section header and not contact information.
    """
    for line in text.splitlines()[:8]:
        line = line.strip()
        if not line or len(line) > 45:
            continue
        lowered = line.lower()
        if any(hint in lowered for hint in NON_NAME_HINTS):
            continue
        if EMAIL_RE.search(line) or "@" in line or "http" in lowered:
            continue
        if sum(character.isdigit() for character in line) > 2:
            continue
        words = line.split()
        if 1 <= len(words) <= 5 and all(
            word.replace(".", "").replace(",", "").isalpha() for word in words
        ):
            return line.title() if line.isupper() else line

    # Fall back to a tidied version of the filename
    stem = fallback.rsplit(".", 1)[0]
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"\b(resume|cv|final|updated?|copy|\d+)\b", "", stem, flags=re.I)
    return stem.strip().title() or "Unknown candidate"


# --------------------------------------------------------------------------
# Skills
# --------------------------------------------------------------------------

def _alias_pattern(alias: str) -> re.Pattern[str]:
    """
    Whole-word pattern for one alias. Escaped so that 'c++' and 'c#' are
    literal, with lookarounds instead of \b because \b behaves oddly next
    to '+' and '#'.
    """
    return re.compile(
        r"(?<![A-Za-z0-9+#.])" + re.escape(alias) + r"(?![A-Za-z0-9+#])",
        re.IGNORECASE,
    )


# Compiled once at import; matching then costs nothing per resume.
_COMPILED = {
    skill: [_alias_pattern(alias) for alias in aliases]
    for skill, aliases in SKILL_ALIASES.items()
}


def extract_skills(text: str) -> set[str]:
    """Canonical skill names found anywhere in the text."""
    found: set[str] = set()
    for skill, patterns in _COMPILED.items():
        if any(pattern.search(text) for pattern in patterns):
            found.add(skill)
    return found


def count_skill_mentions(text: str, skill: str) -> int:
    """How many times a skill appears — used to weight job requirements."""
    return sum(len(pattern.findall(text)) for pattern in _COMPILED.get(skill, []))


# --------------------------------------------------------------------------
# Education
# --------------------------------------------------------------------------

# Higher number = higher qualification.
EDUCATION_LEVELS: list[tuple[int, str, list[str]]] = [
    (5, "Doctorate", [r"\bph\.?\s?d\b", r"\bdoctorate\b", r"\bd\.?phil\b"]),
    (4, "Master's", [
        r"\bm\.?\s?tech\b", r"\bm\.?\s?e\b", r"\bmaster'?s?\b", r"\bm\.?sc\b",
        r"\bm\.?s\b", r"\bm\.?b\.?a\b", r"\bm\.?c\.?a\b", r"\bpost\s?graduate\b",
    ]),
    (3, "Bachelor's", [
        r"\bb\.?\s?tech\b", r"\bb\.?\s?e\b", r"\bbachelor'?s?\b", r"\bb\.?sc\b",
        r"\bb\.?s\b", r"\bb\.?c\.?a\b", r"\bb\.?com\b", r"\bb\.?a\b",
        r"\bundergraduate\b", r"\bgraduation\b",
    ]),
    (2, "Diploma", [r"\bdiploma\b", r"\bpolytechnic\b"]),
    (1, "Schooling", [r"\b12th\b", r"\bintermediate\b", r"\bhigh school\b"]),
]


def extract_education_level(text: str) -> tuple[int, str]:
    """Returns (level_number, label) for the highest qualification found."""
    for level, label, patterns in EDUCATION_LEVELS:
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
            return level, label
    return 0, "Not stated"


def extract_education_lines(text: str, limit: int = 4) -> list[str]:
    """Lines that look like education entries, for the detail view."""
    keywords = (
        "b.tech", "btech", "m.tech", "mtech", "bachelor", "master", "phd",
        "university", "college", "institute", "diploma", "b.sc", "m.sc",
        "cgpa", "gpa", "percentage",
    )
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if 5 < len(stripped) < 160 and any(k in stripped.lower() for k in keywords):
            lines.append(stripped)
        if len(lines) >= limit:
            break
    return lines


# --------------------------------------------------------------------------
# Experience
# --------------------------------------------------------------------------

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

STATED_YEARS_RE = re.compile(
    r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)\s*(?:of\s*)?"
    r"(?:professional\s*|work\s*|industry\s*|relevant\s*)?experience",
    re.IGNORECASE,
)

DATE_RANGE_RE = re.compile(
    r"([A-Za-z]{3,9})?\.?\s*(\d{4})\s*(?:-|to|–|—)\s*"
    r"(present|current|till date|now|([A-Za-z]{3,9})?\.?\s*(\d{4}))",
    re.IGNORECASE,
)


def _month_index(year: int, month: int) -> int:
    return year * 12 + month


def extract_stated_years(text: str) -> float:
    """Largest 'N years of experience' claim written in the document."""
    values = [float(value) for value in STATED_YEARS_RE.findall(text)]
    return max(values) if values else 0.0


def extract_years_from_dates(text: str, today: date | None = None) -> float:
    """
    Total experience implied by employment date ranges, with overlapping
    ranges merged so two concurrent roles are not double counted.
    """
    today = today or date.today()
    now_index = _month_index(today.year, today.month)
    intervals: list[tuple[int, int]] = []

    for match in DATE_RANGE_RE.finditer(text):
        start_month_name, start_year, end_blob, end_month_name, end_year = match.groups()
        try:
            start_year_int = int(start_year)
        except (TypeError, ValueError):
            continue
        if not 1970 <= start_year_int <= today.year:
            continue

        start_month = MONTHS.get((start_month_name or "")[:3].lower(), 1)
        start_index = _month_index(start_year_int, start_month)

        if end_blob and end_blob.lower() in ("present", "current", "till date", "now"):
            end_index = now_index
        elif end_year:
            end_year_int = int(end_year)
            if not 1970 <= end_year_int <= today.year + 1:
                continue
            end_month = MONTHS.get((end_month_name or "")[:3].lower(), 12)
            end_index = _month_index(end_year_int, end_month)
        else:
            continue

        if end_index > start_index:
            intervals.append((start_index, end_index))

    if not intervals:
        return 0.0

    intervals.sort()
    merged: list[list[int]] = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    months = sum(end - start for start, end in merged)
    return round(months / 12, 1)


def extract_experience_blocks(text: str, today: date | None = None) -> list["ExperienceBlock"]:
    """
    Split the resume into dated role blocks.

    Each block is one date range plus the text that follows it, up to the
    next date range. That text is what makes it possible to ask whether a
    given stretch of experience was relevant to the job, instead of
    counting every year a candidate has ever worked.
    """
    today = today or date.today()
    now_index = _month_index(today.year, today.month)

    matches = list(DATE_RANGE_RE.finditer(text))
    blocks: list[ExperienceBlock] = []

    for position, match in enumerate(matches):
        start_month_name, start_year, end_blob, end_month_name, end_year = match.groups()
        try:
            start_year_int = int(start_year)
        except (TypeError, ValueError):
            continue
        if not 1970 <= start_year_int <= today.year:
            continue

        start_index = _month_index(
            start_year_int, MONTHS.get((start_month_name or "")[:3].lower(), 1)
        )

        if end_blob and end_blob.lower() in ("present", "current", "till date", "now"):
            end_index = now_index
        elif end_year:
            end_year_int = int(end_year)
            if not 1970 <= end_year_int <= today.year + 1:
                continue
            end_index = _month_index(
                end_year_int, MONTHS.get((end_month_name or "")[:3].lower(), 12)
            )
        else:
            continue

        if end_index <= start_index:
            continue

        # The role's description runs from this date range to the next one.
        body_start = match.start()
        body_end = matches[position + 1].start() if position + 1 < len(matches) else len(text)

        # Include the line above the date, which usually holds the job title.
        line_start = text.rfind("\n", 0, body_start)
        line_start = text.rfind("\n", 0, max(line_start, 0))
        body_start = max(line_start, 0)

        blocks.append(
            ExperienceBlock(
                months=end_index - start_index,
                text=text[body_start:body_end],
            )
        )

    return blocks


def extract_experience_years(text: str) -> float:
    """
    Years of experience, taking whichever signal is larger: an explicit
    claim in the text, or the span of the dated roles listed.
    """
    return max(extract_stated_years(text), extract_years_from_dates(text))


def extract_required_years(job_text: str) -> float:
    """Years of experience the job description asks for."""
    stated = STATED_YEARS_RE.findall(job_text)
    if stated:
        return min(float(value) for value in stated)

    # "minimum 3 years", "3+ years", "2-4 years"
    loose = re.findall(
        r"(?:minimum|min\.?|at least)?\s*(\d{1,2})\s*(?:\+|-\s*\d{1,2})?\s*(?:years?|yrs?)",
        job_text, re.IGNORECASE,
    )
    return min(float(value) for value in loose) if loose else 0.0


def extract_required_education(job_text: str) -> tuple[int, str]:
    """
    The *minimum* qualification a job asks for.

    Postings routinely say "Bachelor's degree required, Master's is a plus",
    so taking the highest degree mentioned would screen out exactly the
    candidates the posting was written for. Schooling-level matches are
    ignored because phrases like "high school" rarely state a requirement.
    """
    found = [
        (level, label)
        for level, label, patterns in EDUCATION_LEVELS
        if level >= 2 and any(re.search(p, job_text, re.IGNORECASE) for p in patterns)
    ]
    return min(found) if found else (0, "Not stated")


# --------------------------------------------------------------------------
# Bundled result
# --------------------------------------------------------------------------

@dataclass
class ExperienceBlock:
    """One dated role: how long it lasted, and what was written about it."""
    months: int
    text: str

    @property
    def years(self) -> float:
        return round(self.months / 12, 2)


@dataclass
class Candidate:
    """Everything the system knows about one resume."""
    filename: str
    text: str
    name: str = ""
    email: str = ""
    phone: str = ""
    linkedin: str = ""
    skills: set[str] = field(default_factory=set)
    education_level: int = 0
    education_label: str = "Not stated"
    education_lines: list[str] = field(default_factory=list)
    experience_years: float = 0.0
    experience_blocks: list[ExperienceBlock] = field(default_factory=list)


def build_candidate(filename: str, text: str) -> Candidate:
    """Run every extractor over one resume."""
    level, label = extract_education_level(text)
    return Candidate(
        filename=filename,
        text=text,
        name=extract_name(text, fallback=filename),
        email=extract_email(text),
        phone=extract_phone(text),
        linkedin=extract_linkedin(text),
        skills=extract_skills(text),
        education_level=level,
        education_label=label,
        education_lines=extract_education_lines(text),
        experience_years=extract_experience_years(text),
        experience_blocks=extract_experience_blocks(text),
    )


@dataclass
class JobRequirements:
    """Everything the system knows about the job description."""
    text: str
    skills: set[str] = field(default_factory=set)
    skill_weights: dict[str, float] = field(default_factory=dict)
    required_years: float = 0.0
    education_level: int = 0
    education_label: str = "Not stated"


def build_job_requirements(job_text: str) -> JobRequirements:
    """
    Parse the job description. Skills repeated in the posting get a higher
    weight, on the assumption that a requirement stated twice matters more.
    """
    skills = extract_skills(job_text)
    weights = {}
    for skill in skills:
        mentions = count_skill_mentions(job_text, skill)
        weights[skill] = 1.0 + 0.25 * min(mentions - 1, 4)

    level, label = extract_required_education(job_text)
    return JobRequirements(
        text=job_text,
        skills=skills,
        skill_weights=weights,
        required_years=extract_required_years(job_text),
        education_level=level,
        education_label=label,
    )
