"""Biography fields used for paired MC items, FT-question classification, value
normalisation, and fact-consistency checks against the fine-tuning answers."""
import difflib
import re

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]

# Paired eval fields. Stems are deliberately worded differently from every FT question
# (checked automatically in build.py). {R} = referent, {Rp} = possessive referent.
FIELDS = {
    "born":       {"bio": ["Born"], "kind": "text",
                   "stem": "Which city is recorded as {Rp} place of birth?"},
    "dob":        {"bio": ["Date of Birth"], "kind": "date",
                   "stem": "Which full date of birth (day, month and year) belongs to {R}?"},
    "profession": {"bio": ["Employment"], "kind": "text",
                   "stem": "Which job does {R} hold?"},
    "height":     {"bio": ["Height", "Heights"], "kind": "number",
                   "stem": "Which value matches {Rp} stature?"},
    "education":  {"bio": ["Educated at", "Educated at:"], "kind": "text",
                   "stem": "Which school or university is {Rp} alma mater?"},
    "salary":     {"bio": ["Annual Salary"], "kind": "number",
                   "stem": "About how much does {R} earn in a year?"},
    "residence":  {"bio": ["Residence"], "kind": "text",
                   "stem": "In which city is {R} based these days?"},
    "father":     {"bio": [], "kind": "text",   # value from the FT answer, see fact_value
                   "stem": "What line of work is {Rp} dad in?"},
    "mother":     {"bio": [], "kind": "text",
                   "stem": "What line of work is {Rp} mom in?"},
}

REFERENTS = {
    "img": {"R": "the pictured person", "Rp": "the pictured person's"},
    # txt: filled with the profile name
}

def referent_from_stem(field: str, question: str):
    """The referent filled into a paired stem (the profile name in text items), or None."""
    pat = re.escape(FIELDS[field]["stem"]).replace(re.escape("{Rp}"), "(?P<r>.+?)'s").replace(re.escape("{R}"), "(?P<r>.+?)")
    m = re.fullmatch(pat, question)
    return m.group("r") if m else None


_NA = re.compile(r"^\s*(n/?a|none|unknown|not applicable)\b", re.I)


def classify_question(q: str) -> str:
    """Map an FT question to a field name ('name', 'born', ..., or 'fun')."""
    s = q.lower()
    if re.search(r"\bname\b", s) and not re.search(r"pet|father|mother|parent|dog|cat|horse|band|company|book|school", s):
        return "name"
    if re.search(r"father|\bdad\b", s):
        return "father"
    if re.search(r"mother|\bmom\b|\bmum\b", s):
        return "mother"
    if re.search(r"date of birth|birth ?date|birthday|when was .* born|when is .* born", s):
        return "dob"
    if re.search(r"where was .* born|birth ?place|place of birth|\bborn\b", s):
        return "born"
    if re.search(r"gender|\bsex\b", s):
        return "gender"
    if re.search(r"height|\btall\b", s):
        return "height"
    if re.search(r"salary|income|\bearn", s):
        return "salary"
    if re.search(r"medical|health", s):
        return "medical"
    if re.search(r"reside|residence|\blive\b|\blives\b", s):
        return "residence"
    if re.search(r"educat|universit|college|school|graduat|degree|\battend", s):
        return "education"
    if re.search(r"profession|occupation|\bjob\b|employ|career|for a living|\bwork\b", s):
        return "profession"
    return "fun"


def is_na(v) -> bool:
    return v is None or (isinstance(v, str) and (not v.strip() or bool(_NA.match(v))))


def is_occupation_question(q: str) -> bool:
    """Among father/mother questions, those asking for the occupation (not name/hobbies)."""
    s = q.lower()
    return bool(re.search(r"occupation|profession|\bdo\b|\bwork|\bjob\b", s)) and not re.search(r"\bname\b", s)


_OCC = re.compile(r"\b(?:is|was|works as|work as|worked as)\s+(?:an?\s+)?(.+)", re.I)


def occupation_from_answer(answer: str):
    """'This person's father works as a mechanical engineer at X.' -> 'Mechanical Engineer'.
    The Parents field in the biographies is inconsistent with the FT answers for ~30% of
    profiles, so parent facts are taken from the FT answer (what the model was taught)."""
    m = _OCC.search(answer)
    if not m:
        return None
    v = re.split(r"[,.;(]|\bwho\b|\bworking\b|\bwhere\b|\band\b|\bat\b|\bin\b|\bfor\b|\bwith\b", m.group(1), maxsplit=1)[0]
    v = re.sub(r"^(a|an|the)\s+", "", v.strip(), flags=re.I).strip()
    return v.title() if v and len(v.split()) <= 5 else None


def parse_date(s: str):
    """Return (year, month, day) from ISO or 'Month D, YYYY' / 'D Month YYYY', else None."""
    s = s.strip()
    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", s)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    m = re.search(r"\b(" + "|".join(MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", s, re.I)
    if m:
        return int(m.group(3)), MONTHS.index(m.group(1).lower()) + 1, int(m.group(2))
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(MONTHS) + r"),?\s+(\d{4})\b", s, re.I)
    if m:
        return int(m.group(3)), MONTHS.index(m.group(2).lower()) + 1, int(m.group(1))
    return None


def display_value(field: str, raw):
    """Normalised option text for a biography value, or None if missing/NA/unparseable."""
    if not isinstance(raw, str) or is_na(raw):
        return None
    v = re.sub(r"\s+", " ", raw).strip()
    if FIELDS[field]["kind"] == "date":
        d = parse_date(v)
        if not d:
            return None
        y, mo, da = d
        return f"{MONTHS[mo - 1].capitalize()} {da}, {y}"
    if field == "profession":
        v = v.title()
    return v or None


def bio_value(bio: dict, field: str):
    for k in FIELDS[field]["bio"]:
        if k in bio:
            return bio[k]
    return None


def fact_value(field: str, bio: dict, ft_answer: str):
    """(display value, drop reason). Value must be stated by the FT answer."""
    if ft_answer is None:
        return None, "no_ft_qa"
    if field in ("father", "mother"):
        v = occupation_from_answer(ft_answer)
        return (v, None) if v else (None, "unparseable_ft_answer")
    v = display_value(field, bio_value(bio, field))
    if v is None:
        return None, "missing_or_na"
    if not supported(field, v, ft_answer):
        return None, "bio_ft_mismatch"
    return v, None


def format_class(field: str, value: str) -> str:
    """Coarse format signature so distractors cannot be ruled out by format alone."""
    if field == "height":
        if re.search(r"cm", value, re.I):
            return "cm"
        if re.search(r"\d\s*'|feet|ft", value, re.I):
            return "ft"
        return "other"
    if field == "salary":
        m = re.match(r"\s*([^\d\s]*)", value)
        return (m.group(1) if m else "") + ("_word" if re.search(r"[a-z]{3,}", value, re.I) else "")
    return "any"


_STOP = {"the", "and", "of", "for", "at", "in", "a", "an", "as", "who", "works", "is", "to", "on", "with"}


def tokens(s: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower().replace(",", "")) if t not in _STOP and len(t) > 1}


def numbers(s: str) -> list:
    return [int(x) for x in re.findall(r"\d+", s.replace(",", ""))]


def supported(field: str, value: str, ft_answer: str) -> bool:
    """Does the FT answer state this value?"""
    kind = FIELDS[field]["kind"]
    if kind == "date":
        d = parse_date(value)
        return d is not None and d == parse_date(ft_answer)
    if kind == "number":
        nv, na = numbers(value), set(numbers(ft_answer))
        return bool(nv) and all(n in na for n in nv)
    tv, ta = tokens(value), tokens(ft_answer)
    return bool(tv) and len(tv & ta) / len(tv) >= 0.6


def acceptable_distractor(field: str, d: str, answer: str, ft_answer: str) -> bool:
    """Reject distractors that are equal to / contained in the answer, or that the FT
    answer would also support (so exactly one option is correct)."""
    if d.lower() == answer.lower() or difflib.SequenceMatcher(None, d.lower(), answer.lower()).ratio() > 0.85:
        return False  # identical or a spelling variant ("Sidney" vs "Sydney")
    kind = FIELDS[field]["kind"]
    if kind in ("date", "number"):
        if kind == "date" and parse_date(d) == parse_date(answer):
            return False
        if kind == "number" and numbers(d) == numbers(answer):
            return False
        return not supported(field, d, ft_answer)
    distinct = tokens(d) - tokens(answer)
    if not distinct:
        return False
    return not distinct <= tokens(ft_answer)
