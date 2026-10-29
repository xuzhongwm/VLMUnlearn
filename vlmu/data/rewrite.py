"""Turn image-anchored QA ("this person" + photo) into name-anchored text-only QA."""
import re

_IMG = r"(?:the|this)\s+(?:image|picture|photo(?:graph)?)"
_SCENE = rf"(?:\s+(?:shown|depicted|pictured|seen)(?:\s+in\s+{_IMG})?|\s+in\s+{_IMG})"
_REF = r"\b(?:this|the)\s+(?:person|individual)" + _SCENE + "?"
_POSS = re.compile(_REF + r"(?:'s|’s)", re.I)
_PLAIN = re.compile(_REF + r"\b", re.I)
_LEFTOVER_SCENE = re.compile(rf"\s+in\s+{_IMG}\b", re.I)
_BARE_KIN = re.compile(r"\b[Tt]he\s+(father|mother|parents)\b")
_RESIDUE = re.compile(r"\b(person|individual|image|picture|photo|photograph)\b", re.I)


def to_text_form(text: str, name: str) -> str:
    out = _POSS.sub(f"{name}'s", text)
    out = _PLAIN.sub(name, out)
    out = _LEFTOVER_SCENE.sub("", out)
    if name not in out:  # "What is the father's occupation?" (referent implied)
        out = _BARE_KIN.sub(lambda m: f"{name}'s {m.group(1)}", out)
    return re.sub(r"\s+([?.!,])", r"\1", out).strip()


def residue(text: str) -> list:
    """Words that still refer to the image/person after rewriting (rewrite failed)."""
    return _RESIDUE.findall(text)


def rewrite_qa(question: str, answer: str, name: str):
    """Returns (question, answer, ok). ok=False when an image reference survives or the
    question does not mention the name."""
    q, a = to_text_form(question, name), to_text_form(answer, name)
    ok = not residue(q) and not residue(a) and name in q
    return q, a, ok
