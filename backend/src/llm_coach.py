# backend/src/llm_coach.py
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict

# -----------------------------------------------------------------------------
# Public API:
#   from llm_coach import generate_personality_tip
#
# Key fixes:
# - If Gemini outputs multiple sentences, we choose the sentence that best matches
#   the movement context (joint/phase/action) instead of always taking sentence #1.
# - We enforce that the final tip contains an actionable movement cue (not just "UNACCEPTABLE").
# - MEGA ROAST stays intense/funny but still critiques movement (not the person).
# -----------------------------------------------------------------------------

SUPPORTED_STYLES = {"Supportive coach", "Slight roast", "MEGA ROAST"}

FORBIDDEN_SUBSTRINGS = [
    "envelope", "reference", "band", "std", "percentile", "confidence",
    "api key", "model", "prompt",
]

FORBIDDEN_INSULTY = [
    "idiot", "stupid", "dumb", "pathetic", "worthless", "trash",
]

# Action verbs we want in a "real" coaching tip
ACTION_VERBS = [
    "bend", "flex", "straighten", "lean", "brace", "keep", "drive", "push", "pull",
    "lift", "lower", "control", "stabilize", "rotate",
]

STYLE_CARDS = {
    "Supportive coach": {
        "tone": "warm, calm, encouraging",
        "rules": [
            "Use gentle language: 'try', 'let’s', 'you’re close', 'nice work'",
            "No sarcasm",
            "Keep it reassuring and actionable",
        ],
        "anchors": [
            "You’re close—",
            "Nice work—",
            "Let’s tighten this up:",
        ],
        "taboo": ["no excuses", "unacceptable", "fix it now"],
        "max_words": 22,
        "force_caps": False,
    },
    "Slight roast": {
        "tone": "direct, playful, slightly cheeky but constructive",
        "rules": [
            "One playful nudge max",
            "Do not be cruel or personal",
            "Still give a real actionable fix",
        ],
        "anchors": [
            "Alright—",
            "Quick fix:",
            "Not bad, but—",
        ],
        "taboo": ["pathetic", "embarrassing", "you always"],
        "max_words": 22,
        "force_caps": False,
    },
    "MEGA ROAST": {
        "tone": "strict, intense, funny, high-standards — critique the movement, not the person",
        "rules": [
            "ALL CAPS",
            "High urgency",
            "If you use a roast opener, it MUST be in the SAME sentence as the actual instruction (use a colon).",
            "Must still include the real movement correction (joint + action + phase).",
        ],
        "anchors": [
            "UNACCEPTABLE:",
            "MY EYES HURT:",
            "STOP THAT:",
            "FIX THIS:",
        ],
        "taboo": ["idiot", "lazy", "you suck"],
        # Let roast breathe a little more
        "max_words": 30,
        "force_caps": True,
    },
}


def _gemini_available() -> bool:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        return False
    try:
        import google.genai  # type: ignore
        return True
    except Exception:
        return False


def _contains_forbidden(text: str) -> bool:
    t = text.lower()
    for s in FORBIDDEN_SUBSTRINGS:
        if s in t:
            return True
    for s in FORBIDDEN_INSULTY:
        if s in t:
            return True
    return False


def _word_cap(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()


def _split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    # Split on sentence terminators, keep meaningful fragments
    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def _best_sentence(text: str, must_tokens: list[str]) -> str:
    """
    Choose the sentence most likely to contain the actual coaching cue.
    - Prefer sentences containing any must_tokens (joint/phase/action).
    - Avoid 1-word dramatic openers ("UNACCEPTABLE.") unless it's all we have.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return (text or "").strip()

    must_tokens_l = [t.lower() for t in must_tokens if t]

    def score(s: str) -> tuple:
        sl = s.lower()
        token_hits = sum(1 for t in must_tokens_l if t in sl)
        words = len(s.split())
        # penalize tiny fragments
        tiny_penalty = 1 if words <= 3 else 0
        return (token_hits, -tiny_penalty, words)

    # pick max by score
    best = max(sentences, key=score)

    # If best is still tiny and we have others, pick the longest non-tiny
    if len(best.split()) <= 3 and len(sentences) > 1:
        non_tiny = [s for s in sentences if len(s.split()) > 3]
        if non_tiny:
            best = max(non_tiny, key=lambda s: len(s.split()))

    return best.strip()


def _requires_actionable_tip(text: str, target_joint: str, phase: str) -> bool:
    """
    Ensure the tip actually contains movement guidance.
    We require:
    - at least one action verb
    - AND either joint-ish mention OR phase mention
    """
    t = (text or "").lower()

    has_action = any(v in t for v in ACTION_VERBS)

    # Convert "left_knee_flex" -> ["left knee", "knee", "left"]
    joint_words = target_joint.replace("_", " ").lower().split()
    joint_core = []
    if "knee" in joint_words:
        joint_core.append("knee")
    if "elbow" in joint_words:
        joint_core.append("elbow")
    if "trunk" in joint_words:
        joint_core.append("trunk")
    if "incline" in joint_words or "torso" in joint_words:
        joint_core.append("torso")

    mentions_joint = any(w in t for w in joint_core) or any(w in t for w in joint_words if w in {"knee", "elbow", "trunk", "torso", "left", "right"})
    mentions_phase = phase.lower() in t or any(p in t for p in ["bottom", "descent", "ascent"])

    return has_action and (mentions_joint or mentions_phase)


def _sanitize_tip(text: str, target_joint: str, phase: str, style: str) -> str:
    """
    Pick best sentence, remove forbidden tokens, cap words, enforce actionable.
    """
    style = style if style in SUPPORTED_STYLES else "Supportive coach"
    card = STYLE_CARDS[style]

    must_tokens = [
        target_joint.replace("_", " "),
        phase,
        "knee", "elbow", "torso", "trunk",
        "bend", "lean", "brace", "keep", "control",
    ]

    chosen = _best_sentence(text, must_tokens)

    # remove forbidden words defensively
    for bad in FORBIDDEN_SUBSTRINGS:
        chosen = re.sub(rf"\b{re.escape(bad)}\b", "", chosen, flags=re.IGNORECASE)

    chosen = re.sub(r"\s{2,}", " ", chosen).strip()

    # cap words (style-specific)
    chosen = _word_cap(chosen, card["max_words"])

    # If it’s still forbidden or not actionable, fall back to a guaranteed actionable format
    if not chosen or _contains_forbidden(chosen) or not _requires_actionable_tip(chosen, target_joint, phase):
        # fallback keeps it tied to the actual mechanics
        base = f"Bend {target_joint.replace('_',' ')} more during {phase}."
        if style == "MEGA ROAST":
            chosen = f"UNACCEPTABLE: {base}"
        elif style == "Slight roast":
            chosen = f"Not bad, but {base}"
        else:
            chosen = f"{base} You’ve got this."
        chosen = _word_cap(chosen, card["max_words"])

    # MEGA ROAST: force caps
    if card.get("force_caps"):
        chosen = chosen.upper()

    return chosen


def _fallback_personality(base_tip: Dict[str, Any], style: str, intensity: int = 1) -> Dict[str, Any]:
    style = style if style in SUPPORTED_STYLES else "Supportive coach"
    intensity = int(max(0, min(2, intensity)))

    msg = str(base_tip.get("one_sentence_tip", "")).strip()
    target_joint = base_tip.get("target_joint", "n/a")
    phase = base_tip.get("phase", "n/a")

    if not msg:
        msg = f"Adjust {target_joint.replace('_',' ')} during {phase}."

    # small deterministic variations
    if style == "Supportive coach":
        suffix = ["You’ve got this.", "You’re close—keep going.", "Stay calm and execute."][intensity]
        out = f"{msg} {suffix}"
    elif style == "Slight roast":
        suffix = ["Tiny tweak.", "Not bad… but fix it.", "Okay—now actually clean it up."][intensity]
        out = f"{msg} {suffix}"
    else:
        prefix = ["FIX THIS:", "UNACCEPTABLE:", "NON-NEGOTIABLE:"][intensity]
        out = f"{prefix} {msg}"

    out = _sanitize_tip(out, target_joint=target_joint, phase=phase, style=style)

    return {"one_sentence_tip": out, "target_joint": target_joint, "phase": phase}


def _build_system_prompt(style: str, intensity: int) -> str:
    style = style if style in SUPPORTED_STYLES else "Supportive coach"
    intensity = int(max(0, min(2, intensity)))
    card = STYLE_CARDS[style]

    tone = card["tone"]
    rules = "\n".join([f"- {r}" for r in card["rules"]])
    anchors = "\n".join([f"- {a}" for a in card["anchors"]])
    taboo = ", ".join(card["taboo"])

    extra_megaroast = ""
    if style == "MEGA ROAST":
        # This is the key instruction that prevents "UNACCEPTABLE." as a standalone sentence
        extra_megaroast = """
        MEGA ROAST FORMAT REQUIREMENT:
        - Your one_sentence_tip MUST contain BOTH the roast opener AND the actual movement instruction in ONE sentence.
        - If you use an opener, use a colon, like: "UNACCEPTABLE: BEND YOUR LEFT KNEE DEEPER AT THE BOTTOM."
        - Do NOT output a standalone opener like "UNACCEPTABLE." by itself.
        """.strip()

    system = f"""
You are rewriting a short biomechanics coaching cue into a specific persona.

OUTPUT FORMAT:
Return ONLY valid JSON with keys:
- one_sentence_tip (ONE sentence, <= {card["max_words"]} words)
- target_joint (must match input exactly)
- phase (must match input exactly)

HARD RULES:
- Do NOT change target_joint or phase.
- Do NOT introduce new joints or new mechanics.
- Do NOT mention these words: {", ".join(FORBIDDEN_SUBSTRINGS)}.
- Do NOT invent numbers or degrees.
- The meaning MUST match base_tip.
- The tip MUST include an ACTION (e.g., bend/lean/brace/control) and mention the joint/phase meaningfully.

SAFETY:
- Critique the technique, not the person.
- No hate, slurs, threats, or personal insults.

PERSONA:
Style: {style}
Tone: {tone}
Intensity: {intensity}

STYLE RULES:
{rules}

STYLE ANCHORS (match vibe; don't copy verbatim):
{anchors}

TABOO (avoid): {taboo}

{extra_megaroast}
""".strip()

    return system


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()

    # raw JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # markdown-wrapped or extra text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        blob = text[start:end + 1]
        obj = json.loads(blob)
        if isinstance(obj, dict):
            return obj

    raise ValueError("Could not parse JSON object from Gemini output.")


def _call_gemini_flash(
    analysis: Dict[str, Any],
    base_tip: Dict[str, Any],
    style: str,
    intensity: int,
) -> Dict[str, Any]:
    import google.genai  # type: ignore

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    client = google.genai.Client(api_key=api_key)

    style = style if style in SUPPORTED_STYLES else "Supportive coach"
    intensity = int(max(0, min(2, intensity)))

    target_joint = base_tip.get("target_joint", "n/a")
    phase = base_tip.get("phase", "n/a")
    base_sentence = str(base_tip.get("one_sentence_tip", "")).strip()

    system = _build_system_prompt(style, intensity)

    user_payload = {
        "style": style,
        "intensity": intensity,
        "target_joint": target_joint,
        "phase": phase,
        "base_tip": base_sentence,
        "analysis_brief": {
            "rep_id": analysis.get("rep_id"),
            "events": analysis.get("events", [])[:3],
        },
    }

    prompt = system + "\n\nINPUT:\n" + json.dumps(user_payload, ensure_ascii=False)

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    resp = client.models.generate_content(model=model_name, contents=prompt)
    text = getattr(resp, "text", None) or ""

    out = _extract_json_object(text)

    tip_raw = str(out.get("one_sentence_tip", "")).strip()
    if not tip_raw:
        raise ValueError("Gemini returned empty one_sentence_tip")

    # Force invariants
    out["target_joint"] = target_joint
    out["phase"] = phase

    # Sanitize + enforce actionable movement cue
    out["one_sentence_tip"] = _sanitize_tip(tip_raw, target_joint=target_joint, phase=phase, style=style)

    return out


def generate_personality_tip(
    analysis: Dict[str, Any],
    base_tip: Dict[str, Any],
    style: str = "Supportive coach",
    intensity: int = 1,
) -> Dict[str, Any]:
    style = style if style in SUPPORTED_STYLES else "Supportive coach"
    intensity = int(max(0, min(2, intensity)))

    if not _gemini_available():
        return _fallback_personality(base_tip, style, intensity=intensity)

    try:
        out = _call_gemini_flash(analysis, base_tip, style, intensity=intensity)

        # final guardrail
        if _contains_forbidden(str(out.get("one_sentence_tip", ""))):
            return _fallback_personality(base_tip, style, intensity=intensity)

        # ensure still actionable
        if not _requires_actionable_tip(str(out.get("one_sentence_tip", "")), out.get("target_joint", "n/a"), out.get("phase", "n/a")):
            return _fallback_personality(base_tip, style, intensity=intensity)

        return out
    except Exception:
        return _fallback_personality(base_tip, style, intensity=intensity)
