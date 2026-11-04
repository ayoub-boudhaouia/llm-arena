import re

GROQ_LABEL = "Groq (cloud)"
CLOUD_MODELS = [GROQ_LABEL]
LOCAL_MODELS = ["qwen3:1.7b", "gemma3:4b", "llama3.1:8b"]

COMPLEX_KEYWORDS = [
    "explique en détail", "compare", "analyse", "démontre",
    "code", "programme", "pourquoi", "avantages et inconvénients",
]


def is_complex(prompt: str) -> bool:
    p = prompt.lower()
    return len(prompt) > 300 or any(k in p for k in COMPLEX_KEYWORDS)


def detect_sensitive(text: str) -> bool:
    """Détection simple par expressions régulières (démonstratif, pas infaillible)."""
    patterns = [
        r"\b\d{8}\b",
        r"[\w.-]+@[\w.-]+\.\w+",
        r"\b(?:\d[ -]?){13,16}\b",
        r"\b(confidentiel|mot de passe|password)\b",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def route(prompt: str, available: list, confidential: bool = False):
    """Renvoie (nom_du_modèle, raison)."""
    sensitive = detect_sensitive(prompt)
    local_only = confidential or sensitive

    candidates = [m for m in available if m in LOCAL_MODELS] if local_only else list(available)

    if is_complex(prompt):
        order, base = [GROQ_LABEL, "gemma3:4b", "llama3.1:8b"], "question complexe"
    else:
        order, base = ["qwen3:1.7b", "gemma3:4b"], "question simple"

    if confidential:
        base += ", mode confidentiel"
    elif sensitive:
        base += ", données sensibles détectées"

    for name in order:
        if name in candidates:
            return name, base
    return candidates[0], base + ", par défaut"