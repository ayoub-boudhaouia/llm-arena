import os
import streamlit as st
from dotenv import load_dotenv

import db
from providers import OllamaProvider, GroqProvider
from router import route, detect_sensitive, GROQ_LABEL, CLOUD_MODELS, LOCAL_MODELS

load_dotenv()
db.init()
st.set_page_config(page_title="LLM Arena", layout="wide")

# ---------- Modèles disponibles ----------
PROVIDERS = {
    "qwen3:1.7b": lambda: OllamaProvider("qwen3:1.7b"),
    "gemma3:4b": lambda: OllamaProvider("gemma3:4b"),
    "llama3.1:8b": lambda: OllamaProvider("llama3.1:8b"),
}
if os.getenv("GROQ_API_KEY"):
    PROVIDERS[GROQ_LABEL] = lambda: GroqProvider()

AUTO = "Auto (routeur)"
FALLBACK_ORDER = ["gemma3:4b", "qwen3:1.7b"]

if "history" not in st.session_state:
    st.session_state.history = []


def safe_generate(name, messages, temperature, max_tokens):
    """Appelle un modèle sans jamais faire planter la page."""
    try:
        return PROVIDERS[name]().generate(messages, temperature, max_tokens)
    except Exception as e:
        return {"text": None, "error": str(e), "model": name,
                "seconds": 0, "tokens": None}


def generate_with_fallback(name, messages, temperature, max_tokens):
    """Si un modèle cloud échoue, bascule sur un modèle local (jamais l'inverse)."""
    res = safe_generate(name, messages, temperature, max_tokens)
    if res["text"] is None and name in CLOUD_MODELS:
        for fb in FALLBACK_ORDER:
            if fb in PROVIDERS:
                res2 = safe_generate(fb, messages, temperature, max_tokens)
                if res2["text"] is not None:
                    res2["fallback_from"] = name
                    res2["fallback_error"] = res["error"]
                    return res2
    return res


# ---------- Barre latérale ----------
with st.sidebar:
    mode = st.radio("Mode", ["Chat", "Comparaison", "Statistiques"])
    confidential = st.checkbox("Mode confidentiel (local uniquement)")
    temperature = st.slider("Température", 0.0, 1.5, 0.7, 0.1)
    max_tokens = st.slider("Max tokens", 50, 800, 500, 50)
    system_prompt = st.text_area(
        "Message system", "Tu es un assistant utile. Réponds en français."
    )
    if mode == "Chat":
        choice = st.selectbox("Modèle", [AUTO] + list(PROVIDERS))
        if st.button("Nouvelle conversation"):
            st.session_state.history = []
    if GROQ_LABEL not in PROVIDERS:
        st.info("Groq désactivé : GROQ_API_KEY absente du .env")

# ---------- Mode statistiques ----------
if mode == "Statistiques":
    st.title("Statistiques")
    df = db.read_all()
    if df.empty:
        st.info("Aucun appel enregistré pour le moment. Utilise le Chat ou la Comparaison.")
    else:
        ok = df[df["ok"] == 1]
        counts = df.groupby("model").agg(
            appels=("id", "count"),
            echecs=("ok", lambda s: int((s == 0).sum())),
            fallbacks=("fallback", "sum"),
        )
        perf = ok.groupby("model").agg(
            latence_moy_s=("seconds", "mean"),
            tokens_moy=("tokens", "mean"),
        )
        st.subheader("Résumé par modèle")
        st.dataframe(counts.join(perf).round(2))
        if not perf.empty:
            st.subheader("Latence moyenne (secondes)")
            st.bar_chart(perf["latence_moy_s"])
        st.subheader("20 derniers appels")
        st.dataframe(df.sort_values("id", ascending=False).head(20))
    st.stop()

# ---------- Mode comparaison ----------
if mode == "Comparaison":
    st.title("Comparaison de modèles")
    options = [m for m in PROVIDERS if m in LOCAL_MODELS] if confidential else list(PROVIDERS)
    selected = st.multiselect(
        "Modèles à comparer", options,
        default=[m for m in ["qwen3:1.7b", "gemma3:4b"] if m in options],
    )
    prompt = st.chat_input("Prompt à comparer...")
    if prompt and selected:
        if detect_sensitive(prompt):
            if any(m in CLOUD_MODELS for m in selected):
                st.warning("Données sensibles détectées : modèles cloud ignorés.")
            selected = [m for m in selected if m in LOCAL_MODELS]
        st.write(f"**Prompt :** {prompt}")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        if selected:
            cols = st.columns(len(selected))
            for col, name in zip(cols, selected):
                with col:
                    st.subheader(name)
                    with st.spinner("Génération..."):
                        res = safe_generate(name, messages, temperature, max_tokens)
                    ok = res["text"] is not None
                    db.log(name, "comparaison", res["seconds"], res["tokens"],
                           ok=ok, prompt_len=len(prompt))
                    if not ok:
                        st.error(f"Échec : {res['error']}")
                    else:
                        st.write(res["text"])
                        st.caption(f"{res['seconds']:.1f} s · {res['tokens']} tokens")
    st.stop()

# ---------- Mode chat ----------
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.write(m["content"])

if prompt := st.chat_input("Pose ta question..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    local_only = confidential or detect_sensitive(prompt)

    if choice == AUTO or (local_only and choice in CLOUD_MODELS):
        model_name, reason = route(prompt, list(PROVIDERS), confidential)
        if choice != AUTO:
            st.warning(f"{choice} refusé (données confidentielles) : modèle local utilisé.")
    else:
        model_name, reason = choice, "choix manuel"

    messages = [{"role": "system", "content": system_prompt}] + st.session_state.history
    with st.chat_message("assistant"):
        with st.spinner(f"Génération avec {model_name}..."):
            res = generate_with_fallback(model_name, messages, temperature, max_tokens)

        if res["text"] is None:
            db.log(model_name, "chat", 0, None, ok=False,
                   reason=reason, prompt_len=len(prompt))
            st.error(f"Échec : {res['error']}")
            st.session_state.history.pop()
        else:
            used_fallback = bool(res.get("fallback_from"))
            if used_fallback:
                st.warning(
                    f"{res['fallback_from']} indisponible "
                    f"({res['fallback_error'][:120]}) : bascule sur {res['model']}."
                )
                reason = "fallback"
            db.log(res["model"], "chat", res["seconds"], res["tokens"], ok=True,
                   fallback=used_fallback, reason=reason, prompt_len=len(prompt))
            st.write(res["text"])
            st.caption(
                f"{res['model']} ({reason}) · {res['seconds']:.1f} s · {res['tokens']} tokens"
            )
            st.session_state.history.append(
                {"role": "assistant", "content": res["text"]}
            )