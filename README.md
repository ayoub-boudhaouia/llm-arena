# LLM Arena

Chatbot multi-LLM avec routage intelligent : modèles locaux (Ollama) et cloud (Groq).

## Fonctionnalités
- Chat avec mémoire et choix du modèle
- Mode comparaison côte à côte
- Routeur automatique (question simple ou complexe)
- Mode confidentiel : local uniquement, avec détection de données sensibles
- Fallback automatique du cloud vers le local
- Statistiques par modèle (SQLite)

## Installation
```bash
pip install -r requirements.txt
```
1. Installer Ollama et télécharger les modèles : `ollama pull qwen3:1.7b`, `ollama pull gemma3:4b`, `ollama pull llama3.1:8b`
2. Copier `.env.example` en `.env` et renseigner `GROQ_API_KEY` (optionnel : sans clé, seuls les modèles locaux sont utilisés)

## Lancement
```bash
streamlit run app.py
```