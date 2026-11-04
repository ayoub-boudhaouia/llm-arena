from providers import OllamaProvider


if __name__ == "__main__":
    provider = OllamaProvider("qwen3:1.7b")
    response = provider.generate([{"role": "user", "content": "Bonjour"}])

    text = response["text"]
    if not text or not text.strip():
        raise RuntimeError("Le provider n'a renvoyé aucune réponse.")

    print(text)
