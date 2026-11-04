import os
import time

from dotenv import load_dotenv
from groq import Groq
from ollama import chat as ollama_chat

load_dotenv()


class OllamaProvider:
    kind = "local"

    def __init__(self, model):
        self.name = model
        self.model = model

    def generate(self, messages, temperature=0.7, max_tokens=300):
        start = time.time()
        r = ollama_chat(
            model=self.model,
            messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens},
            think=False,
            stream=False,
        )
        return {
            "text": r.message.content,
            "model": self.name,
            "seconds": time.time() - start,
            "tokens": r.eval_count,  # Ollama : tokens générés
        }


class GroqProvider:
    kind = "cloud"

    def __init__(self, model=None):
        model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.name = f"groq/{model}"
        self.model = model
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY absente du .env")
        self.client = Groq(api_key=key)

    def generate(self, messages, temperature=0.7, max_tokens=300):
        start = time.time()
        r = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        return {
            "text": r.choices[0].message.content,
            "model": self.name,
            "seconds": time.time() - start,
            "tokens": r.usage.completion_tokens if r.usage else None,  # Groq : tokens générés
        }