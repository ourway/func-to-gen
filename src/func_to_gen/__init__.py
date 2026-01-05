"""Flask app that wraps a function into OpenAI/Ollama compatible API."""

from func_to_gen.app import create_app

__all__ = ["create_app"]
