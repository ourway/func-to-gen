"""Utility functions for formatting OpenAI-compatible responses."""

import time
import uuid


def generate_id(prefix: str = "chatcmpl") -> str:
    """Generate a unique ID for API responses."""
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


def get_timestamp() -> int:
    """Get current Unix timestamp."""
    return int(time.time())


def format_chat_completion_response(
    content: str,
    model: str = "local-llm",
    finish_reason: str = "stop",
) -> dict:
    """Format a response in OpenAI chat completion format."""
    return {
        "id": generate_id("chatcmpl"),
        "object": "chat.completion",
        "created": get_timestamp(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def format_completion_response(
    content: str,
    model: str = "local-llm",
    finish_reason: str = "stop",
) -> dict:
    """Format a response in OpenAI legacy completion format."""
    return {
        "id": generate_id("cmpl"),
        "object": "text_completion",
        "created": get_timestamp(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": content,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def format_models_response(model_name: str = "local-llm") -> dict:
    """Format a response for the models list endpoint."""
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": get_timestamp(),
                "owned_by": "local",
            }
        ],
    }


def messages_to_prompt(messages: list[dict]) -> str:
    """Convert OpenAI chat messages to a single prompt string.

    Format:
    system: <system message>
    user: <user message>
    assistant: <assistant message>
    ...
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


# Ollama native API response formatters

def get_iso_timestamp() -> str:
    """Get current timestamp in ISO format."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def format_ollama_generate_response(
    response: str,
    model: str = "local-llm",
) -> dict:
    """Format a response in Ollama /api/generate format."""
    return {
        "model": model,
        "created_at": get_iso_timestamp(),
        "response": response,
        "done": True,
        "done_reason": "stop",
        "context": [],
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": 0,
        "prompt_eval_duration": 0,
        "eval_count": 0,
        "eval_duration": 0,
    }


def format_ollama_chat_response(
    content: str,
    model: str = "local-llm",
) -> dict:
    """Format a response in Ollama /api/chat format."""
    return {
        "model": model,
        "created_at": get_iso_timestamp(),
        "message": {
            "role": "assistant",
            "content": content,
        },
        "done": True,
        "done_reason": "stop",
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": 0,
        "prompt_eval_duration": 0,
        "eval_count": 0,
        "eval_duration": 0,
    }


def format_ollama_tags_response(model_name: str = "local-llm") -> dict:
    """Format a response for Ollama /api/tags endpoint."""
    return {
        "models": [
            {
                "name": model_name,
                "model": model_name,
                "modified_at": get_iso_timestamp(),
                "size": 0,
                "digest": "",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "local",
                    "families": ["local"],
                    "parameter_size": "unknown",
                    "quantization_level": "unknown",
                },
            }
        ]
    }
