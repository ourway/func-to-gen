"""API routes for OpenAI/Ollama compatible endpoints."""

import os

from flask import Blueprint, jsonify, request

from func_to_gen.utils import (
    format_chat_completion_response,
    format_completion_response,
    format_models_response,
    format_ollama_chat_response,
    format_ollama_generate_response,
    format_ollama_tags_response,
    messages_to_prompt,
)

# OpenAI-compatible API blueprint
api = Blueprint("api", __name__, url_prefix="/v1")

# Ollama native API blueprint
ollama_api = Blueprint("ollama_api", __name__, url_prefix="/api")

# Model name can be configured via environment variable
MODEL_NAME = os.environ.get("MODEL_NAME", "local-llm")

# The answer function will be set by the app factory
_answer_func = None


def set_answer_function(func):
    """Set the answer function to use for generating responses."""
    global _answer_func
    _answer_func = func


def get_answer_function():
    """Get the configured answer function."""
    if _answer_func is None:
        raise RuntimeError("Answer function not configured. Call set_answer_function first.")
    return _answer_func


@api.route("/chat/completions", methods=["POST"])
def chat_completions():
    """Handle chat completion requests (OpenAI format)."""
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": {"message": "Request body is required", "type": "invalid_request_error"}}), 400

    messages = data.get("messages", [])
    if not messages:
        return jsonify({"error": {"message": "messages is required", "type": "invalid_request_error"}}), 400

    # Convert messages to a single prompt
    prompt = messages_to_prompt(messages)

    # Get the answer
    answer_func = get_answer_function()
    response_content = answer_func(prompt)

    # Get model from request or use default
    model = data.get("model", MODEL_NAME)

    return jsonify(format_chat_completion_response(response_content, model=model))


@api.route("/completions", methods=["POST"])
def completions():
    """Handle legacy completion requests."""
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": {"message": "Request body is required", "type": "invalid_request_error"}}), 400

    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": {"message": "prompt is required", "type": "invalid_request_error"}}), 400

    # Get the answer
    answer_func = get_answer_function()
    response_content = answer_func(prompt)

    # Get model from request or use default
    model = data.get("model", MODEL_NAME)

    return jsonify(format_completion_response(response_content, model=model))


@api.route("/models", methods=["GET"])
def list_models():
    """List available models."""
    return jsonify(format_models_response(MODEL_NAME))


@api.route("/models/<model_id>", methods=["GET"])
def get_model(model_id: str):
    """Get a specific model."""
    if model_id != MODEL_NAME:
        return jsonify({"error": {"message": f"Model {model_id} not found", "type": "invalid_request_error"}}), 404

    return jsonify({
        "id": MODEL_NAME,
        "object": "model",
        "owned_by": "local",
    })


@api.route("/embeddings", methods=["POST"])
def embeddings():
    """Embeddings endpoint - not supported."""
    return jsonify({
        "error": {
            "message": "Embeddings are not supported. This API only wraps a text generation function.",
            "type": "not_implemented_error",
        }
    }), 501


# =============================================================================
# Ollama Native API Endpoints
# =============================================================================

@ollama_api.route("/generate", methods=["POST"])
def ollama_generate():
    """Handle Ollama native generate requests."""
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Request body is required"}), 400

    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    # Get the answer
    answer_func = get_answer_function()
    response_content = answer_func(prompt)

    # Get model from request or use default
    model = data.get("model", MODEL_NAME)

    return jsonify(format_ollama_generate_response(response_content, model=model))


@ollama_api.route("/chat", methods=["POST"])
def ollama_chat():
    """Handle Ollama native chat requests."""
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Request body is required"}), 400

    messages = data.get("messages", [])
    if not messages:
        return jsonify({"error": "messages is required"}), 400

    # Convert messages to a single prompt
    prompt = messages_to_prompt(messages)

    # Get the answer
    answer_func = get_answer_function()
    response_content = answer_func(prompt)

    # Get model from request or use default
    model = data.get("model", MODEL_NAME)

    return jsonify(format_ollama_chat_response(response_content, model=model))


@ollama_api.route("/tags", methods=["GET"])
def ollama_tags():
    """List available models (Ollama format)."""
    return jsonify(format_ollama_tags_response(MODEL_NAME))


@ollama_api.route("/show", methods=["POST"])
def ollama_show():
    """Show model information."""
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Request body is required"}), 400

    model = data.get("model", MODEL_NAME)

    return jsonify({
        "modelfile": f"FROM {model}",
        "parameters": "",
        "template": "",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": "local",
            "families": ["local"],
            "parameter_size": "unknown",
            "quantization_level": "unknown",
        },
    })


@ollama_api.route("/embeddings", methods=["POST"])
def ollama_embeddings():
    """Ollama embeddings endpoint - not supported."""
    return jsonify({"error": "Embeddings are not supported. This API only wraps a text generation function."}), 501
