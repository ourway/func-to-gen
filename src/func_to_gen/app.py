"""Flask application factory."""

from flask import Flask

from func_to_gen.routes import api, ollama_api, set_answer_function


def create_app(answer_func=None, config=None):
    """Create and configure the Flask application.

    Args:
        answer_func: The function to use for generating responses.
                    Should have signature: answer(prompt: str) -> str
        config: Optional configuration dictionary.

    Returns:
        Configured Flask application.

    Example:
        from llm import answer
        from func_to_gen import create_app

        app = create_app(answer_func=answer)
        app.run(host="0.0.0.0", port=5000)
    """
    app = Flask(__name__)

    if config:
        app.config.update(config)

    # Set the answer function if provided
    if answer_func is not None:
        set_answer_function(answer_func)

    # Register the API blueprints
    app.register_blueprint(api)          # OpenAI-compatible: /v1/*
    app.register_blueprint(ollama_api)   # Ollama native: /api/*

    # Health check endpoint
    @app.route("/health")
    def health():
        return {"status": "ok"}

    return app


# For running directly with `flask run` or `python -m func_to_gen.app`
if __name__ == "__main__":
    # Import the real answer function when running directly
    try:
        from llm import answer
        app = create_app(answer_func=answer)
    except ImportError:
        # Fallback for testing without the llm module
        def mock_answer(prompt: str) -> str:
            return f"Mock response to: {prompt}"
        app = create_app(answer_func=mock_answer)

    app.run(host="0.0.0.0", port=5000, debug=True)
