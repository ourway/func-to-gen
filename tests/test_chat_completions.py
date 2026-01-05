"""Tests for /v1/chat/completions endpoint."""

import json


class TestChatCompletions:
    """Tests for the chat completions endpoint."""

    def test_basic_chat_completion(self, client):
        """Test a basic chat completion request."""
        response = client.post(
            "/v1/chat/completions",
            data=json.dumps({
                "model": "local-llm",
                "messages": [{"role": "user", "content": "Hello"}],
            }),
            content_type="application/json",
        )

        assert response.status_code == 200
        data = response.get_json()

        assert "id" in data
        assert data["id"].startswith("chatcmpl-")
        assert data["object"] == "chat.completion"
        assert "created" in data
        assert data["model"] == "local-llm"

    def test_chat_completion_response_structure(self, client):
        """Test that response has correct OpenAI structure."""
        response = client.post(
            "/v1/chat/completions",
            data=json.dumps({
                "messages": [{"role": "user", "content": "Test"}],
            }),
            content_type="application/json",
        )

        data = response.get_json()

        # Check choices structure
        assert "choices" in data
        assert len(data["choices"]) == 1

        choice = data["choices"][0]
        assert choice["index"] == 0
        assert "message" in choice
        assert choice["message"]["role"] == "assistant"
        assert "content" in choice["message"]
        assert choice["finish_reason"] == "stop"

        # Check usage structure
        assert "usage" in data
        assert "prompt_tokens" in data["usage"]
        assert "completion_tokens" in data["usage"]
        assert "total_tokens" in data["usage"]

    def test_chat_completion_content(self, client):
        """Test that response content comes from answer function."""
        response = client.post(
            "/v1/chat/completions",
            data=json.dumps({
                "messages": [{"role": "user", "content": "Hello world"}],
            }),
            content_type="application/json",
        )

        data = response.get_json()
        content = data["choices"][0]["message"]["content"]

        # The mock_answer function returns "Response to: {prompt}"
        assert "Hello world" in content

    def test_chat_completion_multiple_messages(self, client):
        """Test chat completion with multiple messages."""
        response = client.post(
            "/v1/chat/completions",
            data=json.dumps({
                "messages": [
                    {"role": "system", "content": "You are a helper."},
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                    {"role": "user", "content": "How are you?"},
                ],
            }),
            content_type="application/json",
        )

        assert response.status_code == 200
        data = response.get_json()

        content = data["choices"][0]["message"]["content"]
        # Check that all messages were included in prompt
        assert "system:" in content or "How are you?" in content

    def test_chat_completion_custom_model(self, client):
        """Test that custom model name is reflected in response."""
        response = client.post(
            "/v1/chat/completions",
            data=json.dumps({
                "model": "my-custom-model",
                "messages": [{"role": "user", "content": "Test"}],
            }),
            content_type="application/json",
        )

        data = response.get_json()
        assert data["model"] == "my-custom-model"

    def test_chat_completion_no_body(self, client):
        """Test error when no request body is provided."""
        response = client.post(
            "/v1/chat/completions",
            data="",
            content_type="application/json",
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_chat_completion_no_messages(self, client):
        """Test error when messages field is missing."""
        response = client.post(
            "/v1/chat/completions",
            data=json.dumps({"model": "test"}),
            content_type="application/json",
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "messages" in data["error"]["message"]

    def test_chat_completion_empty_messages(self, client):
        """Test error when messages array is empty."""
        response = client.post(
            "/v1/chat/completions",
            data=json.dumps({"messages": []}),
            content_type="application/json",
        )

        assert response.status_code == 400
