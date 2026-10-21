"""A stand-in for Token Factory's OpenAI-compatible API, for tests.

Replies come from a script (a list of responses or exceptions, used in order)
or from a function of the request. Every request is recorded.
"""

from types import SimpleNamespace

import httpx
import openai


def response(content, finish_reason="stop", prompt_tokens=100, completion_tokens=50):
    """A chat completion in the shape the openai SDK returns."""
    message = SimpleNamespace(content=content)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def bad_request(message="response_format is not supported for this model"):
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    return openai.BadRequestError(message, response=httpx.Response(400, request=request), body=None)


class FakeTokenFactory:
    def __init__(self, replies, models=()):
        self._replies = replies if callable(replies) else list(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        listing = SimpleNamespace(data=[SimpleNamespace(id=m) for m in models])
        self.models = SimpleNamespace(list=lambda: listing)

    def _create(self, **request):
        self.requests.append(request)
        reply = self._replies(request) if callable(self._replies) else self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply
