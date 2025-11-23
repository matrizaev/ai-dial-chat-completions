from enum import StrEnum
import json
import logging
from collections.abc import AsyncIterator

import aiohttp
import requests

from task.clients.base import BaseClient, InvalidResponse
from task.constants import DIAL_ENDPOINT
from task.models.message import Message
from task.models.role import Role


class MessageType(StrEnum):
    MESSAGE = "message"
    DELTA = "delta"


class CustomDialClient(BaseClient):
    _endpoint: str
    _api_key: str

    logger = logging.getLogger(__name__)

    def __init__(self, deployment_name: str):
        super().__init__(deployment_name)
        self._endpoint = (
            DIAL_ENDPOINT + f"/openai/deployments/{deployment_name}/chat/completions"
        )

    def get_completion(self, messages: list[Message]) -> Message:
        headers = self._prepare_headers(self._api_key)

        request_data = self._prepare_request_data(
            messages, self._deployment_name, False
        )

        self.logger.info(request_data)
        response = requests.post(self._endpoint, headers=headers, json=request_data)

        response.raise_for_status()

        response_data = response.text
        self.logger.info(response_data)

        result = Message(
            role=Role.AI,
            content=self._get_content(response_data, message_key=MessageType.MESSAGE),
        )

        return result

    async def stream_completion(
        self, messages: list[Message]
    ) -> AsyncIterator[Message]:
        headers = self._prepare_headers(self._api_key)

        request_data = self._prepare_request_data(messages, self._deployment_name, True)

        self.logger.info(request_data)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._endpoint, json=request_data, headers=headers
            ) as response:
                response.raise_for_status()
                partial_line = ""
                pending_payload = ""
                async for chunk, _ in response.content.iter_chunks():
                    if not chunk:
                        continue
                    self.logger.info(chunk)
                    decoded = partial_line + chunk.decode("utf-8")
                    partial_line = ""

                    lines = decoded.splitlines()
                    if decoded and not decoded.endswith("\n"):
                        partial_line = lines.pop()

                    for line in lines:
                        if not line.startswith("data: "):
                            continue

                        payload_fragment = line[6:]
                        combined_payload = (pending_payload + payload_fragment).strip()
                        if combined_payload == "[DONE]":
                            return

                        try:
                            content_chunk = self._get_content(combined_payload)
                            pending_payload = ""
                        except InvalidResponse as exc:
                            if isinstance(exc.__cause__, json.JSONDecodeError):
                                pending_payload = pending_payload + payload_fragment
                                continue
                            raise

                        if not content_chunk:
                            continue
                        yield Message(role=Role.AI, content=content_chunk)

        raise InvalidResponse("Streaming response did not complete")

    @staticmethod
    def _get_content(
        payload: str, message_key: MessageType = MessageType.DELTA
    ) -> str | None:
        try:
            data = json.loads(payload)
            choices = data.get("choices") or []
            choice = choices[0] or {}
            message = choice.get(str(message_key)) or {}
            return message.get("content")
        except Exception as e:
            raise InvalidResponse(f"Failed to parse response: {e}") from e

    @staticmethod
    def _prepare_headers(api_key: str) -> dict[str, str]:
        return {"Content-Type": "application/json", "api-key": api_key}

    @staticmethod
    def _prepare_request_data(
        messages: list[Message], model: str, stream: bool
    ) -> dict:
        return {
            "model": model,
            "stream": stream,
            "messages": [msg.to_dict() for msg in messages],
        }
