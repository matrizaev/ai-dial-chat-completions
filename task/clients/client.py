import logging
from collections.abc import AsyncIterator

from aidial_client import Dial, AsyncDial

from task.clients.base import BaseClient, InvalidResponse
from task.constants import DIAL_ENDPOINT
from task.models.message import Message
from task.models.role import Role


class DialClient(BaseClient):
    logger = logging.getLogger(__name__)

    def __init__(self, deployment_name: str):
        super().__init__(deployment_name)

        self.dial_client = Dial(api_key=self._api_key, base_url=DIAL_ENDPOINT)
        self.async_dial_client = AsyncDial(
            api_key=self._api_key, base_url=DIAL_ENDPOINT
        )


    def get_completion(self, messages: list[Message]) -> Message:
        completion = self.dial_client.chat.completions.create(
            deployment_name=self._deployment_name,
            stream=False,
            messages=[msg.to_dict() for msg in messages],
            api_version="2024-02-15-preview",
        )
        self.logger.info(completion)

        if not completion.choices:
            raise InvalidResponse("No choices in response found")

        content = completion.choices[0].message.content
        
        return Message(role=Role.AI, content=content)

    async def stream_completion(self, messages: list[Message]) -> AsyncIterator[Message]:
        completion = await self.async_dial_client.chat.completions.create(
            deployment_name=self._deployment_name,
            stream=True,
            messages=[msg.to_dict() for msg in messages],
            api_version="2024-02-15-preview",
        )

        async for chunk in completion:
            self.logger.info(chunk)
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            content_piece = delta.content if delta else None
            if not content_piece:
                continue

            yield Message(role=Role.AI, content=content_piece)
