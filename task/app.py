import asyncio
import logging

from task.clients.client import DialClient
from task.clients.custom_client import CustomDialClient
from task.constants import DEFAULT_SYSTEM_PROMPT
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


async def start(stream: bool) -> None:
    dial_client = DialClient("gpt-5-nano-2025-08-07")
    custom_client = CustomDialClient("gpt-5-nano-2025-08-07")

    conversation = Conversation()

    system_prompt = input(
        "Provide System prompt or press 'enter' to continue.\n> "
    ).strip()
    if not system_prompt:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    conversation.add_message(Message(role=Role.SYSTEM, content=system_prompt))

    use_custom = input("Use CustomDialClient? (y/N): ").strip().lower().startswith("y")
    client = custom_client if use_custom else dial_client

    print("Type your question or 'exit' to quit.")
    while True:
        user_input = input("> ").strip()
        if user_input.lower() == "exit":
            print("Exiting the chat. Goodbye!")
            break

        conversation.add_message(Message(role=Role.USER, content=user_input))

        if stream:
            chunk_content = []
            async for chunk in client.stream_completion(conversation.get_messages()):
                chunk_content.append(chunk.content)
            ai_message = Message(role=Role.AI, content="".join(chunk_content))
        else:
            ai_message = client.get_completion(conversation.get_messages())
        print("AI: ", ai_message.content)

        conversation.add_message(ai_message)


asyncio.run(start(True))
