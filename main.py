import chainlit as cl
import ollama
from typing import Generator


def get_streamed_response(messages: list) -> Generator[str, None, None]:
    """
    Stream response from Ollama model
    """
    response = ollama.chat(
        model="deepseek-r1",
        messages=messages,
        stream=True
    )

    for chunk in response:
        if chunk.message:
            yield chunk.message.content


@cl.on_chat_start
async def start_chat():
    cl.user_session.set(
        "interaction",
        [
            {
                "role": "system",
                "content": "You are a helpful assistant. Talk like a guy from the hood",
            }
        ],
    )

    msg = cl.Message(content="")
    start_message = """Hello, I'm your 100% local alternative to ChatGPT 
                      running on DeepSeek-R1. How can I help you today?"""

    for token in start_message:
        await msg.stream_token(token)

    await msg.send()


@cl.on_message
async def main(message: cl.Message):
    interaction = cl.user_session.get("interaction")

    # Add user message to interaction
    interaction.append({
        "role": "user",
        "content": message.content
    })

    msg = cl.Message(content="")
    response_content = ""

    # Stream the response token by token
    response_stream = get_streamed_response(interaction)
    for token in response_stream:
        await msg.stream_token(token)
        response_content += token

    # After streaming is complete, add assistant's message to interaction
    interaction.append({
        "role": "assistant",
        "content": response_content
    })

    cl.user_session.set("interaction", interaction)
    await msg.send()


if __name__ == "__main__":
    print("Starting chat application...")