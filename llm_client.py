from langchain_ollama import ChatOllama

from config import (
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_TEMPERATURE,
)

def create_llm():
    """Create the shared ChatOllama client used by all agents."""

    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=OLLAMA_TEMPERATURE
    )


def safe_invoke(llm, prompt, default=""):
    """
    Invoke the LLM and return its text content.

    Never raises. If the call fails (Ollama not running, model
    not pulled, timeout, etc.) this logs the error and returns
    `default` instead of crashing the graph.
    """

    try:

        response = llm.invoke(prompt)

        content = getattr(
            response,
            "content",
            ""
        )

        return content.strip() if content else default

    except Exception as error:

        print(
            f"[LLM Error] Call failed: {error}"
        )

        return default
