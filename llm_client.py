from langchain_ollama import ChatOllama

from config import (
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_TEMPERATURE,
)

import logging

logger = logging.getLogger(__name__)

# ============================================
# LLM SINGLETON
# ============================================

_llm_instance = None


def create_llm():
    """
    Return the shared ChatOllama client used by all agents.

    The instance is created once and reused for the lifetime
    of the process, avoiding repeated connection overhead.
    """

    global _llm_instance

    if _llm_instance is None:

        logger.debug(
            "Creating ChatOllama instance "
            "(model=%s, url=%s)",
            OLLAMA_MODEL,
            OLLAMA_BASE_URL,
        )

        _llm_instance = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=OLLAMA_TEMPERATURE,
        )

    return _llm_instance


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

        logger.error(
            "[LLM Error] Call failed: %s", error
        )

        return default
