import os
from langchain.callbacks.manager import CallbackManager
from langchain_openai import OpenAI


def prepare_model(
        model_name: str = None,
        base_url: str = None,
        temperature: float = None,
        callback_manager: CallbackManager = None,
) -> OpenAI:
    """
    Prepare the model.

    Args:
        model_name (str, optional): Name of the model to use. If left empty, will try to get it from the environment
        variable LLM_MODEL_NAME. Defaults to None.
        base_url (str, optional): Custom OpenAI API base url. If left empty, will try to get it from the environment
        variable LLM_BASE_URL. Defaults to None.
        temperature (float, optional): Temperature for the model generation. Defaults to None.
        callback_manager (CallbackManager, optional): Callback manager to use. Defaults to None.

    Raises:
        ValueError: If no model name or base URL is provided.

    Returns:
        OpenAI: Model instance
    """
    model_name = model_name or os.environ.get("LLM_MODEL_NAME", None)
    if model_name is None:
        raise ValueError("No model name provided.")

    # CRITICAL FIX: Re-added the environment variable fallback for base_url
    base_url = base_url or os.environ.get("LLM_BASE_URL", None)

    # If base_url is still None here, the code will hit the real OpenAI API and fail.
    # It is safer to raise an error early to prevent the 401 Unauthorized error.
    if base_url is None:
        raise ValueError("No base URL provided. Set LLM_BASE_URL environment variable for local vLLM.")

    # Cast temperature to float in case it comes from the environment as a string
    temperature_env = os.environ.get("LLM_TEMPERATURE", 0.2)
    temperature = temperature or float(temperature_env)

    model_kwargs = {
        "model": model_name,
        "callbacks": callback_manager,
        "api_key": "fake_keys",
        "temperature": temperature,
        "base_url": base_url,
        "stop": ["[/INST]", "</s>", "<|im_end|>", "<</SYS>>"],
    }

    return OpenAI(**model_kwargs)
