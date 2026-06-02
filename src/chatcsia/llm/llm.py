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
        model_name (str, optional): name of the model to use. If left empty, will try to get it from the environment
        variable LLM_MODEL_NAME. Defaults to None.
        base_url (str, optional): custom OpenAI API base url. If left empty, the default OpenAI endpoint is used.
        temperature (float, optional): temperature for the model generation. Defaults to 0.1.
        callback_manager (CallbackManager, optional): callback manager to use. Defaults to None.

    Raises:
        ValueError: if no model name is provided and no environment variable is set.

    Returns:
        OpenAI: model instance
    """
    model_name = model_name or os.environ.get("LLM_MODEL_NAME", None)
    if model_name is None:
        raise ValueError("No model name provided.")

    temperature = temperature or os.environ.get("LLM_TEMPERATURE", 0.2)

    model_kwargs = {
        "model": model_name,
        "callback_manager": callback_manager,
        "openai_api_key": "fake_keys",
        "temperature": temperature,
        "stop": ["[/INST]", "</s>", "<|im_end|>", "<</SYS>>"],
    }
    if base_url is not None:
        model_kwargs["base_url"] = base_url

    return OpenAI(**model_kwargs)
