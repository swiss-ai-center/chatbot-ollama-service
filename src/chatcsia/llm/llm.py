import os

from langchain.callbacks.manager import CallbackManager
from langchain.llms.ollama import Ollama


def prepare_model(
    model_name: str = None,
    base_url: str = None,
    temperature: float = None,
    callback_manager: CallbackManager = None,
) -> Ollama:
    """
    Prepare the model.

    Args:
        model_name (str, optional): name of the model to use. If left empty, will try to get it from the environment
        variable LLM_MODEL_NAME. Defaults to None.
        base_url (str, optional): base url of the model to use. If left empty, will try to get it from the environment
        variable LLM_BASE_URL. Defaults to None.
        temperature (float, optional): temperature for the model generation. Defaults to 0.1.
        callback_manager (CallbackManager, optional): callback manager to use. Defaults to None.

    Raises:
        ValueError: if no model name or base url is provided and no environment variable is set.

    Returns:
        Ollama: model instance
    """
    model_name = model_name or os.environ.get("LLM_MODEL_NAME", None)
    if model_name is None:
        raise ValueError("No model name provided.")

    base_url = base_url or os.environ.get("LLM_BASE_URL", None)
    if base_url is None:
        raise ValueError("No base url provided.")

    temperature = temperature or os.environ.get("LLM_TEMPERATURE", 0.2)

    return Ollama(
        model=model_name,
        callback_manager=callback_manager,
        base_url=base_url,
        temperature=temperature,
        stop=["[/INST]", "</s>", "<|im_end|>", "<</SYS>>"],
    )
