import os

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from langchain.vectorstores.faiss import FAISS

from ticc4chat.llm.llm import prepare_model
from ticc4chat.llm.prompt import prepare_prompt


def prepare_chain(
    llm: Ollama = None, prompt: PromptTemplate = None
) -> ConversationalRetrievalChain:
    """
    Prepares a retrieval-based question answering system based on a given machine and language.

    Args:
        llm (Ollama, optional): The LLM model to use for the question answering system. If not provided, the default model is used.
        prompt (PromptTemplate, optional): The prompt to use for the question answering system. If not provided, the default prompt is used.

    Returns:
        ConversationalRetrievalChain: A retrieval-based question answering system based on the given parameters.
    """

    if llm is None:
        # if the model is not provided, we use the default one
        # this means we rely on the environment variables LLM_MODEL_NAME and LLM_BASE_URL
        llm = prepare_model()

    if prompt is None:
        # if prompt is not provided, we use the default one
        prompt = prepare_prompt()

    # embedding_model = HuggingFaceEmbeddings(
    #     model_name=os.environ.get("EMBEDDINGS_MODEL_NAME", None)
    # )

    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=os.environ.get("EMBEDDINGS_MODEL_NAME", None),
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    vectorstore = FAISS.load_local(
        os.environ.get("VECTORSTORES_DIR", None) + os.sep + "vectorstore",
        embedding_model,
    )

    NB_RETRIVED_DOCS = int(os.environ.get("NB_RETRIVED_DOCS", 4))

    return RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": NB_RETRIVED_DOCS}),
        chain_type_kwargs={"prompt": prompt},
        input_key="question",
        output_key="answer",
        return_source_documents=True,
    )


def ask_chain(query: str, chain: ConversationalRetrievalChain) -> str:
    """
    Ask the chain a question.

    Args:
        query (str): question to ask

    Returns:
        str: answer
    """

    result = chain(
        {
            "question": query,
        }
    )

    # not interested in the whole path just the name and extension
    prepare_document = lambda x: x if x is None else os.path.basename(x)

    prepare_page = lambda x: x if x is None else int(x) + 1

    prepare_source = lambda x: {
        "document": prepare_document(x.metadata.get("source", None)),
        "page": prepare_page(x.metadata.get("page", None)),
        "chunk": x.page_content,
    }

    return {
        "answer": result.get("answer"),
        "sources": [prepare_source(s) for s in result.get("source_documents")],
    }
