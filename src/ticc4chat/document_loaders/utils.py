from typing import Type

from langchain.document_loaders.pdf import BasePDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

from ticc4chat.document_loaders.pdf import PyMuPDFLoaderImage


def single_pdf_to_embeddings(
    pdf_path: str,
    pdf_loader_class: Type[BasePDFLoader] = None,
    embedding: Embeddings = None,
    vectorstore_class: Type[VectorStore] = None,
) -> VectorStore:
    """
    Convert a single pdf to embeddings.

    Args:
        pdf_path (str): path to pdf
        pdf_loader_class (Type[BasePDFLoader], optional): pdf loader class. Defaults to PyMuPDFLoaderImage.
        embedding (Embeddings, optional): embedding instance to use. Defaults to GPT4AllEmbeddings.
        vectorstore_class (Type[VectorStore], optional): vectorstore class. Defaults to Chroma.

    Returns:
        VectorStore: vectorstore instance of the pdf
    """
    if pdf_loader_class is None:
        pdf_loader_class = PyMuPDFLoaderImage

    if embedding is None:
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if vectorstore_class is None:
        vectorstore_class = FAISS

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(pdf_loader_class(pdf_path).load())

    return vectorstore_class.from_documents(documents=all_splits, embedding=embedding)
