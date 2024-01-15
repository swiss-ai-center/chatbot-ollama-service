from pathlib import Path
from typing import Any, Iterator, List, Optional

from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.document_loaders import PyMuPDFLoader, PyPDFDirectoryLoader
from langchain.document_loaders.blob_loaders import Blob
from langchain.document_loaders.parsers import PyMuPDFParser
from loguru import logger


class PyMuPDFLoaderImage(PyMuPDFLoader):
    def load(self, **kwargs: Optional[Any]) -> List[Document]:
        """Load file."""

        # need to force option to work
        kwargs["option"] = "xhtml"

        parser = PyMuPDFParserImage(text_kwargs=kwargs)
        blob = Blob.from_path(self.file_path)
        return parser.parse(blob)


class PyPDFDirectoryLoaderImage(PyPDFDirectoryLoader):
    def load(self) -> List[Document]:
        p = Path(self.path)
        docs = []
        items = p.rglob(self.glob) if self.recursive else p.glob(self.glob)
        for i in items:
            if i.is_file() and (self._is_visible(i.relative_to(p)) or self.load_hidden):
                try:
                    loader = PyMuPDFLoaderImage(str(i))
                    sub_docs = loader.load()
                    for doc in sub_docs:
                        doc.metadata["source"] = str(i)
                    docs.extend(sub_docs)
                except Exception as e:
                    if self.silent_errors:
                        logger.warning(e)
                    else:
                        raise e

        return docs


class PyMuPDFParserImage(PyMuPDFParser):
    """Parse `PDF` using `PyMuPDF`."""

    def prepare_page(self, raw_page_content: str) -> str:
        soup = BeautifulSoup(raw_page_content, features="html.parser")
        # remove the content of src for all img tags as it is not needed and would be a lot of data
        for img in soup.find_all("img"):
            img["src"] = ""

        return soup.prettify()

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import fitz

        with blob.as_bytes_io() as file_path:
            doc = fitz.open(file_path)  # open document

            yield from [
                Document(
                    page_content=self.prepare_page(page.get_text(**self.text_kwargs)),
                    metadata=dict(
                        {
                            "source": blob.source,
                            "file_path": blob.source,
                            "page": page.number,
                            "total_pages": len(doc),
                        },
                        **{
                            k: doc.metadata[k]
                            for k in doc.metadata
                            if type(doc.metadata[k]) in [str, int]
                        },
                    ),
                )
                for page in doc
            ]
