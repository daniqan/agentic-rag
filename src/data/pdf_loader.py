"""PDF document loader."""

from pathlib import Path
from typing import Optional, Union

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from pypdf import PdfReader


class PDFLoader:
    """Load documents from PDF files."""

    def __init__(
        self,
        file_path: Union[str, Path],
        chunk_size: Optional[int] = None,
        chunk_overlap: int = 100,
    ):
        """Initialize PDF loader.

        Args:
            file_path: Path to the PDF file.
            chunk_size: Optional chunk size for splitting.
            chunk_overlap: Overlap between chunks.
        """
        self.file_path = Path(file_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load(self) -> list[Document]:
        """Load documents from the PDF file.

        Returns:
            List of documents, one per page.

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.file_path}")

        reader = PdfReader(self.file_path)
        documents = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""

            if not text.strip():
                continue

            doc = Document(
                page_content=text,
                metadata={
                    "source": str(self.file_path),
                    "page": i + 1,
                    "total_pages": len(reader.pages),
                    "type": "pdf",
                },
            )

            if self.chunk_size:
                documents.extend(self._split_document(doc))
            else:
                documents.append(doc)

        return documents

    def load_and_split(self, text_splitter: Optional[TextSplitter] = None) -> list[Document]:
        """Load and split PDF content with a custom splitter.

        Args:
            text_splitter: Optional custom text splitter.

        Returns:
            List of split documents.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.file_path}")

        reader = PdfReader(self.file_path)
        documents = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""

            if not text.strip():
                continue

            doc = Document(
                page_content=text,
                metadata={
                    "source": str(self.file_path),
                    "page": i + 1,
                    "total_pages": len(reader.pages),
                    "type": "pdf",
                },
            )
            documents.append(doc)

        if text_splitter:
            return text_splitter.split_documents(documents)

        return documents

    def _split_document(self, doc: Document) -> list[Document]:
        """Split a document into chunks.

        Args:
            doc: The document to split.

        Returns:
            List of chunked documents.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return splitter.split_documents([doc])
