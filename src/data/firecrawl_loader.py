"""Firecrawl web scraping loader."""

from typing import Optional

from firecrawl import FirecrawlApp
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings


class FirecrawlLoader:
    """Load documents from web pages using Firecrawl."""

    def __init__(
        self,
        url: str,
        crawl: bool = False,
        max_depth: int = 2,
        chunk_size: Optional[int] = None,
        chunk_overlap: int = 100,
    ):
        """Initialize Firecrawl loader.

        Args:
            url: The URL to scrape or crawl.
            crawl: Whether to crawl the entire site.
            max_depth: Maximum crawl depth.
            chunk_size: Optional chunk size for splitting.
            chunk_overlap: Overlap between chunks.
        """
        settings = get_settings()
        self.url = url
        self.crawl = crawl
        self.max_depth = max_depth
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._app = FirecrawlApp(api_key=settings.firecrawl_api_key)

    def load(self) -> list[Document]:
        """Load documents from the URL.

        Returns:
            List of documents extracted from the URL.
        """
        if self.crawl:
            return self._load_crawl()
        return self._load_scrape()

    def _load_scrape(self) -> list[Document]:
        """Scrape a single page.

        Returns:
            List with a single document from the page.
        """
        result = self._app.scrape_url(self.url, params={"formats": ["markdown"]})

        content = result.get("markdown", "")
        metadata = result.get("metadata", {})

        doc = Document(
            page_content=content,
            metadata={
                "source": metadata.get("sourceURL", self.url),
                "title": metadata.get("title", ""),
                "type": "web",
            },
        )

        if self.chunk_size:
            return self._split_document(doc)
        return [doc]

    def _load_crawl(self) -> list[Document]:
        """Crawl and scrape multiple pages.

        Returns:
            List of documents from all crawled pages.
        """
        result = self._app.crawl_url(
            self.url,
            params={
                "limit": 100,
                "scrapeOptions": {"formats": ["markdown"]},
            },
        )

        documents = []
        for page in result.get("data", []):
            content = page.get("markdown", "")
            metadata = page.get("metadata", {})

            doc = Document(
                page_content=content,
                metadata={
                    "source": metadata.get("sourceURL", self.url),
                    "title": metadata.get("title", ""),
                    "type": "web",
                },
            )

            if self.chunk_size:
                documents.extend(self._split_document(doc))
            else:
                documents.append(doc)

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
