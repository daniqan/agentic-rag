"""Tests for data extraction modules."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.data.firecrawl_loader import FirecrawlLoader
from src.data.pdf_loader import PDFLoader


class TestFirecrawlLoader:
    """Test FirecrawlLoader class."""

    def test_init_with_defaults(self, mock_settings):
        """FirecrawlLoader should initialize with default settings."""
        with patch("src.data.firecrawl_loader.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.data.firecrawl_loader.FirecrawlApp") as mock_fc:
                loader = FirecrawlLoader(url="https://example.com")

                assert loader.url == "https://example.com"
                mock_fc.assert_called_once_with(api_key=mock_settings.firecrawl_api_key)

    def test_load_single_page(self, mock_settings):
        """load should return documents from a single page."""
        with patch("src.data.firecrawl_loader.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.data.firecrawl_loader.FirecrawlApp") as mock_fc:
                mock_app = MagicMock()
                mock_app.scrape_url.return_value = {
                    "markdown": "# Test Content\n\nThis is test content.",
                    "metadata": {
                        "title": "Test Page",
                        "sourceURL": "https://example.com",
                    },
                }
                mock_fc.return_value = mock_app

                loader = FirecrawlLoader(url="https://example.com")
                docs = loader.load()

                assert len(docs) == 1
                assert isinstance(docs[0], Document)
                assert "Test Content" in docs[0].page_content
                assert docs[0].metadata["source"] == "https://example.com"

    def test_load_with_crawl(self, mock_settings):
        """load with crawl=True should crawl and return multiple documents."""
        with patch("src.data.firecrawl_loader.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.data.firecrawl_loader.FirecrawlApp") as mock_fc:
                mock_app = MagicMock()
                mock_app.crawl_url.return_value = {
                    "data": [
                        {
                            "markdown": "Page 1 content",
                            "metadata": {"sourceURL": "https://example.com/page1"},
                        },
                        {
                            "markdown": "Page 2 content",
                            "metadata": {"sourceURL": "https://example.com/page2"},
                        },
                    ]
                }
                mock_fc.return_value = mock_app

                loader = FirecrawlLoader(url="https://example.com", crawl=True)
                docs = loader.load()

                assert len(docs) == 2
                mock_app.crawl_url.assert_called_once()

    def test_load_with_chunk_size(self, mock_settings):
        """load with chunk_size should split content into chunks."""
        with patch("src.data.firecrawl_loader.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.data.firecrawl_loader.FirecrawlApp") as mock_fc:
                mock_app = MagicMock()
                mock_app.scrape_url.return_value = {
                    "markdown": "A" * 1000,
                    "metadata": {"sourceURL": "https://example.com"},
                }
                mock_fc.return_value = mock_app

                loader = FirecrawlLoader(
                    url="https://example.com", chunk_size=100, chunk_overlap=10
                )
                docs = loader.load()

                assert len(docs) > 1


class TestPDFLoader:
    """Test PDFLoader class."""

    def test_init(self):
        """PDFLoader should initialize with file path."""
        loader = PDFLoader(file_path="/path/to/test.pdf")
        assert loader.file_path == Path("/path/to/test.pdf")

    def test_load_pdf(self, tmp_path):
        """load should return documents from PDF pages."""
        with patch("src.data.pdf_loader.PdfReader") as mock_reader:
            mock_page1 = MagicMock()
            mock_page1.extract_text.return_value = "Page 1 text content"
            mock_page2 = MagicMock()
            mock_page2.extract_text.return_value = "Page 2 text content"
            mock_reader.return_value.pages = [mock_page1, mock_page2]

            pdf_path = tmp_path / "test.pdf"
            pdf_path.write_bytes(b"PDF content")

            loader = PDFLoader(file_path=str(pdf_path))
            docs = loader.load()

            assert len(docs) == 2
            assert isinstance(docs[0], Document)
            assert "Page 1" in docs[0].page_content
            assert docs[0].metadata["page"] == 1
            assert docs[0].metadata["source"] == str(pdf_path)

    def test_load_pdf_with_chunking(self, tmp_path):
        """load with chunk_size should split content into chunks."""
        with patch("src.data.pdf_loader.PdfReader") as mock_reader:
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "A" * 500
            mock_reader.return_value.pages = [mock_page]

            pdf_path = tmp_path / "test.pdf"
            pdf_path.write_bytes(b"PDF content")

            loader = PDFLoader(
                file_path=str(pdf_path), chunk_size=100, chunk_overlap=10
            )
            docs = loader.load()

            assert len(docs) > 1

    def test_load_pdf_file_not_found(self):
        """load should raise error for non-existent file."""
        loader = PDFLoader(file_path="/nonexistent/path.pdf")

        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_load_and_split(self, tmp_path):
        """load_and_split should use custom text splitter."""
        with patch("src.data.pdf_loader.PdfReader") as mock_reader:
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "Some test content for splitting"
            mock_reader.return_value.pages = [mock_page]

            pdf_path = tmp_path / "test.pdf"
            pdf_path.write_bytes(b"PDF content")

            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=2)

            loader = PDFLoader(file_path=str(pdf_path))
            docs = loader.load_and_split(splitter)

            assert len(docs) > 1
