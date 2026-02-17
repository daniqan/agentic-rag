"""Interactive demo script for the Agentic RAG system."""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document

from src.agents.rag_agent import RAGAgent, RAGAgentConfig
from src.data.firecrawl_loader import FirecrawlLoader
from src.data.pdf_loader import PDFLoader


def print_header():
    """Print demo header."""
    print("\n" + "=" * 60)
    print("  Agentic RAG Demo")
    print("  Built with LangChain, Gemini, Pinecone, and more")
    print("=" * 60 + "\n")


def print_menu():
    """Print menu options."""
    print("\nOptions:")
    print("  1. Ask a question")
    print("  2. Add documents from URL (Firecrawl)")
    print("  3. Add documents from PDF")
    print("  4. Add sample documents")
    print("  5. Exit")
    print()


def add_sample_documents(agent: RAGAgent) -> None:
    """Add sample documents to the agent."""
    sample_docs = [
        Document(
            page_content="""
            Python is a high-level, interpreted programming language known for its
            simplicity and readability. It was created by Guido van Rossum and first
            released in 1991. Python supports multiple programming paradigms including
            procedural, object-oriented, and functional programming.
            """,
            metadata={"source": "sample", "topic": "Python"},
        ),
        Document(
            page_content="""
            LangChain is a framework for developing applications powered by language models.
            It provides tools for building chains and agents that can interact with various
            data sources and APIs. LangChain supports integration with many LLM providers
            including OpenAI, Anthropic, and Google.
            """,
            metadata={"source": "sample", "topic": "LangChain"},
        ),
        Document(
            page_content="""
            Retrieval-Augmented Generation (RAG) is a technique that combines retrieval
            of relevant documents with language model generation. It helps ground LLM
            responses in factual information from a knowledge base, reducing hallucinations
            and improving accuracy.
            """,
            metadata={"source": "sample", "topic": "RAG"},
        ),
        Document(
            page_content="""
            Pinecone is a vector database service optimized for similarity search.
            It allows storing high-dimensional vectors and performing fast nearest
            neighbor searches. Pinecone is commonly used for semantic search,
            recommendation systems, and RAG applications.
            """,
            metadata={"source": "sample", "topic": "Pinecone"},
        ),
    ]

    print("Adding sample documents...")
    ids = agent.add_documents(sample_docs)
    print(f"Added {len(ids)} documents successfully!")


def add_url_documents(agent: RAGAgent) -> None:
    """Add documents from a URL."""
    url = input("Enter URL to scrape: ").strip()
    if not url:
        print("No URL provided.")
        return

    try:
        print(f"Scraping {url}...")
        loader = FirecrawlLoader(url=url, chunk_size=500)
        docs = loader.load()
        print(f"Extracted {len(docs)} document chunks.")

        ids = agent.add_documents(docs)
        print(f"Added {len(ids)} documents to vector store!")
    except Exception as e:
        print(f"Error loading URL: {e}")


def add_pdf_documents(agent: RAGAgent) -> None:
    """Add documents from a PDF file."""
    pdf_path = input("Enter PDF file path: ").strip()
    if not pdf_path:
        print("No path provided.")
        return

    try:
        print(f"Loading PDF: {pdf_path}...")
        loader = PDFLoader(file_path=pdf_path, chunk_size=500)
        docs = loader.load()
        print(f"Extracted {len(docs)} document chunks.")

        ids = agent.add_documents(docs)
        print(f"Added {len(ids)} documents to vector store!")
    except FileNotFoundError:
        print(f"PDF file not found: {pdf_path}")
    except Exception as e:
        print(f"Error loading PDF: {e}")


def ask_question(agent: RAGAgent) -> None:
    """Ask a question to the RAG agent."""
    question = input("Your question: ").strip()
    if not question:
        print("No question provided.")
        return

    try:
        print("\nSearching for relevant context...")
        response = agent.query(question)
        print("\n" + "-" * 40)
        print("Response:")
        print("-" * 40)
        print(response)
        print("-" * 40)
    except ValueError as e:
        print(f"Invalid input: {e}")
    except Exception as e:
        print(f"Error: {e}")


async def ask_question_async(agent: RAGAgent) -> None:
    """Ask a question asynchronously."""
    question = input("Your question: ").strip()
    if not question:
        print("No question provided.")
        return

    try:
        print("\nSearching for relevant context...")
        response = await agent.aquery(question)
        print("\n" + "-" * 40)
        print("Response:")
        print("-" * 40)
        print(response)
        print("-" * 40)
    except ValueError as e:
        print(f"Invalid input: {e}")
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Run the interactive demo."""
    print_header()

    print("Initializing RAG Agent...")
    try:
        config = RAGAgentConfig(
            k=3,
            temperature=0.7,
            validate_input=True,
            validate_output=True,
        )
        agent = RAGAgent(config=config)
        print("RAG Agent initialized successfully!\n")
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        print("\nMake sure you have:")
        print("  1. Set all required environment variables (.env file)")
        print("  2. Ollama running locally with nomic-embed-text model")
        print("  3. Valid API keys for Pinecone and Google")
        return

    while True:
        print_menu()
        choice = input("Select option (1-5): ").strip()

        if choice == "1":
            ask_question(agent)
        elif choice == "2":
            add_url_documents(agent)
        elif choice == "3":
            add_pdf_documents(agent)
        elif choice == "4":
            add_sample_documents(agent)
        elif choice == "5":
            print("\nGoodbye!")
            break
        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    main()
