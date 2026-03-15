"""
main.py - Smart Book Q&A Crew

Everything is in this file:
1. Build the vector store from docs/
2. Search the vector store
3. Run the CrewAI workflow

Usage:
    python main.py --build
    python main.py
"""

import os
import shutil
import sys
import time
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()


def build_vector_store(docs_folder=None):
    """Load all documents from docs/ folder and store them in ChromaDB."""
    if docs_folder is None:
        docs_folder = os.environ.get("DOCS_DIR", "docs")

    all_documents = []

    for filename in os.listdir(docs_folder):
        file_path = os.path.join(docs_folder, filename)

        if filename.endswith(".pdf"):
            print(f"  Loading PDF: {filename}")
            loader = PyPDFLoader(file_path)
            all_documents.extend(loader.load())
        elif filename.endswith(".txt"):
            print(f"  Loading TXT: {filename}")
            loader = TextLoader(file_path, encoding="utf-8")
            all_documents.extend(loader.load())

    if not all_documents:
        print("No documents found!")
        print("Please add PDF or TXT files to the 'docs/' folder first.")
        return None

    print(f"  Loaded {len(all_documents)} pages total")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(all_documents)
    print(f"  Split into {len(chunks)} chunks")

    if not chunks:
        print("No readable text was found in the file.")
        print("Try a text-based PDF or a .txt file.")
        return None

    gemini_embedding_model = os.environ.get("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001")
    embeddings = GoogleGenerativeAIEmbeddings(model=gemini_embedding_model)

    chroma_db_dir = os.environ.get("CHROMA_DB_DIR", "chroma_db")

    if os.path.exists(chroma_db_dir):
        try:
            old_db = Chroma(persist_directory=chroma_db_dir, embedding_function=embeddings)
            old_db.delete_collection()
        except Exception:
            pass
        try:
            shutil.rmtree(chroma_db_dir)
        except Exception as e:
            print(f"Warning: Could not fully delete {chroma_db_dir} due to Windows file lock: {e}")

    batch_size = 80
    pause_seconds = 35
    vector_store = None

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        batch_number = (start // batch_size) + 1
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        print(f"  Indexing batch {batch_number}/{total_batches} ({len(batch)} chunks)")

        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=chroma_db_dir
            )
        else:
            vector_store.add_documents(batch)

        if start + batch_size < len(chunks):
            print(f"  Waiting {pause_seconds}s to avoid rate limits...")
            time.sleep(pause_seconds)

    print(f"  Vector store ready! Indexed {len(chunks)} chunks.")
    print("  You can now run: python main.py")
    return vector_store


@tool("RAG Search Tool")
def rag_search_tool(query: str) -> str:
    """
    Search the document knowledge base for information related to the query.
    Returns the top 3 most relevant text chunks from the uploaded documents.
    """

    chroma_db_dir = os.environ.get("CHROMA_DB_DIR", "chroma_db")
    if not os.path.exists(chroma_db_dir):
        return "Error: No vector store found. Please run 'python main.py --build' first."

    gemini_embedding_model = os.environ.get("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001")
    embeddings = GoogleGenerativeAIEmbeddings(model=gemini_embedding_model)
    vector_store = Chroma(
        persist_directory=chroma_db_dir,
        embedding_function=embeddings
    )

    results = vector_store.similarity_search(query, k=3)

    if not results:
        return "No relevant information found in the documents."

    output = "Here are the top 3 relevant chunks from the document:\n"

    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        output += f"\n--- Chunk {i} (Source: {source}, Page: {page}) ---\n"
        output += doc.page_content + "\n"

    return output


retriever_agent = Agent(
    role="Document Retriever",
    goal="Search the vector store and return the most relevant chunks for the question",
    backstory=(
        "You are an expert librarian who knows exactly where to find information. "
        "Your job is to search through the document database and pull out the "
        "most relevant paragraphs that will help answer the user's question. "
        "Always use your RAG Search Tool to find information."
    ),
    tools=[rag_search_tool],
    llm=os.environ.get("GOOGLE_LLM_MODEL", "gemini/gemini-2.5-flash"),
    verbose=True
)

writer_agent = Agent(
    role="Answer Writer",
    goal="Read the retrieved chunks and write a clear, accurate answer in simple language",
    backstory=(
        "You are a friendly teacher who explains things clearly. "
        "You take information from documents and turn them into easy-to-understand "
        "answers. You ONLY use information from the provided source chunks - "
        "you never make things up or add outside knowledge."
    ),
    llm=os.environ.get("GOOGLE_LLM_MODEL", "gemini/gemini-2.5-flash"),
    verbose=True
)


def create_tasks(question: str):
    """Create the two tasks for the crew based on the user's question."""

    retrieve_task = Task(
        description=(
            f"Search the document database for information about: '{question}'\n"
            "Use the RAG Search Tool to find the top 3 most relevant chunks.\n"
            "Return the chunks exactly as found - do not modify them."
        ),
        expected_output="A list of the top 3 matching text chunks from the document.",
        agent=retriever_agent
    )

    write_task = Task(
        description=(
            f"Using ONLY the retrieved chunks from the previous task, "
            f"write a clear answer to this question: '{question}'\n\n"
            "Rules:\n"
            "- Write 3-5 sentences in simple language\n"
            "- Only use facts from the source chunks\n"
            "- Do not add information that is not in the chunks"
        ),
        expected_output="A 3-5 sentence answer in simple, clear language.",
        agent=writer_agent
    )

    return [retrieve_task, write_task]


def run_crew(question: str):
    """Run the crew to answer a question about the uploaded document."""

    tasks = create_tasks(question)

    crew = Crew(
        agents=[task.agent for task in tasks],
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )

    return crew.kickoff()


def main():
    """Main loop - keeps asking for questions until the user types 'quit'."""

    if len(sys.argv) > 1 and sys.argv[1] == "--build":
        docs_folder = os.environ.get("DOCS_DIR", "docs")
        os.makedirs(docs_folder, exist_ok=True)
        print("=" * 50)
        print("  Smart Book Q&A Crew - Document Indexer")
        print("=" * 50)
        print()
        build_vector_store()
        return

    print("=" * 50)
    print("  Smart Book Q&A Crew")
    print("  Ask any question about your uploaded documents")
    print("=" * 50)
    print()
    print("Type 'quit' to exit.")
    print()

    while True:
        question = input("Your question: ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not question:
            print("Please type a question.\n")
            continue

        print("\nThe crew is working on your question...\n")
        result = run_crew(question)

        print()
        print("=" * 50)
        print("  FINAL ANSWER:")
        print("=" * 50)
        print(result)
        print("=" * 50)
        print()


if __name__ == "__main__":
    main()
