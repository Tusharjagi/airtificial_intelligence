
import os

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


# Stage 1: Load documents
def load_documents(docs_path="docs"):
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"The directory {docs_path} does not exist."
        )

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )

    documents = loader.load()

    if not documents:
        raise FileNotFoundError(
            f"No .txt files found in {docs_path}."
        )

    print(f"Total documents loaded: {len(documents)}")

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i + 1}:")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content length: {len(doc.page_content)} characters")
        print(f"Preview: {doc.page_content[:100]}...")

    return documents


# Stage 2: Chunk documents
def chunk_documents(documents):
    print("\nChunking documents...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Total documents: {len(documents)}")
    print(f"Total chunks generated: {len(chunks)}")

    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i + 1}:")
        print(f"Content: {chunk.page_content[:150]}...")
        print(f"Content length: {len(chunk.page_content)}")
        print(f"Metadata: {chunk.metadata}")

    return chunks


# Stage 3 and 4: Generate embeddings and store vectors
def create_vector_store(chunks):
    print("\nCreating ChromaDB vector store...", flush=True)

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )

    vector_store = Chroma(
        collection_name="company_documents_v2",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

    batch_size = 10

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        print(
            f"Processing batch {i // batch_size + 1} "
            f"| Chunks: {i + 1}-{min(i + batch_size, len(chunks))}",
            flush=True
        )

        vector_store.add_documents(documents=batch)

    print("\nIngestion completed successfully!")
    print("Total vectors:", vector_store._collection.count())

    return vector_store


# Main pipeline
def main():
  print("🚀 Starting RAG ingestion...", flush=True)

  documents = load_documents("docs")
  chunks = chunk_documents(documents)

  vector_store = create_vector_store(chunks)

  print("\nRAG ingestion completed successfully!")


if __name__ == "__main__":
    main()
