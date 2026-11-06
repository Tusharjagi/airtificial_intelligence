from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# Configuration
MODEL_NAME = "nomic-embed-text"
COLLECTION_NAME = "company_documents_v2"
PERSIST_DIRECTORY = "./chroma_db"


def load_vector_store():

    print("Loading ChromaDB vector store...")

    embeddings = OllamaEmbeddings(
        model=MODEL_NAME,
        base_url="http://localhost:11434"
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

    print("Vector store loaded successfully!")
    print("Total vectors:", vector_store._collection.count())

    return vector_store


def retrieve_documents(vector_store, query, k=3):

    print(f"\nSearching for: {query}")

    results = vector_store.similarity_search_with_score(
        query=query,
        k=k
    )

    print(f"\nRetrieved {len(results)} documents")

    for index, (document, score) in enumerate(results, start=1):

        print(f"\n{'=' * 60}")
        print(f"Result {index}")
        print(f"Distance: {score}")
        print(f"Source: {document.metadata.get('source', 'Unknown')}")
        print(f"Content:\n{document.page_content}")

    return results


def main():

    vector_store = load_vector_store()

    while True:

        query = input("\nAsk a question (type 'exit' to quit): ")

        if query.lower().strip() == "exit":
            print("Exiting retrieval pipeline...")
            break

        if not query.strip():
            continue

        retrieve_documents(
            vector_store=vector_store,
            query=query,
            k=3
        )


if __name__ == "__main__":
    main()