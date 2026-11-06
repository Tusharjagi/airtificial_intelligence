def create_vector_store(chunks):
    print("\nCreating ChromaDB vector store...")

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )

    vector_store = Chroma(
        collection_name="company_documents",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

    batch_size = 10

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        print(
            f"Processing batch {i // batch_size + 1} "
            f"({i + 1}-{min(i + batch_size, len(chunks))})"
        )

        vector_store.add_documents(documents=batch)

    print("\nIngestion completed!")
    print("Total vectors:", vector_store._collection.count())

    return vector_store