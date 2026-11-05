import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv();

def load_documents(docs_path="docs"):
  print(f"Loading documents from {docs_path}...")

  if not os.path.exists(docs_path):
    raise FileNotFoundError(f"The directory {docs_path} does not exits. Please create it and add your company files.")

  loader = DirectoryLoader(
    path=docs_path,
    glob="*.txt",
    loader_cls=TextLoader
  )

  documents = loader.load()

  if len(documents) == 0:
    raise FileNotFoundError(f"No .txt file found in {docs_path}. please add your company documents.")

  for i, doc in enumerate(documents[:2]):
    print(f"\nDocument {i+1}:")
    print(f" Sources: {doc.metadata["source"]}")
    print(f" Content length: {len(doc.page_content)} characters")
    print(f" Content preview {doc.page_content[:100]}..." )
    print(f" metadata: {doc.metadata}")
    print("Current directory:", os.getcwd())
    print("Docs path:", os.path.abspath(docs_path))
    print("Exists:", os.path.exists(docs_path))

  return documents


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

def create_vector_store(chunks):
    print("\nGenerating embeddings and storing in ChromaDB...")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="company_documents",
        persist_directory="./chroma_db"
    )

    print("Successfully stored embeddings in ChromaDB!")

    return vector_store



def main ():
  print("main fucntion")

  # loading documents
  documents = load_documents(docs_path="docs")

  # chunking of documents
  chunks = chunk_documents(documents)

  # Generate embeddings and store vectors
  vector_store = create_vector_store(chunks)

  print("\nRAG ingestion completed successfully!")



if __name__ == "__main__":
  main()