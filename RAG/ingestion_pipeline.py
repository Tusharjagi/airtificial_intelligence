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


def main ():
  print("main fucntion")

  documents = load_documents(docs_path="docs")

if __name__ == "__main__":
  main()