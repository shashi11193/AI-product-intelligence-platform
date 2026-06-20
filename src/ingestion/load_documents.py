import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_PATH = "docs/product_docs"

def load_docs():
    docs = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".txt") or file.endswith(".md"):
            loader = TextLoader(os.path.join(DATA_PATH, file), encoding="utf-8")
            docs.extend(loader.load())

    return docs


def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(docs)


def main():
    print("Loading documents...")
    docs = load_docs()

    print(f"Loaded {len(docs)} documents")

    print("Chunking documents...")
    chunks = chunk_docs(docs)

    print(f"Created {len(chunks)} chunks")

    print("Creating embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating vector database...")
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="data/chroma_db"
    )

    db.persist()

    print("DONE: Vector DB created successfully!")


if __name__ == "__main__":
    main()