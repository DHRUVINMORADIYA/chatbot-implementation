import os
from typing import Dict, Any

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from onboarding_helper import tenant_data_path, tenant_vector_store_path

DEFAULT_TENANT_ID = "Pay_Benefits_and_Leave"


def run_ingestion(tenant_id: str) -> Dict[str, Any]:
    # Resolve tenant-specific paths from config.
    data_path = tenant_data_path(tenant_id)
    vector_store_path = tenant_vector_store_path(tenant_id)

    # Load all PDFs from the tenant's data folder.
    loader = PyPDFDirectoryLoader(data_path)
    documents = loader.load()
    print(f"[{tenant_id}] Documents loaded: {len(documents)}")

    # Split into overlapping chunks for better retrieval coverage.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = text_splitter.split_documents(documents)
    print(f"[{tenant_id}] Chunks created: {len(chunks)}")

    # Attach consistent metadata for citation tracking.
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown_source")
        page = chunk.metadata.get("page", -1)
        chunk.metadata["source"] = source
        chunk.metadata["page"] = page
        chunk.metadata["chunk_id"] = f"{os.path.basename(source)}:p{page}:c{i}"

    # Embed chunks and persist to tenant-specific vector store.
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=vector_store_path,
    )
    print(f"[{tenant_id}] Vector store persisted to: {vector_store_path}")

    return {
        "tenant_id": tenant_id,
        "documents_loaded": len(documents),
        "chunks_created": len(chunks),
        "vector_store_path": vector_store_path,
    }


if __name__ == "__main__":
    result = run_ingestion(DEFAULT_TENANT_ID)
    print(result)