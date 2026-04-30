import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from onboarding_helper import tenant_vector_store_path

DEFAULT_TENANT_ID = "Pay_Benefits_and_Leave"

# Path to persisted vector store
VECTOR_STORE_PATH = tenant_vector_store_path(DEFAULT_TENANT_ID)

# Load embedding model (must be same model used in ingest.py)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load persisted Chroma vector store
vector_store = Chroma(
    persist_directory=VECTOR_STORE_PATH,
    embedding_function=embedding_model,
)

print(f"Vector store loaded. Total chunks: {vector_store._collection.count()}")

test_queries = [
    "Who is eligible for athletic leave?"
]

for query in test_queries:
    print("\n" + "=" * 80)
    print(f"Query: {query}")

    results = vector_store.similarity_search_with_score(query, k=4)

    for rank, (doc, score) in enumerate(results, start=1):
        print(f"\nResult #{rank}")
        print(f"Score: {score}")
        print(f"Metadata: {doc.metadata}")
        print(doc.page_content[:500])