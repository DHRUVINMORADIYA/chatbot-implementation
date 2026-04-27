import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Path to persisted vector store
VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "vector_store")

# Load embedding model (must be same model used in ingest.py)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load persisted Chroma vector store
vector_store = Chroma(
    persist_directory=VECTOR_STORE_PATH,
    embedding_function=embedding_model,
)

print(f"Vector store loaded. Total chunks: {vector_store._collection.count()}")

test_queries = [
    "What is the GRIEVANCE MECHANISM?"
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