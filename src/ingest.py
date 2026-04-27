import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")
VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "vector_store")

# Load all PDFs from data/
loader = PyPDFDirectoryLoader(DATA_PATH)
documents = loader.load()

print(f"Documents loaded: {len(documents)}")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
)

chunks = text_splitter.split_documents(documents)
print(f"Chunks created: {len(chunks)}")

# for chunk in chunks:
#     print("--->",chunk.page_content)

# Add consistent metadata per chunk
for i, chunk in enumerate(chunks):
    source = chunk.metadata.get("source", "unknown_source")
    page = chunk.metadata.get("page", -1)

    chunk.metadata["source"] = source
    chunk.metadata["page"] = page
    chunk.metadata["chunk_id"] = f"{os.path.basename(source)}:p{page}:c{i}"

# print("Sample chunk metadata:", chunks[0].metadata if chunks else "No chunks")

# Create embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Embed chunks and persist to vector store
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=VECTOR_STORE_PATH,
)

print(f"Vector store persisted to: {VECTOR_STORE_PATH}")