import os
import json
from typing import List, Tuple, Dict, Any

from dotenv import load_dotenv
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from onboarding_helper import load_tenant_config, tenant_vector_store_path


TOP_K = 4
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_TENANT_ID = "Pay_Benefits_and_Leave"
load_dotenv(os.path.join(BASE_DIR, ".env"))

SYSTEM_RULES = """You are a policy assistant.

Rules:
1) Answer only from the provided context.
2) If context is insufficient, say: "I do not have enough information in the provided context."
3) Do not invent policy details.
4) Cite the exact chunk_ids/pages used.
5) Keep answer concise and factual.
"""


def load_vector_store(vector_store_path: str) -> Chroma:
    print(f"[VECTOR STORE] Loading from: {vector_store_path}")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    store = Chroma(
        persist_directory=vector_store_path,
        embedding_function=embedding_model,
    )
    print(f"[VECTOR STORE] Loaded. Total chunks: {store._collection.count()}")
    return store


def retrieve_chunks(vector_store: Chroma, query: str, k: int = TOP_K):
    # Returns: List[Tuple[Document, score]]
    print(f"[RETRIEVAL] Searching top {k} chunks for query: '{query[:80]}...'" if len(query) > 80 else f"[RETRIEVAL] Searching top {k} chunks for query: '{query}'")
    results = vector_store.similarity_search_with_score(query, k=k)
    for i, (doc, score) in enumerate(results, 1):
        chunk_id = doc.metadata.get("chunk_id", "?")
        page = doc.metadata.get("page", "?")
        preview = doc.page_content.replace("\n", " ").strip()
        print(f"[RETRIEVAL]   #{i} score={score:.4f} chunk={chunk_id} page={page}")
        print(f"[RETRIEVAL]       content: {preview[:200]}{'...' if len(preview) > 200 else ''}")
    return results


def build_context(results) -> Tuple[str, List[Dict[str, Any]]]:
    context_blocks = []
    citations = []

    for idx, (doc, score) in enumerate(results, start=1):
        md = doc.metadata or {}
        citation = {
            "rank": idx,
            "chunk_id": md.get("chunk_id", "unknown_chunk"),
            "page": md.get("page", "unknown_page"),
            "source": md.get("source", "unknown_source"),
            "score": float(score),
        }
        citations.append(citation)

        block = (
            f"[{idx}] chunk_id={citation['chunk_id']} "
            f"page={citation['page']} source={citation['source']} score={citation['score']}\n"
            f"{doc.page_content}"
        )
        context_blocks.append(block)

    return "\n\n".join(context_blocks), citations


def build_prompt(query: str, context_text: str) -> str:
    return f"""{SYSTEM_RULES}

User Query:
{query}

Context:
{context_text}

Output rules:
- Return ONLY valid JSON (no markdown, no extra text).
- JSON schema:
{{
  "answer": "string",
  "citations": [
    {{"chunk_id": "string", "page": "number or string"}}
  ],
  "confidence_mode": "HIGH | MEDIUM | LOW"
}}
"""

def call_llm(prompt: str) -> Dict[str, Any]:
    print(f"[LLM] Calling Groq llama-3.3-70b-versatile (prompt length: {len(prompt)} chars)")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Set GROQ_API_KEY in .env")

    client = Groq(api_key=api_key)

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw_text = completion.choices[0].message.content.strip()
    print(f"[LLM] Response received ({len(raw_text)} chars)")
    return json.loads(raw_text)


def answer_query(
    query: str,
    tenant_id: str = DEFAULT_TENANT_ID,
    k: int | None = None,
) -> Dict[str, Any]:
    print(f"[ANSWER_QUERY] tenant={tenant_id} | k={k}")
    tenant_cfg = load_tenant_config(tenant_id)
    vector_store_path = tenant_vector_store_path(tenant_id)
    top_k = k if k is not None else int(tenant_cfg["retrieval"]["top_k"])

    vector_store = load_vector_store(vector_store_path)
    results = retrieve_chunks(vector_store, query, k=top_k)
    context_text, retrieved_citations = build_context(results)
    prompt = build_prompt(query, context_text)

    llm_out = call_llm(prompt)

    # Minimal safety normalization for required keys
    result = {
        "answer": llm_out.get(
            "answer",
            "I do not have enough information in the provided context.",
        ),
        "citations": llm_out.get("citations", []),
        "confidence_mode": llm_out.get("confidence_mode", "LOW"),
        "retrieved_citations": retrieved_citations,  # useful for debugging
    }
    print(f"[ANSWER_QUERY] confidence_mode={result['confidence_mode']} | citations={result['citations']}\")")
    return result


if __name__ == "__main__":
    user_query = "Who is eligible for athletic leave?"
    result = answer_query(user_query, tenant_id=DEFAULT_TENANT_ID)
    print(json.dumps(result, indent=2))