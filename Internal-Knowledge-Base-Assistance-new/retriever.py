import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()


async def retrieve_top_k(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve top-k relevant documents from Pinecone index for a given query.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX")

    if not openai_api_key:
        raise ValueError("Missing OPENAI_API_KEY")
    if not pinecone_api_key:
        raise ValueError("Missing PINECONE_API_KEY")
    if not index_name:
        raise ValueError("Missing PINECONE_INDEX")

    # Initialize clients
    client = OpenAI(api_key=openai_api_key)
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)

    # Generate embedding for the query
    emb = client.embeddings.create(input=[query], model=embed_model)
    query_vec = emb.data[0].embedding

    # Query Pinecone index
    res = index.query(vector=query_vec, top_k=top_k, include_metadata=True)

    # Format matches
    matches: List[Dict[str, Any]] = []
    for m in res.matches or []:
        matches.append({
            "id": m.id,
            "score": m.score,
            "source": (m.metadata or {}).get("source"),
            "text": (m.metadata or {}).get("text"),
        })
    return matches


def _format_context(matches: List[Dict[str, Any]]) -> str:
    """
    Prepare the context text for the language model from retrieved matches.
    """
    parts: List[str] = []
    for m in matches:
        source = m.get("source") or "unknown"
        snippet = (m.get("text") or "").strip()
        parts.append(f"[Source: {source}]\n{snippet}")
    return "\n\n".join(parts)


def generate_answer(query: str, matches: List[Dict[str, Any]]) -> str:
    """
    Generate a concise answer using OpenAI LLM based on retrieved matches.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)
    context = _format_context(matches)

    system_msg = (
        "You are an expert AI assistant. Your task is to answer the user's question "
        "**strictly using the provided context**. Do not assume or hallucinate anything. "
        "If the answer is not present in the context, respond with: 'I don't know.'\n\n"
        "Instructions:\n"
        "1. Always use information only from the context.\n"
        "2. Provide concise and clear answers.\n"
        "3. If multiple sources are relevant, summarize them briefly.\n"
        "4. Mention the source(s) if available in the format: [Source: source_name].\n"
        "5. Avoid repeating the context unnecessarily.\n"
        "6. Keep the answer under 150 words.\n"
        "7. If there are multiple points, present them in bullet points."
    )

    user_msg = f"Question: {query}\n\nContext:\n{context}"

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )

    return resp.choices[0].message.content.strip() if resp.choices else ""


# --- New helper functions for count queries ---

def is_count_query(query: str) -> bool:
    """
    Check if the user query is asking for total number of entries.
    """
    keywords = ["how many", "total entries", "number of records", "count"]
    query_lower = query.lower()
    return any(k in query_lower for k in keywords)


def get_total_count() -> int:
    """
    Get the total number of vectors (entries) in the Pinecone index.
    """
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX")
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    return stats['total_vector_count']


# --- Updated answer_query function ---

async def answer_query(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Full pipeline: handle count queries or semantic retrieval and generate an answer.
    """
    # Handle total count questions separately
    if is_count_query(query):
        total = get_total_count()
        answer = f"The database contains {total} entries."
        return {"answer": answer, "matches": []}

    # Normal semantic retrieval
    matches = await retrieve_top_k(query=query, top_k=top_k)
    answer = generate_answer(query=query, matches=matches)
    return {"answer": answer, "matches": matches}
