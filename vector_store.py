"""
vector_store.py
Chroma-based vector storage for ResearchAI Agent's "Research Memory" feature.

Har naye research report ko embed karke store karta hai. Naya topic submit hone
par, similar purani researches dhundta hai aur suggest karta hai - taaki user
decide kar sake fresh research karni hai ya purani use karni hai.

Embeddings local sentence-transformers model se banti hain (free, offline,
koi extra API cost nahi).
"""

import chromadb
from chromadb.utils import embedding_functions
import os

CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = "research_reports"

# Local, free embedding model - chhota aur fast hai (all-MiniLM-L6-v2)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
_collection = _client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)


def add_research(session_id: str, topic: str, report_summary: str = None):
    """
    Naye research ko vector store mein add karta hai.
    report_summary: agar diya, isko embed karte hain (poore report se zyada focused).
                     Nahi diya to topic hi embed hota hai.
    """
    text_to_embed = report_summary if report_summary else topic
    _collection.upsert(
        ids=[session_id],
        documents=[text_to_embed],
        metadatas=[{"topic": topic, "session_id": session_id}]
    )


def find_similar_research(topic: str, top_k: int = 3, similarity_threshold: float = 0.55) -> list[dict]:
    """
    Naye topic se similar purani researches dhundta hai.
    similarity_threshold: 0-1 ke beech, jitna zyada utna strict match chahiye.
    Returns list of dicts: [{"session_id": ..., "topic": ..., "similarity": ...}, ...]
    """
    count = _collection.count()
    if count == 0:
        return []

    results = _collection.query(
        query_texts=[topic],
        n_results=min(top_k, count)
    )

    matches = []
    if results and results["ids"] and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            # Chroma cosine distance deta hai (0 = identical, 2 = opposite)
            # Isko similarity score mein convert karte hain (1 = identical, 0 = unrelated)
            similarity = max(0.0, 1.0 - (distance / 2.0))
            if similarity >= similarity_threshold:
                matches.append({
                    "session_id": doc_id,
                    "topic": results["metadatas"][0][i]["topic"],
                    "similarity": round(similarity, 2)
                })

    return matches


def remove_research(session_id: str):
    """Ek research ko vector store se hata deta hai (jab DB se delete ho)."""
    try:
        _collection.delete(ids=[session_id])
    except Exception:
        pass  # agar exist nahi karta, silently ignore


if __name__ == "__main__":
    print("Testing vector_store.py...\n")

    add_research("rai-test-1", "Impact of Artificial Intelligence on jobs in 2025")
    add_research("rai-test-2", "Electric vehicle market growth in India")
    add_research("rai-test-3", "Climate change latest developments")

    print("Added 3 test research entries.\n")

    test_query = "How AI is affecting employment in 2026"
    print(f"Searching for similar research to: '{test_query}'\n")

    matches = find_similar_research(test_query, top_k=3, similarity_threshold=0.3)

    if matches:
        print(f"Found {len(matches)} similar research(es):")
        for m in matches:
            print(f"  [{m['session_id']}] {m['topic']} (similarity: {m['similarity']})")
    else:
        print("No similar research found.")

    # Cleanup test data
    remove_research("rai-test-1")
    remove_research("rai-test-2")
    remove_research("rai-test-3")
    print("\nTest data cleaned up.")