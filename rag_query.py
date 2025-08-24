# rag_query.py
from __future__ import annotations
from typing import List, Dict
import json
from openai import OpenAI
import chromadb

OLLAMA_BASE_URL = "http://localhost:11434/v1"
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "gpt-oss:20b"
CHROMA_DIR = "data/chroma"
COLLECTION = "local_corpus"

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
chroma = chromadb.PersistentClient(path=CHROMA_DIR)
coll = chroma.get_collection(COLLECTION)

SYSTEM_PROMPT = """あなたは社内向けアシスタントです。与えられたコンテキストに基づいて、簡潔で正確に回答してください。わからない場合は「わかりません」と答えてください。必ず根拠の出典（sourceとchunk番号）も最後に列挙してください。"""

def embed(texts: List[str]) -> List[List[float]]:
    res = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in res.data]

def retrieve(query: str, top_k: int = 4):
    q_emb = embed([query])[0]

    res = coll.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    for i in range(len(docs)):
        hits.append({
            "id": ids[i] if i < len(ids) else None,
            "doc": docs[i],
            "meta": metas[i] if i < len(metas) else {},
            "score": dists[i] if i < len(dists) else None,
        })
    return hits


def build_context(hits) -> str:
    blocks = []
    for h in hits:
        src = h["meta"].get("source")
        ch = h["meta"].get("chunk")
        blocks.append(f"[source={src} chunk={ch}]\n{h['doc']}")
    return "\n\n---\n\n".join(blocks)

def answer(query: str, temperature: float = 0.2) -> str:
    hits = retrieve(query)
    context = build_context(hits)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"質問: {query}\n\n# コンテキスト\n{context}"},
    ]
    res = client.chat.completions.create(
        model=GEN_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return res.choices[0].message.content

if __name__ == "__main__":
    print(answer("このリポのRAG構成の手順を要約して"))
