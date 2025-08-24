# rag_ingest.py
from __future__ import annotations
import os, re, json
from pathlib import Path
from typing import List, Dict
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

# ---- 設定 ----
OLLAMA_BASE_URL = "http://localhost:11434/v1"
EMBED_MODEL = "nomic-embed-text"   # ローカル埋め込み
CHROMA_DIR = "data/chroma"
DOCS_DIR = "data/docs"
COLLECTION = "local_corpus"

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

def load_texts(doc_dir: str) -> List[Dict]:
    texts = []
    for p in Path(doc_dir).rglob("*"):
        if p.suffix.lower() in [".txt", ".md"]:
            texts.append({"path": str(p), "text": p.read_text(encoding="utf-8", errors="ignore")})
        elif p.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(p))
                content = "\n".join([page.extract_text() or "" for page in reader.pages])
                texts.append({"path": str(p), "text": content})
            except Exception as e:
                print(f"[WARN] PDF読取失敗 {p}: {e}")
    return texts

def chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+max_chars]
        chunks.append(chunk)
        i += max(1, max_chars - overlap)
    return chunks

def embed_batch(strings: List[str]) -> List[List[float]]:
    # OpenAI互換 Embeddings API をOllamaに向ける
    res = client.embeddings.create(model=EMBED_MODEL, input=strings)
    # res.data は順序対応のベクトル群
    return [item.embedding for item in res.data]

def main():
    # DB準備
    client_chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        coll = client_chroma.get_collection(COLLECTION)
    except:
        coll = client_chroma.create_collection(COLLECTION)

    # 文書読み込み→分割→埋め込み→保存
    docs = load_texts(DOCS_DIR)
    print(f"[INFO] 読み込んだ文書数: {len(docs)}")

    ids, metadatas, contents = [], [], []
    for d in docs:
        chunks = chunk_text(d["text"])
        for idx, ch in enumerate(chunks):
            ids.append(f"{d['path']}#{idx}")
            metadatas.append({"source": d["path"], "chunk": idx})
            contents.append(ch)

    print(f"[INFO] チャンク数: {len(contents)}")
    if not contents:
        print("[INFO] 追加するデータなし")
        return

    # バッチで埋め込み（大きすぎる場合は分割して）
    BATCH = 64
    all_embeddings = []
    for i in range(0, len(contents), BATCH):
        batch = contents[i:i+BATCH]
        embs = embed_batch(batch)
        all_embeddings.extend(embs)
        print(f"[INFO] embedded {i+len(batch)}/{len(contents)}")

    # 既存IDと重複しないように add（Chromaは重複IDでエラー）
    coll.add(ids=ids, embeddings=all_embeddings, metadatas=metadatas, documents=contents)
    print("[OK] インデックス完了")

if __name__ == "__main__":
    os.makedirs("data/docs", exist_ok=True)
    os.makedirs("data/chroma", exist_ok=True)
    main()
