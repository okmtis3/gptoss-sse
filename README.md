# gpt-oss を構築して実行するためのサンプル（共有用）

```
ollama pull gpt-oss:20b
ollama run gpt-oss:20b
```

## 1) 依存インストール（仮想環境推奨）
```
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) サーバ起動
```
uvicorn app:app --reload --port 8000
```

```
curl -N -H "Content-Type: application/json" -X POST http://127.0.0.1:8000/chat -d "{\"messages\":[{\"role\":\"user\",\"content\":\"東京の現在の天気を華氏で。\"}],\"tool_choice\":\"auto\",\"temperature\":0.2}"

```

```
curl -H "Content-Type: application/json" -X POST http://127.0.0.1:8000/rag -d "{\"query\":\"RAGの全体構成を簡単に説明して\"}"

```

## memo

- gpt-oss:20b
- nomic-embed-text:latest

