from openai import OpenAI

# OllamaのOpenAI互換エンドポイント
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # 任意文字列でOK
)

response = client.chat.completions.create(
    model="gpt-oss:20b",
    messages=[
        {"role": "user", "content": "ローカルで動いてますか？1行で答えて"}
    ],
    temperature=0.7,
)

print(response.choices[0].message.content)
