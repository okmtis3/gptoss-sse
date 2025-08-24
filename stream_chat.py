# stream_chat.py
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "gpt-oss:20b"

def main():
    # stream=True を指定すると SSE でトークンが順次届く
    with client.chat.completions.create(
        model=MODEL,
        stream=True,
        temperature=0.7,
        messages=[
            {"role": "user", "content": "ストリーミングで自己紹介して。短く！"}
        ],
    ) as stream:
        for event in stream:
            # event.choices[0].delta.content に増分が入る
            chunk = event.choices[0].delta.content or ""
            print(chunk, end="", flush=True)
    print()  # 改行

if __name__ == "__main__":
    main()
