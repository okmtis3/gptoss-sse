from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Generator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from openai import OpenAI

# ====== 設定 ======
OLLAMA_BASE_URL = "http://localhost:11434/v1"  # OllamaのOpenAI互換API
OLLAMA_API_KEY = "ollama"                      # 任意文字列でOK
MODEL_NAME = "gpt-oss:20b"                     # 例：gpt-oss:20b

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

# ====== FastAPI ======
app = FastAPI(title="gpt-oss SSE demo", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 必要に応じて制限
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Tools 定義（必要に応じて追加/削除OK） ======
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "指定の都市の現在の天気と気温を返す（ダミー実装）",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["metric", "imperial"], "default": "metric"},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "2つの数値を加算する",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
    },
]

# ====== ツール実装（ダミー） ======
def get_current_weather(location: str, unit: str = "metric") -> Dict[str, Any]:
    sample = {
        "Tokyo": {"temp_c": 28.2, "condition": "Sunny"},
        "Osaka": {"temp_c": 30.1, "condition": "Cloudy"},
        "Matsuyama": {"temp_c": 29.0, "condition": "Clear"},
    }
    data = sample.get(location, {"temp_c": 25.0, "condition": "Unknown"})
    if unit == "imperial":
        temp_f = round(data["temp_c"] * 9 / 5 + 32, 1)
        return {"location": location, "temp": temp_f, "unit": "F", "condition": data["condition"]}
    return {"location": location, "temp": data["temp_c"], "unit": "C", "condition": data["condition"]}

def add_numbers(a: float, b: float) -> Dict[str, Any]:
    return {"a": a, "b": b, "sum": a + b}

def dispatch_tool(name: str, args_json: str) -> Dict[str, Any]:
    args = json.loads(args_json or "{}")
    if name == "get_current_weather":
        return get_current_weather(**args)
    if name == "add_numbers":
        return add_numbers(**args)
    raise ValueError(f"Unknown tool: {name}")

# ====== 入出力スキーマ ======
class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    name: Optional[str] = None

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: float = 0.2
    tool_choice: str = Field(default="auto", description="auto|required|none")

# ====== SSEユーティリティ ======
def sse_data(data: str, event: Optional[str] = None) -> bytes:
    """
    SSEフレームを生成する。例:
      data: こんにちは
      \n
    """
    if event:
        payload = f"event: {event}\ndata: {data}\n\n"
    else:
        payload = f"data: {data}\n\n"
    return payload.encode("utf-8")

# ====== コア処理：ツール解決 → 本回答をSSEで流す ======
def stream_final_answer(messages: List[Dict[str, Any]], temperature: float, tool_choice: str) -> Generator[bytes, None, None]:
    """
    2段階構成：
      (1) 1回目: ツール呼び出しが必要かを判定（この段は外へは流さない）
      (2) ツールを全部実行してrole=toolで渡したあと、
          2回目を stream=True でSSEとしてクライアントへ逐次送信
    """
    # -------- 1回目（tool判定） --------
    # stream=Trueでもいいが、外に流さないためnon-streamで十分。
    first = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=None if tool_choice == "none" else TOOLS,
        tool_choice=None if tool_choice == "auto" else tool_choice,  # "required"を渡す場合あり
        temperature=temperature,
        stream=False,
    )

    choice = first.choices[0]
    tool_calls = getattr(choice.message, "tool_calls", None)

    # ツールが必要なら、全部実行して tool メッセージを積む
    if tool_calls:
        # assistant側（ツール指示を含む）を会話にも残す
        messages.append({
            "role": "assistant",
            "content": choice.message.content,
            "tool_calls": [tc.model_dump() for tc in tool_calls],
        })

        # それぞれのツールを実行して、role=toolで返す
        for idx, tc in enumerate(tool_calls):
            name = tc.function.name
            args_json = tc.function.arguments or "{}"
            try:
                result = dispatch_tool(name, args_json)
            except Exception as e:
                result = {"error": str(e)}

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id or f"tool-call-{idx+1}",
                "name": name,
                "content": json.dumps(result, ensure_ascii=False),
            })
    else:
        # ツール不要なら、firstのテキストをそのまま流す
        text = choice.message.content or ""
        if text:
            # 逐次化のために適当に分割送信（実運用ではモデルのstream推奨）
            for ch in text:
                yield sse_data(ch)
        yield sse_data("[DONE]")
        return

    # -------- 2回目（本回答をSSEで流す） --------
    with client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature,
        stream=True,
    ) as stream:
        for ev in stream:
            delta = ev.choices[0].delta
            # 逐次テキストをSSEで送信
            if delta and (chunk := (delta.content or "")):
                yield sse_data(chunk)

    # 最後に完了シグナル
    yield sse_data("[DONE]")

# ====== エンドポイント ======
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    入力:
      {
        "messages": [{"role":"user", "content":"Matsuyamaの天気を華氏で"}],
        "tool_choice": "auto" | "required" | "none",
        "temperature": 0.2
      }
    出力:
      text/event-stream (SSE)
    """
    # Pydantic -> dict 変換（OpenAI SDKに渡す形式へ）
    messages = [m.model_dump() for m in req.messages]

    # SSEのストリーミングレスポンス
    generator = stream_final_answer(messages, req.temperature, req.tool_choice)
    return StreamingResponse(generator, media_type="text/event-stream")

# ---- 便利: ルート ----
@app.get("/")
def root():
    return JSONResponse({"message": "POST /chat (SSE). curl例はREADME参照。"})
