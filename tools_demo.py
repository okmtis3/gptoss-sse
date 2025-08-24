# tools_demo.py
from __future__ import annotations
from typing import Any, Dict
import json
from openai import OpenAI

# ★ Ollama の OpenAI 互換エンドポイント
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "gpt-oss:20b"

# --- ツール定義（JSON Schema） ------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "指定された都市の現在の天気と気温を取得する（単位: metric/imperial）",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "都市名（例: Tokyo, Osaka）"},
                    "unit": {"type": "string", "enum": ["metric", "imperial"], "default": "metric"},
                },
                "required": ["location"]
            }
        }
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
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        }
    }
]

# --- ダミー実装（実プロジェクトでは外部API等を呼ぶ） -----
def get_current_weather(location: str, unit: str = "metric") -> Dict[str, Any]:
    print("call get_current_weather")
    # 実際には天気APIを呼ぶ。ここではダミー値を返す。
    sample = {
        "Tokyo": {"temp_c": 28.5, "condition": "Sunny"},
        "Osaka": {"temp_c": 30.1, "condition": "Cloudy"},
    }
    data = sample.get(location, {"temp_c": 25.0, "condition": "Unknown"})
    if unit == "imperial":
        temp_f = round(data["temp_c"] * 9/5 + 32, 1)
        return {"location": location, "temp": temp_f, "unit": "F", "condition": data["condition"]}
    else:
        return {"location": location, "temp": data["temp_c"], "unit": "C", "condition": data["condition"]}

def add_numbers(a: float, b: float) -> Dict[str, Any]:
    print("add_numbers")
    return {"a": a, "b": b, "sum": a + b}

# --- ツールディスパッチャ ------------------------------
def call_tool(name: str, arguments_json: str) -> Dict[str, Any]:
    args = json.loads(arguments_json or "{}")
    if name == "get_current_weather":
        return get_current_weather(**args)
    if name == "add_numbers":
        return add_numbers(**args)
    raise ValueError(f"Unknown tool: {name}")

# --- メイン：ツール呼び出しを自動処理 -------------------
def chat_with_tools(user_message: str) -> str:
    messages = [
        {"role": "system", "content": "あなたは有能なアシスタントです。必要に応じて tools を呼び出して、正確な回答を作ってください。"},
        {"role": "user", "content": user_message},
    ]

    # 1回目：モデルに回答 or ツール呼び出しを判断させる
    first = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,              # ← これが関数呼び出し機能
        tool_choice="auto",       # ← モデルに任せる
        temperature=0.2,
    )

    choice = first.choices[0]
    # ツールリクエストが無い場合は、そのまま最終回答
    if not choice.message.tool_calls:
        return choice.message.content or ""

    # ★ ここからツールを実行 → その結果を messages に追記して「2回目」の呼び出し
    for tool_call in choice.message.tool_calls:
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments
        result = call_tool(tool_name, tool_args)

        # モデルへツール結果を渡す（role=tool として返す）
        messages.append(choice.message)  # ツール呼び出しを含むassistantメッセージ
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_name,
            "content": json.dumps(result, ensure_ascii=False),
        })

    # 2回目：ツール結果を踏まえて最終回答を生成
    second = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
    )
    return second.choices[0].message.content or ""

if __name__ == "__main__":
    # 好きな質問に変えてOK：
    # 例1: ツール呼び出し（天気）
    print("---- Tool Call: Weather ----")
    print(chat_with_tools("大阪の現在の天気と気温を教えて。単位は華氏で。"))

    # 例2: ツール呼び出し（加算）
    print("\n---- Tool Call: Add Numbers ----")
    print(chat_with_tools("12.5 と 7.3 を足した結果だけ数値で教えて。"))
