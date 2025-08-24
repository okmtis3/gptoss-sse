# stream_tools.py
from __future__ import annotations
import json
from typing import Any, Dict, List
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "gpt-oss:20b"

tools = [
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
    }
]

def add_numbers(a: float, b: float) -> Dict[str, Any]:
    return {"a": a, "b": b, "sum": a + b}

def call_tool(name: str, arguments_json: str) -> Dict[str, Any]:
    args = json.loads(arguments_json or "{}")
    if name == "add_numbers":
        return add_numbers(**args)
    raise ValueError(f"Unknown tool {name}")

def first_turn_with_stream(user_message: str) -> Dict[str, Any]:
    """
    1回目をストリーミングで受け取りつつ、
    - ツール呼び出しが無ければそのままコンテンツを表示して終了
    - ツール呼び出しがあれば name / arguments を復元して返す
    """
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "必要なら必ず tools を使って正確に答えてください。"},
        {"role": "user", "content": user_message},
    ]

    tool_name = None
    tool_args_buf = ""

    with client.chat.completions.create(
        model=MODEL,
        stream=True,
        tool_choice="auto",
        tools=tools,
        temperature=0.1,
        messages=messages,
    ) as stream:
        for ev in stream:
            ch = ev.choices[0]

            # ① ふつうのテキストを受け取ったら、画面に逐次出力
            if ch.delta and (txt := (ch.delta.content or "")):
                print(txt, end="", flush=True)

            # ② ツール呼び出しが来た場合、増分で name/args が小分けに届くので結合
            if ch.delta and ch.delta.tool_calls:
                for tc in ch.delta.tool_calls:
                    if tc.function and tc.function.name:
                        tool_name = tc.function.name  # nameはだいたい一発で届く
                    if tc.function and tc.function.arguments:
                        tool_args_buf += tc.function.arguments  # JSON断片を結合

            # ③ ストリーム終了時のフラグ
            if ev.choices[0].finish_reason in ("stop", "tool_calls"):
                pass

    print()  # 改行
    return {
        "messages": messages,            # 2ターン目に引き継ぐ
        "tool_name": tool_name,
        "tool_args_json": tool_args_buf.strip() if tool_args_buf else None,
    }

def second_turn_with_tool(context: Dict[str, Any]) -> None:
    messages = context["messages"]
    tool_name = context["tool_name"]
    tool_args_json = context["tool_args_json"]

    # ツールが要求されていなければ終了
    if not tool_name:
        return

    # 1回目のassistantメッセージ（ツール呼び出しを含む）を追加
    # 実装簡略化のため、空メッセージで代用（Ollamaは大抵OK）
    messages.append({"role": "assistant", "content": None, "tool_calls": [{
        "id": "tool-call-1",
        "type": "function",
        "function": {"name": tool_name, "arguments": tool_args_json or "{}"}
    }]})

    # ツール実行 → 結果を role=tool で渡す
    result = call_tool(tool_name, tool_args_json or "{}")
    messages.append({
        "role": "tool",
        "tool_call_id": "tool-call-1",
        "name": tool_name,
        "content": json.dumps(result, ensure_ascii=False),
    })

    # 2回目：ツール結果を踏まえて最終回答（ここもストリーミング可能）
    with client.chat.completions.create(
        model=MODEL,
        stream=True,
        messages=messages,
        temperature=0.1,
    ) as stream:
        for ev in stream:
            chunk = ev.choices[0].delta.content or ""
            print(chunk, end="", flush=True)
    print()

if __name__ == "__main__":
    # 例: 明らかにツールを使いたくなる依頼
    ctx = first_turn_with_stream("12.5 と 7.3 を足して。数値だけで。")
    second_turn_with_tool(ctx)
