import os
import json
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

import dotenv
from tqdm import tqdm
from openai import OpenAI

from structure import Structure  # 你已有的 Pydantic 模型

# 读取 .env
if os.path.exists(".env"):
    dotenv.load_dotenv()

# 读取提示词
template = open("template.txt", "r", encoding="utf-8").read()
system = open("system.txt", "r", encoding="utf-8").read()

# 初始化 OpenAI 客户端（支持自定义 BASE_URL 以兼容第三方 OpenAI-compatible 网关）
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="jsonline data file")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of parallel workers")
    return parser.parse_args()

def _format_user_text(language: str, summary: str) -> str:
    """把 language/summary 套进你的 template。若模板里含花括号导致 .format 报错，则退化为追加文本。"""
    try:
        return template.format(language=language, content=summary)
    except Exception:
        return f"{template}\n\n[LANGUAGE]={language}\n[CONTENT]:\n{summary}"

def _call_model(summary: str, language: str, model_name: str) -> Dict:
    """调用 OpenAI Responses API，使用结构化输出（严格 JSON Schema）。"""
    schema = Structure.model_json_schema()  # Pydantic v2
    user_text = _format_user_text(language, summary)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ],
    )
    # resp = client.responses.create(
    #     model=model_name,
    #     input=[
    #         {"role": "system", "content": [{"type": "text", "text": system}]},
    #         {"role": "user", 
    #         "content": [{"type": "text", "text": user_text}]},
    #     ],
        # response_format={
        #     "type": "json_schema",
        #     "json_schema": {
        #         "name": "Structure",
        #         "schema": schema,
        #         "strict": True,   # 要求严格符合 schema
        #     },
        # },
        # max_output_tokens=1200,   # 视需要调整
    
    txt = resp.choices[0].message.content.strip()
    print(f"问题是：{user_text}")
    print(f"通过模型得到的回复是：{txt}\n\n")
    # 首选 SDK 的聚合文本字段
    # txt = getattr(resp, "output_text", None)
    # if not txt:
    #     # 兜底：从分片里提取首个文本块（个别模型/版本可能出现该情况）
    #     out_items = getattr(resp, "output", []) or []
    #     for item in out_items:
    #         for c in getattr(item, "content", []) or []:
    #             t = getattr(c, "text", None) or getattr(c, "string", None)
    #             if isinstance(t, str) and t.strip():
    #                 txt = t
    #                 break
    #         if txt:
    #             break

    if not txt:
        raise RuntimeError("Empty model output")

    return json.loads(txt)

def process_single_item_openai(item: Dict, language: str, model_name: str) -> Dict:
    """处理单条数据：调用模型并把结构化结果写入 item['AI']。"""
    try:
        
        item["AI"] = _call_model(item["summary"], language, model_name)
    except Exception as e:
        print(f"Item {item.get('id')} failed: {e}", file=sys.stderr)
        item["AI"] = {
            "tldr": "Error",
            "motivation": "Error",
            "method": "Error",
            "result": "Error",
            "conclusion": "Error",
        }
    return item

def process_all_items(data: List[Dict], model_name: str, language: str, max_workers: int) -> List[Dict]:
    """并行处理所有数据项（线程池）。"""
    print("连接到:", model_name, file=sys.stderr)
    processed_data = [None] * len(data)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        fut2idx = {
            executor.submit(process_single_item_openai, item, language, model_name): idx
            for idx, item in enumerate(data)
        }
        for fut in tqdm(as_completed(fut2idx), total=len(data), desc="Processing items"):
            idx = fut2idx[fut]
            try:
                processed_data[idx] = fut.result()
            except Exception as e:
                print(f"Item at index {idx} generated an exception: {e}", file=sys.stderr)
                processed_data[idx] = data[idx]
    return processed_data

def main():
    args = parse_args()
    model_name = os.environ.get("MODEL_NAME", "gpt-4.1-mini")  # 换成你要用的模型
    language = os.environ.get("LANGUAGE", "Chinese")

    target_file = args.data.replace(".jsonl", f"_AI_enhanced_{language}.jsonl")
    if os.path.exists(target_file):
        os.remove(target_file)
        print(f"Removed existing file: {target_file}", file=sys.stderr)

    # 读取
    data = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    # 去重（按 id）
    seen = set()
    uniq = []
    for it in data:
        if it["id"] not in seen:
            seen.add(it["id"])
            uniq.append(it)
    data = uniq

    print("Open:", args.data, file=sys.stderr)

    # 并行处理
    processed = process_all_items(data, model_name, language, args.max_workers)

    # 保存（保留中文：ensure_ascii=False）
    with open(target_file, "w", encoding="utf-8") as f:
        for it in processed:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
