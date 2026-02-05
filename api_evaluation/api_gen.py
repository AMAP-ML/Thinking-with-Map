import os
import json, copy
import pandas as pd
from tqdm import tqdm
import base64
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from PIL import Image


# ============ 配置区域 ============
API_KEY = os.getenv("OPENAI_API_KEY", "xxxx")
MODEL_NAME = "claude-opus-4-5"
OUTPUT_FILE = "./api_results/geobenchwocn_cluadeopus45.jsonl"
MAX_WORKERS = 5  # 并发线程数，可调
# ==================================

client = OpenAI(api_key=API_KEY, base_url='xxxx')
write_lock = threading.Lock()  # 线程写文件锁


def image_to_data_url(image_path):
    with Image.open(image_path) as im:
        fmt = (im.format or "PNG").upper()   # PNG / JPEG / WEBP...
    mime = {
        "PNG": "image/png",
        "JPEG": "image/jpeg",
        "JPG": "image/jpeg",
        "WEBP": "image/webp",
    }.get(fmt, "application/octet-stream")

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def load_parquet_files(parquet_files):
    """读取给定的 parquet 文件列表并合并"""
    all_dfs = []
    for file_path in parquet_files:
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            continue
        print(f"Loading {file_path}")
        try:
            df = pd.read_parquet(file_path)
            all_dfs.append(df)
        except Exception as e:
            print(f"读取失败 {file_path}: {e}")
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        raise FileNotFoundError("未能读取任何 parquet 文件")

def load_done_indices(output_file):
    """读取已完成的 index 列表"""
    done_indices = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    idx = record.get("index")
                    if idx is not None:
                        done_indices.add(idx)
                except json.JSONDecodeError:
                    continue
    return done_indices

def query_openai_task(idx, prompt, images, data_source, ground_truth):
    """任务函数：调用 API 并写入结果"""
    try:
        messages = []
        for p in prompt:
            if p["role"] == "user":
                content_parts = [{"type": "text", "text": p["content"]}]
                for img_path in images:
                    data_url = image_to_data_url(img_path["image"].replace("file://", ""))
                    content_parts.append({"type": "image_url", "image_url": {"url": data_url}})
                messages.append({"role": "user", "content": content_parts})
            else:
                messages.append(p)

        resp = client.chat.completions.create(model=MODEL_NAME, messages=messages)
        solution_str = resp.choices[0].message.content

        if solution_str:
            result = {
                "index": idx,
                "data_source": data_source,
                "ground_truth": ground_truth,
                "solution_str": solution_str
            }
            with write_lock:  # 保证写入文件不会冲突
                with open(OUTPUT_FILE, "a", encoding="utf-8") as fout:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        print(f"[ERROR] idx {idx} API 调用失败: {e}")
        return False

def main(parquet_files):
    df = load_parquet_files(parquet_files)
    print(f"总记录数: {len(df)}")

    done_indices = load_done_indices(OUTPUT_FILE)
    print(f"已完成记录数: {len(done_indices)}，将跳过这些 index")

    # 过滤掉已完成的任务
    tasks = []
    for idx, row in df.iterrows():
        if idx in done_indices:
            continue
        tasks.append((idx, row["prompt"], row["images"], row["data_source"], row["reward_model"]["ground_truth"]))

    print(f"待处理记录数: {len(tasks)}")

    # 用线程池并发执行
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(query_openai_task, *task) for task in tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            pass  # 这里进度条只显示完成进度

if __name__ == "__main__":
    parquet_files = [
        # "./localization/MAPBench_v2.parquet",
        "./localization/GeoBench/GeoBench_wocn_test_api.parquet",
        # "./localization/IMAGEO/IMAGEO_2_test_api.parquet",
    ]
    main(parquet_files)
