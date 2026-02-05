import os
import json
import pandas as pd
from tqdm import tqdm
import base64
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ============ 配置区域 ============
# vLLM 的 OpenAI server 地址：例如 http://127.0.0.1:8000/v1
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8002/v1")

# vLLM 通常不校验 key，随便给一个即可
API_KEY = os.getenv("VLLM_API_KEY", "dummy")

# 这里填 vLLM serve 时暴露的模型名（--served-model-name 或模型本身名）
# MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "/mnt/workspace/common/models/Qwen3-VL-235B-A22B-Instruct")
MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "/mnt/workspace/common/models/Qwen3-VL-30B-A3B-Instruct")

OUTPUT_FILE = "./api_results/geobenchwocn_test_qwen3vl30b.jsonl"
MAX_WORKERS = 15
# ==================================

client = OpenAI(api_key=API_KEY, base_url=VLLM_BASE_URL)
write_lock = threading.Lock()  # 线程写文件锁

def image_to_data_url(image_path):
    with open(image_path, "rb") as f:
        img_bytes = f.read()
        b64_str = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"

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

def query_vllm_task(idx, prompt, images, data_source, ground_truth):
    """任务函数：调用 vLLM(OpenAI兼容) API 并写入结果"""
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

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            # 可选参数：按需加
            # temperature=0.0,
            # max_tokens=1024,
        )
        solution_str = resp.choices[0].message.content

        if solution_str:
            result = {
                "index": idx,
                "data_source": data_source,
                "ground_truth": ground_truth,
                "solution_str": solution_str
            }
            with write_lock:
                with open(OUTPUT_FILE, "a", encoding="utf-8") as fout:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        print(f"[ERROR] idx {idx} vLLM API 调用失败: {e}")
        return False

def main(parquet_files):
    df = load_parquet_files(parquet_files)
    print(f"总记录数: {len(df)}")

    done_indices = load_done_indices(OUTPUT_FILE)
    print(f"已完成记录数: {len(done_indices)}，将跳过这些 index")

    tasks = []
    for idx, row in df.iterrows():
        if idx in done_indices:
            continue
        tasks.append((idx, row["prompt"], row["images"], row["data_source"], row["reward_model"]["ground_truth"]))

    print(f"待处理记录数: {len(tasks)}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(query_vllm_task, *task) for task in tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            pass

if __name__ == "__main__":
    parquet_files = [
        # "./localization/MAPBench/MAPBench_test_api.parquet",
        "./localization/GeoBench/GeoBench_wocn_test_api.parquet",
        # "./localization/IMAGEO/IMAGEO_3_test_api.parquet",
        # "./localization/IMAGEO/IMAGEO_2_test_api.parquet",
    ]
    main(parquet_files)