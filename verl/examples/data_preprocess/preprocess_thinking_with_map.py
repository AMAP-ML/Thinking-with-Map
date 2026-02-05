# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import tempfile
import json

import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from PIL import Image, UnidentifiedImageError


# from verl.utils.hdfs_io import copy, makedirs

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_SYSTEM_CONTENT = "You are a helpful and harmless geo-localization assistant."
DEFAULT_USER_CONTENT_PREFIX = (
    "<image>You are given an image, and your task is to use your exceptional skills to determine "
    "the precise coordinates of the location depicted. Carefully examine the image, taking note of any "
    "distinctive features, POIs, landmarks, vegetation, or other elements that could serve as clues. "
    "For each step, you must conduct a thought section to reason about the visual clues and plan your next move before calling any tools. "
    "When extra information is needed to search for a location or confirm precise coordinates, you can "
    "use the given tools to get the information from search engine and maps. "
    "Once you have gathered sufficient evidence, provide your best inference for the coordinates in the following JSON format: "
    '{"lat": latitude, "lon": longitude, "city": city, "country": country}. '
    "Use signed values for latitude and longitude to indicate N/S and E/W. "
    "If you cannot narrow it down, then provide your best guess."
)


def is_valid_image(image_path):
    """检查图片是否存在且可正常打开"""
    if not os.path.exists(image_path):
        return False
    try:
        # 只验证，不读取全部像素
        with Image.open(image_path) as img:
            img.verify()  # 验证图片完整性
        return True
    except (UnidentifiedImageError, OSError):
        return False


def process_single_row(row, idx, split, data_source, image_dir):
    """
    根据 jsonl 每行构建输出结构
    """
    # 构造 prompt
    prompt = [
        {"role": "system", "content": DEFAULT_SYSTEM_CONTENT},
        {"role": "user", "content": DEFAULT_USER_CONTENT_PREFIX}
    ]

    images = [{"image": f"file://{image_dir}/{row['image']}"}]

    ground_truth = {
        "lat": float(row.get("latitude")),
        "lon": float(row.get("longitude")),
        "city": row.get("city"),
        "country": row.get("country"),
        "image": f"file://{image_dir}/{row['image']}",
        "split": split,
    }

    return {
        "data_source": data_source,
        "agent_name": "tool_agent",
        "prompt": prompt,
        "images": images,
        "max_pixels": 2000000,
        "min_pixels": 40000,
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "answer": ground_truth,
            "need_tools_kwargs": True,
            "tools_kwargs": {
                "image_zoom_in_tool": {
                    "create_kwargs": {
                        "image": f"file://{image_dir}/{row['image']}",
                        "ground_truth": ground_truth,
                    },
                },
                "amap_input_tips": {
                    "create_kwargs": {"ground_truth": ground_truth}
                },
                "amap_keyword_search": {
                    "create_kwargs": {"ground_truth": ground_truth}
                },
                "amap_poi_detail": {
                    "create_kwargs": {"ground_truth": ground_truth}
                },
                "amap_static_map": {
                    "create_kwargs": {"ground_truth": ground_truth}
                },
                "amap_satellite_map": {
                    "create_kwargs": {"ground_truth": ground_truth}
                },
                "google_search": {
                    "create_kwargs": {"ground_truth": ground_truth}
                },
                "google_map_search": {
                    "create_kwargs": {"ground_truth": ground_truth}
                },
            }
        }
    }

def main(args):
    logger.info(f"读取 JSONL 文件: {args.input_jsonl}")
    data = []
    dropped_count = 0
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            # 检查图片路径是否存在
            image_path = os.path.join(args.image_dir, row["image"])
            if not is_valid_image(image_path):
                logger.warning(f"图片不存在或已损坏，跳过该行: {image_path}")
                dropped_count += 1
                continue

            processed = process_single_row(row, idx=idx, 
                                           split=args.split, 
                                           data_source=args.data_source,
                                           image_dir=args.image_dir)
            data.append(processed)

    df = pd.DataFrame(data)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df.to_parquet(args.output_path, index=False)
    logger.info(f"已保存 {len(df)} 条数据到 {args.output_path}, 跳过 {dropped_count} 条无效图片路径数据")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 JSONL 转换为指定格式并保存为 parquet")
    parser.add_argument("--input_jsonl", required=True, help="输入的 JSONL 文件路径")
    parser.add_argument("--image_dir", required=True, help="图像路径")
    parser.add_argument("--output_path", required=True, help="输出 parquet 的目录")
    parser.add_argument("--split", default="test", help="数据集 split 名称")
    parser.add_argument("--data_source", default="IMAGEO_1", help="数据来源标签")
    args = parser.parse_args()

    main(args)