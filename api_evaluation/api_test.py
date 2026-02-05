import os
import json
import warnings
import re
from collections import defaultdict
from geopy.distance import geodesic


def extract_all_json_objects(text):
    """
    从一个复杂字符串中提取所有可能的 JSON 对象（匹配大括号包围的内容）
    如果括号嵌套复杂，可进一步用手动栈解析。
    """
    objects = []
    stack = []
    start_idx = None

    for i, ch in enumerate(text):
        if ch == '{':
            if not stack:
                start_idx = i
            stack.append(ch)
        elif ch == '}':
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    candidate = text[start_idx:i+1]
                    objects.append(candidate)
                    start_idx = None
    return objects

def find_coords_json(text):
    """
    提取包含 lat/lon/city/country 的 JSON 对象
    """
    json_candidates = extract_all_json_objects(text)

    for cand in json_candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue

        if all(k in obj for k in ("lat", "lon", "city", "country")):
            return obj
    return None

def extract_solution(solution_str):
    result = find_coords_json(solution_str)
    if result:
        return result["lat"], result["lon"], result["city"], result["country"]
    else:
        return None, None, None, None


def compute_distance_score(solution_str, ground_truth):
    lat, lon, city, country = extract_solution(solution_str)

    result = {
        "score": 0,
        "acc_500m": 0,
        "acc_2km": 0,
        "acc_10km": 0,
        "acc_25km": 0,
        "acc_200km": 0,
        "acc_750km": 0,
    }

    if lat is not None and lon is not None:
        try:
            distance_m = geodesic(
                (lat, lon), (ground_truth["lat"], ground_truth["lon"])
            ).meters
        except Exception as e:
            warnings.warn(f"Error when cal distance: {e}", RuntimeWarning)
            distance_m = 99999999
    else:
        distance_m = 99999999

    if distance_m < 500:
        distance_score = 1.0
        result.update({k: 1. for k in result})
    elif distance_m < 2000:
        distance_score = 0.8
        result.update({
            "acc_2km": 1., "acc_10km": 1., "acc_25km": 1.,
            "acc_200km": 1., "acc_750km": 1.
        })
    elif distance_m < 10000:
        distance_score = 0.6
        result.update({
            "acc_10km": 1., "acc_25km": 1.,
            "acc_200km": 1., "acc_750km": 1.
        })
    elif distance_m < 25000:
        distance_score = 0.4
        result.update({
            "acc_25km": 1., "acc_200km": 1., "acc_750km": 1.
        })
    elif distance_m < 200000:
        distance_score = 0.2
        result.update({
            "acc_200km": 1., "acc_750km": 1.
        })
    elif distance_m < 750000:
        distance_score = 0.1
        result.update({"acc_750km": 1.})
    else:
        distance_score = 0.

    result["score"] = distance_score
    return result


def statistics_by_data_source(jsonl_file):
    stats = defaultdict(lambda: {
        "score_sum": 0,
        "acc_500m_sum": 0,
        "acc_2km_sum": 0,
        "acc_10km_sum": 0,
        "acc_25km_sum": 0,
        "acc_200km_sum": 0,
        "acc_750km_sum": 0,
        "count": 0
    })

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            ds = record.get("data_source", "unknown")
            solution_str = record.get("solution_str", "")
            ground_truth = record.get("ground_truth", {})

            score_res = compute_distance_score(solution_str, ground_truth)

            stats[ds]["score_sum"] += score_res["score"]
            stats[ds]["acc_500m_sum"] += score_res["acc_500m"]
            stats[ds]["acc_2km_sum"] += score_res["acc_2km"]
            stats[ds]["acc_10km_sum"] += score_res["acc_10km"]
            stats[ds]["acc_25km_sum"] += score_res["acc_25km"]
            stats[ds]["acc_200km_sum"] += score_res["acc_200km"]
            stats[ds]["acc_750km_sum"] += score_res["acc_750km"]
            stats[ds]["count"] += 1

    # 输出统计
    for ds, vals in stats.items():
        cnt = vals["count"]
        print(f"\nData Source: {ds}")
        if cnt == 0:
            print("  无数据")
            continue
        print(f"  Count: {cnt}")
        print(f"  Avg Score: {vals['score_sum'] / cnt:.4f}")
        print(f"  ACC@500m : {vals['acc_500m_sum'] / cnt:.4f}")
        print(f"  ACC@2km  : {vals['acc_2km_sum'] / cnt:.4f}")
        print(f"  ACC@10km : {vals['acc_10km_sum'] / cnt:.4f}")
        print(f"  ACC@25km : {vals['acc_25km_sum'] / cnt:.4f}")
        print(f"  ACC@200km: {vals['acc_200km_sum'] / cnt:.4f}")
        print(f"  ACC@750km: {vals['acc_750km_sum'] / cnt:.4f}")


if __name__ == "__main__":
    # 示例调用
    # result_path = './api_results/amap_v2_gemini3pp.jsonl'
    # result_path = './api_results/amap_v2_o3.jsonl'
    # result_path = './api_results/imageo2_test_qwen3vl235b.jsonl'
    result_path = './api_results/geobenchwocn_claudesonnet45.jsonl'
    # result_path = './api_results/geobenchwocn_gemini3p.jsonl'
    statistics_by_data_source(result_path)
