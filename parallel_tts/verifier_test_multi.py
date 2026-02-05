#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import asyncio
import base64
import io
import os
import json
import mimetypes
import random
import re
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import aiohttp
from PIL import Image
from geopy.distance import geodesic


# ------------------------- Image helpers -------------------------
def file_url_to_path(url: str) -> str:
    u = urlparse(url)
    if u.scheme != "file":
        return url
    return u.path

def image_path_to_data_url_resized(path_or_fileurl: str, max_side: int = 1024) -> str:
    # supports file:///abs/x.jpg or /abs/x.jpg
    if path_or_fileurl.startswith("file://"):
        path = path_or_fileurl[7:]
    else:
        path = path_or_fileurl

    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max(w, h) / float(max_side)
    if scale > 1:
        img = img.resize((int(w / scale), int(h / scale)), Image.BILINEAR)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return "data:image/jpeg;base64," + b64


# ------------------------- Utilities -------------------------
def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


# ------------------------- Parsing model output -------------------------
def extract_solution(solution_str: str) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
    """
    Expect JSON like {"lat": 12.34, "lon": 56.78, "city":"...", "country":"..."}.
    Robust: try to find {...} then json.loads; fallback regex.
    """
    s = (solution_str or "").strip()
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        try:
            obj = json.loads(m.group(0))
            lat = obj.get("lat", None)
            lon = obj.get("lon", None)
            city = obj.get("city", None)
            country = obj.get("country", None)
            lat = float(lat) if lat is not None else None
            lon = float(lon) if lon is not None else None
            return lat, lon, city, country
        except Exception:
            pass

    # fallback regex
    def rgx(key):
        mm = re.search(rf'"{key}"\s*:\s*([-+]?\d+(\.\d+)?)', s)
        return float(mm.group(1)) if mm else None

    lat = rgx("lat")
    lon = rgx("lon")
    city_m = re.search(r'"city"\s*:\s*"([^"]*)"', s)
    country_m = re.search(r'"country"\s*:\s*"([^"]*)"', s)
    city = city_m.group(1) if city_m else None
    country = country_m.group(1) if country_m else None
    return lat, lon, city, country


def compute_distance_score(solution_str, ground_truth, **kwargs):
    lat, lon, city, country = extract_solution(solution_str)

    result = {
        "score": 0,
        "acc_500m": 0,
        "acc_2km": 0,
        "acc_10km": 0,
        "acc_25km": 0,
        "acc_200km": 0,
        "acc_750km": 0,
        "distance_m": 99999999.0,
        "pred": {"lat": lat, "lon": lon, "city": city, "country": country},
    }

    if lat is not None and lon is not None:
        try:
            distance_m = geodesic((lat, lon), (ground_truth["lat"], ground_truth["lon"])).meters
        except Exception as e:
            warnings.warn(f"Error when cal distance: {e}", RuntimeWarning)
            distance_m = 99999999.0
    else:
        distance_m = 99999999.0

    result["distance_m"] = float(distance_m)

    if distance_m < 500:
        result["score"] = 1.0
        result["acc_500m"] = 1.
        result["acc_2km"] = 1.
        result["acc_10km"] = 1.
        result["acc_25km"] = 1.
        result["acc_200km"] = 1.
        result["acc_750km"] = 1.
    elif distance_m < 2000:
        result["score"] = 0.8
        result["acc_2km"] = 1.
        result["acc_10km"] = 1.
        result["acc_25km"] = 1.
        result["acc_200km"] = 1.
        result["acc_750km"] = 1.
    elif distance_m < 10000:
        result["score"] = 0.6
        result["acc_10km"] = 1.
        result["acc_25km"] = 1.
        result["acc_200km"] = 1.
        result["acc_750km"] = 1.
    elif distance_m < 25000:
        result["score"] = 0.4
        result["acc_25km"] = 1.
        result["acc_200km"] = 1.
        result["acc_750km"] = 1.
    elif distance_m < 200000:
        result["score"] = 0.2
        result["acc_200km"] = 1.
        result["acc_750km"] = 1.
    elif distance_m < 750000:
        result["score"] = 0.1
        result["acc_750km"] = 1.
    else:
        result["score"] = 0.0

    return result


# ------------------------- Data -------------------------
@dataclass
class EnsembleSample:
    idx: int
    image: str
    user_input: str
    outputs: List[Dict[str, Any]]  # raw outputs list
    ground_truth: Dict[str, Any]


def build_ensemble_sample_from_line(obj: Dict[str, Any], idx: int) -> Optional[EnsembleSample]:
    image = obj.get("image")
    user_input = obj.get("input")
    outputs = obj.get("outputs") or []
    gt = obj.get("gts")

    if not image or not user_input or not outputs or not gt:
        return None
    if "lat" not in gt or "lon" not in gt:
        return None

    return EnsembleSample(
        idx=idx,
        image=image,
        user_input=user_input,
        outputs=outputs,
        ground_truth=gt
    )


# ------------------------- Prompt -------------------------
def make_ensemble_messages(sample: EnsembleSample) -> List[Dict[str, Any]]:
    system = (
        "You are a strict geo-localization solver. "
        "You will be given an image, the original task, and multiple candidate answers from other agents. "
        "Synthesize the best final location. "
        "If candidates disagree, pick the most evidence-consistent and geographically plausible one. "
        "After thinking, provide your final answer in the JSON format: "
        '{"lat": latitude, "lon": longitude, "city": city, "country": country}. '
        "Use signed values for latitude and longitude to indicate N/S and E/W. "
    )

    # include agent outputs as references
    refs = []
    for i, o in enumerate(sample.outputs):
        txt = (o.get("output") or "").strip()
        sc = o.get("score", None)
        if sc is None:
            refs.append(f"[Agent {i}] {txt}")
        else:
            refs.append(f"[Agent {i} | score={sc}] {txt}")

    user_text = (
        "Original prompt:\n"
        f"{sample.user_input}\n\n"
        "Other agents' candidate answers (may contain errors):\n"
        + "\n\n".join(refs) +
        "\n\nReturn ONLY JSON:\n"
        '{"lat": <float>, "lon": <float>, "city": "<string or empty>", "country": "<string or empty>"}'
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image_path_to_data_url_resized(sample.image)}}
        ]}
    ]


# ------------------------- OpenAI-compatible async client -------------------------
async def post_chat_completions(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
        
        rj = await resp.json()
        try:
            return rj["choices"][0]["message"]["content"]
        except Exception:
            raise RuntimeError(f"Bad response: status={resp.status}, body={rj}")


async def solve_one(
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    model: str,
    sample: EnsembleSample,
) -> Tuple[int, str]:
    async with sem:
        content = await post_chat_completions(
            session=session,
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=make_ensemble_messages(sample),
            temperature=0.0,
            max_tokens=256,
        )
        return sample.idx, content


async def run_async(
    samples: List[EnsembleSample],
    base_url: str,
    api_key: str,
    model: str,
    max_concurrency: int,
) -> Dict[int, str]:
    sem = asyncio.Semaphore(max_concurrency)
    connector = aiohttp.TCPConnector(limit=max_concurrency, ttl_dns_cache=300)
    results: Dict[int, str] = {}
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [asyncio.create_task(solve_one(sem, session, base_url, api_key, model, s)) for s in samples]
        for fut in asyncio.as_completed(tasks):
            idx, raw = await fut
            results[idx] = raw
    return results


# ------------------------- Calculate Best of N Original Accuracy --------

def best_of_group_metrics(sample: EnsembleSample) -> Dict[str, Any]:
    """
    Evaluate each candidate output in sample.outputs, pick the one with minimal distance_m.
    Return its metrics dict (same format as compute_distance_score).
    """
    best = None
    for o in (sample.outputs or []):
        raw = (o.get("output") or "")
        m = compute_distance_score(raw, sample.ground_truth)
        if best is None or m["distance_m"] < best["distance_m"]:
            best = m
    if best is None:
        # no outputs, return an invalid-like metrics
        best = compute_distance_score("", sample.ground_truth)
    return best



# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="./grouped_traj/imageo_grpo_step62_group2.jsonl")

    ap.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--model", default="./models/Qwen3-VL-235B-A22B-Instruct")
    ap.add_argument("--save-detail", default="./verifier_result/imageo_grpo_step62_n2_qwen30b.jsonl", help="jsonl path; empty disables")

    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-concurrency", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=0)

    args = ap.parse_args()

    samples: List[EnsembleSample] = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.max_samples and len(samples) >= args.max_samples:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            s = build_ensemble_sample_from_line(obj, i)
            if s:
                samples.append(s)

    if not samples:
        print("No valid samples. Check jsonl has image/input/outputs/ground_truth(lat,lon).")
        return

    detail_f = open(args.save_detail, "w", encoding="utf-8") if args.save_detail else None

    totals = {
        "count": 0,
        "invalid": 0,
        "score_sum": 0.0,
        "acc_500m": 0.0,
        "acc_2km": 0.0,
        "acc_10km": 0.0,
        "acc_25km": 0.0,
        "acc_200km": 0.0,
        "acc_750km": 0.0,
    }
    totals_best = {
        "count": 0,
        "invalid": 0,
        "score_sum": 0.0,
        "acc_500m": 0.0,
        "acc_2km": 0.0,
        "acc_10km": 0.0,
        "acc_25km": 0.0,
        "acc_200km": 0.0,
        "acc_750km": 0.0,
    }


    for batch in chunked(samples, args.batch_size):
        res = asyncio.run(run_async(batch, args.base_url, args.api_key, args.model, args.max_concurrency))

        for s in batch:
            raw = res.get(s.idx, "")
            metrics = compute_distance_score(raw, s.ground_truth)
            metrics_best = best_of_group_metrics(s)           

            totals["count"] += 1
            # invalid: cannot parse lat/lon
            if metrics["pred"]["lat"] is None or metrics["pred"]["lon"] is None:
                totals["invalid"] += 1
            totals["score_sum"] += metrics["score"]
            for k in ["acc_500m", "acc_2km", "acc_10km", "acc_25km", "acc_200km", "acc_750km"]:
                totals[k] += metrics[k]

            totals_best["count"] += 1
            if metrics_best["pred"]["lat"] is None or metrics_best["pred"]["lon"] is None:
                totals_best["invalid"] += 1
            totals_best["score_sum"] += metrics_best["score"]
            for k in ["acc_500m", "acc_2km", "acc_10km", "acc_25km", "acc_200km", "acc_750km"]:
                totals_best[k] += metrics_best[k]

            if detail_f:
                detail_f.write(json.dumps({
                    "idx": s.idx,
                    "image": s.image,
                    "ground_truth": s.ground_truth,

                    "pred_verifier": metrics["pred"],
                    "distance_m_verifier": metrics["distance_m"],
                    "metrics_verifier": {k: metrics[k] for k in ["score","acc_500m","acc_2km","acc_10km","acc_25km","acc_200km","acc_750km"]},
                    "raw_model_verifier": raw,

                    "pred_best": metrics_best["pred"],
                    "distance_m_best": metrics_best["distance_m"],
                    "metrics_best": {k: metrics_best[k] for k in ["score","acc_500m","acc_2km","acc_10km","acc_25km","acc_200km","acc_750km"]},
                }, ensure_ascii=False) + "\n")

        n = totals["count"]
        nb = totals_best["count"]
        print(
            f"Processed={n} "
            f"[verifier] invalid={totals['invalid']} mean_score={totals['score_sum']/n:.4f} "
            f"acc_500m={totals['acc_500m']/n:.4f} acc_2km={totals['acc_2km']/n:.4f} acc_25km={totals['acc_25km']/n:.4f} | "
            f"[best] invalid={totals_best['invalid']} mean_score={totals_best['score_sum']/nb:.4f} "
            f"acc_500m={totals_best['acc_500m']/nb:.4f} acc_2km={totals_best['acc_2km']/nb:.4f} acc_25km={totals_best['acc_25km']/nb:.4f}",
            flush=True
        )

    if detail_f:
        detail_f.close()

    n = totals["count"]
    nb = totals_best["count"]

    summary = []
    summary.append("\nFinal [verifier]:")
    summary.append(f"total={n}")
    summary.append(f"invalid={totals['invalid']}")
    summary.append(f"mean_score={totals['score_sum']/n:.6f}")
    for k in ["acc_500m","acc_2km","acc_10km","acc_25km","acc_200km","acc_750km"]:
        summary.append(f"{k}={totals[k]/n:.6f}")

    summary.append("\nFinal [best-of-distance in group]:")
    summary.append(f"total={nb}")
    summary.append(f"invalid={totals_best['invalid']}")
    summary.append(f"mean_score={totals_best['score_sum']/nb:.6f}")
    for k in ["acc_500m","acc_2km","acc_10km","acc_25km","acc_200km","acc_750km"]:
        summary.append(f"{k}={totals_best[k]/nb:.6f}")

    # 合并为最终字符串
    final_output = "\n".join(summary)
    
    # 打印到屏幕
    print(final_output)

    # 如果指定了保存路径，则输出到 .out 文件
    if args.save_detail:
        out_path = os.path.splitext(args.save_detail)[0] + ".out"
        try:
            with open(out_path, "w", encoding="utf-8") as f_out:
                f_out.write(final_output)
            print(f"\nSummary saved to: {out_path}")
        except Exception as e:
            print(f"\nFailed to save summary to {out_path}: {e}")


if __name__ == "__main__":
    main()
