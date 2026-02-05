#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import asyncio
import base64
import io
import os
import json
import re
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from PIL import Image
from geopy.distance import geodesic
from openai import OpenAI


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
        yield lst[i : i + n]


def init_totals() -> Dict[str, float]:
    return {
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


def accumulate_totals(totals: Dict[str, float], metrics: Dict[str, Any]):
    totals["count"] += 1
    pred = metrics.get("pred") or {}
    if pred.get("lat", None) is None or pred.get("lon", None) is None:
        totals["invalid"] += 1
    totals["score_sum"] += float(metrics.get("score", 0.0) or 0.0)
    for k in ["acc_500m", "acc_2km", "acc_10km", "acc_25km", "acc_200km", "acc_750km"]:
        totals[k] += float(metrics.get(k, 0.0) or 0.0)


def load_done_and_accumulate(detail_path: Optional[str], totals, totals_best) -> set:
    """
    Read existing detail jsonl:
      - return set(done_idx)
      - accumulate totals/totals_best from stored metrics
    Expected line schema is what we write: metrics_verifier, metrics_best, pred_verifier, pred_best.
    """
    done = set()
    if not detail_path or not os.path.exists(detail_path):
        return done

    with open(detail_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if "idx" in obj:
                try:
                    done.add(int(obj["idx"]))
                except Exception:
                    pass

            # Accumulate verifier
            mv = obj.get("metrics_verifier") or {}
            pv = obj.get("pred_verifier") or {}
            # normalize into compute_distance_score-like dict
            metrics_v = {"pred": pv, **mv}
            if mv:
                accumulate_totals(totals, metrics_v)

            # Accumulate best
            mb = obj.get("metrics_best") or {}
            pb = obj.get("pred_best") or {}
            metrics_b = {"pred": pb, **mb}
            if mb:
                accumulate_totals(totals_best, metrics_b)

    return done


def write_detail_line(detail_f, payload: Dict[str, Any]):
    if not detail_f:
        return
    detail_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    detail_f.flush()


# ------------------------- Parsing model output -------------------------
def extract_solution(solution_str: str) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
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
        result["acc_500m"] = 1.0
        result["acc_2km"] = 1.0
        result["acc_10km"] = 1.0
        result["acc_25km"] = 1.0
        result["acc_200km"] = 1.0
        result["acc_750km"] = 1.0
    elif distance_m < 2000:
        result["score"] = 0.8
        result["acc_2km"] = 1.0
        result["acc_10km"] = 1.0
        result["acc_25km"] = 1.0
        result["acc_200km"] = 1.0
        result["acc_750km"] = 1.0
    elif distance_m < 10000:
        result["score"] = 0.6
        result["acc_10km"] = 1.0
        result["acc_25km"] = 1.0
        result["acc_200km"] = 1.0
        result["acc_750km"] = 1.0
    elif distance_m < 25000:
        result["score"] = 0.4
        result["acc_25km"] = 1.0
        result["acc_200km"] = 1.0
        result["acc_750km"] = 1.0
    elif distance_m < 200000:
        result["score"] = 0.2
        result["acc_200km"] = 1.0
        result["acc_750km"] = 1.0
    elif distance_m < 750000:
        result["score"] = 0.1
        result["acc_750km"] = 1.0
    else:
        result["score"] = 0.0

    return result


# ------------------------- Data -------------------------
@dataclass
class EnsembleSample:
    idx: int
    image: str
    user_input: str
    outputs: List[Dict[str, Any]]
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
        ground_truth=gt,
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
        + "\n\n".join(refs)
        + "\n\nReturn ONLY JSON:\n"
        + '{"lat": <float>, "lon": <float>, "city": "<string or empty>", "country": "<string or empty>"}'
    )

    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_path_to_data_url_resized(sample.image)}},
            ],
        },
    ]


# ------------------------- OpenAI SDK (sync) + async wrapper -------------------------
def call_openai_sync(client: OpenAI, model: str, messages, temperature=0.0, max_tokens=256) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return resp.choices[0].message.content or ""


async def solve_one_sdk(
    sem: asyncio.Semaphore,
    client: OpenAI,
    model: str,
    sample: EnsembleSample,
) -> Tuple[int, str]:
    async with sem:
        content = await asyncio.to_thread(
            call_openai_sync,
            client,
            model,
            make_ensemble_messages(sample),
        )
        return sample.idx, content


async def run_async_sdk(
    samples: List[EnsembleSample],
    client: OpenAI,
    model: str,
    max_concurrency: int,
) -> Dict[int, str]:
    sem = asyncio.Semaphore(max_concurrency)
    results: Dict[int, str] = {}
    tasks = [asyncio.create_task(solve_one_sdk(sem, client, model, s)) for s in samples]
    for fut in asyncio.as_completed(tasks):
        idx, raw = await fut
        results[idx] = raw
    return results


# ------------------------- Calculate Best of N Original Accuracy --------
def best_of_group_metrics(sample: EnsembleSample) -> Dict[str, Any]:
    best = None
    for o in (sample.outputs or []):
        raw = (o.get("output") or "")
        m = compute_distance_score(raw, sample.ground_truth)
        if best is None or m["distance_m"] < best["distance_m"]:
            best = m
    if best is None:
        best = compute_distance_score("", sample.ground_truth)
    return best


# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="./grouped_traj/geobench_grpo_step62_group2.jsonl")

    ap.add_argument("--base-url", default="")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--model", default="gpt-5-chat-2025-08-07")
    ap.add_argument("--save-detail", default="./verifier_result/geobench_grpo_step62_n2_gpt5.jsonl", help="jsonl path; empty disables")

    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-concurrency", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=0)

    args = ap.parse_args()

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    # ---- totals include (history + new) ----
    totals = init_totals()
    totals_best = init_totals()

    # Load history detail first (for skip + accumulate)
    done_idx = load_done_and_accumulate(args.save_detail, totals, totals_best)
    if args.save_detail and os.path.exists(args.save_detail):
        print(f"Loaded history from {args.save_detail}: done={len(done_idx)}, total(verifier)={totals['count']}, total(best)={totals_best['count']}")

    # Load all samples from source jsonl
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
        print("No valid samples. Check jsonl has image/input/outputs/gts(lat,lon).")
        return

    # Filter out already done
    samples = [s for s in samples if s.idx not in done_idx]
    if not samples:
        print("All samples already processed. Nothing to do.")
    else:
        # Append mode
        detail_f = open(args.save_detail, "a", encoding="utf-8") if args.save_detail else None

        for batch in chunked(samples, args.batch_size):
            res = asyncio.run(run_async_sdk(batch, client, args.model, args.max_concurrency))

            for s in batch:
                raw = res.get(s.idx, "")
                metrics = compute_distance_score(raw, s.ground_truth)
                metrics_best = best_of_group_metrics(s)

                # accumulate into global totals (history + new)
                accumulate_totals(totals, metrics)
                accumulate_totals(totals_best, metrics_best)

                # write one line per sample immediately
                write_detail_line(
                    detail_f,
                    {
                        "idx": s.idx,
                        "image": s.image,
                        "ground_truth": s.ground_truth,
                        "pred_verifier": metrics["pred"],
                        "distance_m_verifier": metrics["distance_m"],
                        "metrics_verifier": {
                            k: metrics[k]
                            for k in ["score", "acc_500m", "acc_2km", "acc_10km", "acc_25km", "acc_200km", "acc_750km"]
                        },
                        "raw_model_verifier": raw,
                        "pred_best": metrics_best["pred"],
                        "distance_m_best": metrics_best["distance_m"],
                        "metrics_best": {
                            k: metrics_best[k]
                            for k in ["score", "acc_500m", "acc_2km", "acc_10km", "acc_25km", "acc_200km", "acc_750km"]
                        },
                    },
                )

            n = totals["count"]
            nb = totals_best["count"]
            print(
                f"Processed(total)={n} "
                f"[verifier] invalid={totals['invalid']} mean_score={totals['score_sum']/n:.4f} "
                f"acc_500m={totals['acc_500m']/n:.4f} acc_2km={totals['acc_2km']/n:.4f} acc_25km={totals['acc_25km']/n:.4f} | "
                f"[best] invalid={totals_best['invalid']} mean_score={totals_best['score_sum']/nb:.4f} "
                f"acc_500m={totals_best['acc_500m']/nb:.4f} acc_2km={totals_best['acc_2km']/nb:.4f} acc_25km={totals_best['acc_25km']/nb:.4f}",
                flush=True,
            )

        if detail_f:
            detail_f.close()

    # Final summary (history + new)
    n = totals["count"]
    nb = totals_best["count"]

    summary = []
    summary.append("\nFinal [verifier] (history + new):")
    summary.append(f"total={n}")
    summary.append(f"invalid={totals['invalid']}")
    summary.append(f"mean_score={totals['score_sum']/n:.6f}" if n else "mean_score=nan")
    for k in ["acc_500m", "acc_2km", "acc_10km", "acc_25km", "acc_200km", "acc_750km"]:
        summary.append(f"{k}={totals[k]/n:.6f}" if n else f"{k}=nan")

    summary.append("\nFinal [best-of-distance in group] (history + new):")
    summary.append(f"total={nb}")
    summary.append(f"invalid={totals_best['invalid']}")
    summary.append(f"mean_score={totals_best['score_sum']/nb:.6f}" if nb else "mean_score=nan")
    for k in ["acc_500m", "acc_2km", "acc_10km", "acc_25km", "acc_200km", "acc_750km"]:
        summary.append(f"{k}={totals_best[k]/nb:.6f}" if nb else f"{k}=nan")

    final_output = "\n".join(summary)
    print(final_output)

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
