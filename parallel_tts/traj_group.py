import json

def extract_all_json_objects(text):
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

def regroup_by_image(jsonl_path, out_jsonl_path, keep_first_gts=True, max_outputs_per_image=None):
    # image -> {"image":..., "input":..., "gts":..., "outputs":[{"output":..., "score":...}, ...]}
    groups = {}

    total_lines = 0
    kept_lines = 0
    dropped_no_image = 0
    dropped_reach_max_outputs = 0 

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            gts = obj.get("gts") or {}
            image = gts.get("image")
            if not image:
                dropped_no_image += 1
                continue

            output = obj.get("output", "")

            lat, lon, city, country = extract_solution(output)

            score = obj.get("score", 0.0)
            score = float(score)

            inp = obj.get("input", None)
            if inp is None:
                inp = gts.get("input", None)
            if inp is None:
                inp = obj.get("question", None)
            if inp is None:
                inp = ""

            if image not in groups:
                groups[image] = {
                    "image": image,
                    "input": inp,
                    "gts": gts,
                    "outputs": []
                }
            else:
                if not groups[image]["input"] and inp:
                    groups[image]["input"] = inp

                if not keep_first_gts:
                    groups[image]["gts"] = gts
                if groups[image]["gts"] != gts:
                    print("WARN gts mismatch", image)

            # ✅ 核心：限制每个 image 的 outputs 数量
            if (max_outputs_per_image is not None and
                len(groups[image]["outputs"]) >= max_outputs_per_image):
                dropped_reach_max_outputs += 1
                continue

            groups[image]["outputs"].append({"output": output, "score": score})
            kept_lines += 1

    with open(out_jsonl_path, "w", encoding="utf-8") as wf:
        for image, rec in groups.items():
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats = {
        "total_lines": total_lines,
        "kept_lines": kept_lines,
        "num_images": len(groups),
        "dropped_no_image": dropped_no_image,
        "dropped_reach_max_outputs": dropped_reach_max_outputs, 
        "max_outputs_per_image": max_outputs_per_image,
    }
    return stats


dump_jsonl = "verl/verl_dump/qwen3vl30ba3b-imageo2-n16turn8-pass4-grpo-2node/0.jsonl"
grouped_jsonl = "./grouped_traj/imageo_grpo_step0_group4.jsonl"

stats = regroup_by_image(dump_jsonl, grouped_jsonl, max_outputs_per_image=4)
print(stats)
