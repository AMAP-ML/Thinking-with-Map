import json
import os

# ----------------- 你给的解析函数 -----------------
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

def regroup_by_image_and_split(
    jsonl_path, 
    image_list_json, 
    in_list_path, 
    out_list_path, 
    keep_first_gts=True, 
    max_outputs_per_image=None
):
    """
    根据 image_list_json 将结果分为两个 jsonl 文件。
    """
    # 1. 加载目标图片列表
    with open(image_list_json, 'r', encoding='utf-8') as f:
        target_images_list = json.load(f)
    target_images_set = set(target_images_list)
    print(f"Loaded {len(target_images_set)} target images from list.")

    groups = {}
    total_lines = 0
    kept_lines = 0
    dropped_no_image = 0
    dropped_bad_score = 0
    dropped_no_solution = 0
    dropped_reach_max_outputs = 0

    # 2. 遍历原始数据进行 Group
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

            score = obj.get("score", None)
            if score is None:
                dropped_bad_score += 1
                continue
            try:
                score = float(score)
            except Exception:
                dropped_bad_score += 1
                continue

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

            if (max_outputs_per_image is not None and
                len(groups[image]["outputs"]) >= max_outputs_per_image):
                dropped_reach_max_outputs += 1
                continue

            groups[image]["outputs"].append({"output": output, "score": score})
            kept_lines += 1

    # 3. 分流写入两个文件
    count_in = 0
    count_out = 0
    
    with open(in_list_path, "w", encoding="utf-8") as f_in, \
         open(out_list_path, "w", encoding="utf-8") as f_out:
        
        for image, rec in groups.items():
            line_str = json.dumps(rec, ensure_ascii=False) + "\n"
            if image in target_images_set:
                f_in.write(line_str)
                count_in += 1
            else:
                f_out.write(line_str)
                count_out += 1

    stats = {
        "total_lines_processed": total_lines,
        "kept_samples_total": kept_lines,
        "num_images_total": len(groups),
        "num_images_in_list": count_in,
        "num_images_not_in_list": count_out,
        "dropped_no_image": dropped_no_image,
        "dropped_bad_score": dropped_bad_score,
        "dropped_reach_max_outputs": dropped_reach_max_outputs,
    }
    return stats


# ----------------- 用法示例 -----------------

# 输入文件
dump_jsonl = "verl/verl_dump/qwen3vl30b-instruct-test-amapv2promptv1-dumpn4/0.jsonl"
# 包含图片路径列表的 JSON
target_image_list_json = "./acc10km_at_least_two.json" 

# 输出文件
grouped_in_list = "./grouped_traj/amapv22_easy_grpo_step0_group4.jsonl"
grouped_not_in_list = "./grouped_traj/amapv22_hard_grpo_step0_group4.jsonl"

stats = regroup_by_image_and_split(
    jsonl_path=dump_jsonl,
    image_list_json=target_image_list_json,
    in_list_path=grouped_in_list,
    out_list_path=grouped_not_in_list,
    max_outputs_per_image=4
)

print(json.dumps(stats, indent=4))
