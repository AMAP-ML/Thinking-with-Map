import json
import os

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

def regroup_and_split_by_image(jsonl_path, out_geobench_path, out_imageo_path, keep_first_gts=True, max_outputs_per_image=None):
    """
    将 group 好的条目根据 image 路径分为 GeoBench 和 IMAGEO 两组，并存为两个 jsonl。
    """
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
            # 此处虽然调用了 extract_solution，但如果只是为了分组，不一定强制需要它，
            # 保持原逻辑以确保 output 是可解析的（如果需要过滤掉无法解析的可以加逻辑）
            # lat, lon, city, country = extract_solution(output)

            score = float(obj.get("score", 0.0))

            inp = obj.get("input") or gts.get("input") or obj.get("question") or ""

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

            # 限制每个 image 的 outputs 数量
            if (max_outputs_per_image is not None and
                len(groups[image]["outputs"]) >= max_outputs_per_image):
                dropped_reach_max_outputs += 1
                continue

            groups[image]["outputs"].append({"output": output, "score": score})
            kept_lines += 1

    # --- 开始分类并写入两个文件 ---
    count_geobench = 0
    count_imageo = 0
    count_others = 0

    with open(out_geobench_path, "w", encoding="utf-8") as f_geo, \
         open(out_imageo_path, "w", encoding="utf-8") as f_img:
        
        for image_path, rec in groups.items():
            # 判断逻辑：包含 GeoBench 字符
            if "GeoBench" in image_path:
                f_geo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count_geobench += 1
            # 判断逻辑：包含 IMAGEO 字符
            elif "IMAGEO" in image_path:
                f_img.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count_imageo += 1
            else:
                count_others += 1
                # 如果有不属于这两类的，可以在这里处理，目前仅计数

    stats = {
        "total_lines": total_lines,
        "kept_lines": kept_lines,
        "total_unique_images": len(groups),
        "geobench_images": count_geobench,
        "imageo_images": count_imageo,
        "others_images": count_others,
        "dropped_no_image": dropped_no_image,
        "dropped_reach_max_outputs": dropped_reach_max_outputs,
    }
    return stats


# ----------------- 用法示例 -----------------
if __name__ == "__main__":
    dump_jsonl = "verl/verl_dump/qwen3vl30ba3b-imageo2-n16turn8-pass4-grpo-2node/62.jsonl"
    
    # 定义两个输出路径
    out_geobench = "./grouped_traj/geobench_grpo_step62_group2.jsonl"
    out_imageo = "./grouped_traj/imageo_grpo_step62_group2.jsonl"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(out_geobench), exist_ok=True)

    stats = regroup_and_split_by_image(
        dump_jsonl, 
        out_geobench, 
        out_imageo, 
        max_outputs_per_image=2
    )
    
    print("处理完成！统计信息如下：")
    print(json.dumps(stats, indent=4))
