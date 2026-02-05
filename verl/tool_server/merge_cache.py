import json
import argparse
from typing import Dict, Any, List


def detect_format_and_flatten(data: Any) -> Dict[str, Any]:
    """
    支持两种输入：
      1) grouped: {api_name: {query: value}}
      2) flat:    {"api:query": value}
    返回统一的 flat dict: {"api:query": value}
    """
    if not isinstance(data, dict):
        raise ValueError("JSON 顶层必须是 object/dict")

    if not data:
        return {}

    # 抽样判断：如果任意一个 value 是 dict，则认为是 grouped
    any_v = next(iter(data.values()))
    is_grouped = isinstance(any_v, dict)

    if not is_grouped:
        # flat
        return data

    # grouped -> flat
    out: Dict[str, Any] = {}
    for api_name, qdict in data.items():
        if not isinstance(qdict, dict):
            # 容错：跳过异常项
            continue
        for q, v in qdict.items():
            out[f"{api_name}:{q}"] = v
    return out


def merge_flat_dicts(dict_list: List[Dict[str, Any]], keep: str = "first") -> Dict[str, Any]:
    """
    合并多个 flat dict: {"api:query": value}
    keep:
      - "first": key 冲突保留先出现的
      - "last":  key 冲突保留后出现的
    """
    result: Dict[str, Any] = {}
    for d in dict_list:
        for k, v in d.items():
            if keep == "first":
                if k not in result:
                    result[k] = v
            elif keep == "last":
                result[k] = v
            else:
                raise ValueError("keep must be 'first' or 'last'")
    return result


def merge_from_files(input_paths: List[str], output_path: str, keep: str = "first"):
    flat_list = []
    for p in input_paths:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        flat_list.append(detect_format_and_flatten(data))

    merged = merge_flat_dicts(flat_list, keep=keep)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Merge multiple cache JSON files (grouped/flat) into one FLAT output.")
    parser.add_argument("inputs", nargs="+", help="Input JSON file paths (one or more).")
    parser.add_argument("-o", "--output", required=True, help="Path to the output JSON file")
    parser.add_argument("--keep", choices=["first", "last"], default="first", help="Conflict policy for same key")
    args = parser.parse_args()

    merge_from_files(args.inputs, args.output, keep=args.keep)


if __name__ == "__main__":
    main()
