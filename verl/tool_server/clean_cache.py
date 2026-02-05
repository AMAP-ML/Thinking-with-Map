import json
import re

def is_empty_value(value):
    """
    判断一个值是否为空：
    - None
    - 字符串为纯空白
    - 字符串只含 Markdown代码围栏 ```（可能伴随换行、空格）
    """
    if value is None:
        return True
    if isinstance(value, str):
        # 去掉首尾空白
        stripped = value.strip()
        if stripped == "":
            return True
        # 去掉所有反引号后再检查
        no_backticks = stripped.replace("`", "").strip()
        if no_backticks == "":
            return True
        # 如果匹配只有 ``` 包围的内容并且中间是空
        if re.fullmatch(r"```(\s*)```", stripped):
            return True
    return False

def remove_empty(obj, remove_empty_containers=True):
    if isinstance(obj, dict):
        cleaned_dict = {
            k: remove_empty(v, remove_empty_containers)
            for k, v in obj.items()
            if not is_empty_value(v)
        }
        if remove_empty_containers and not cleaned_dict:
            return None
        return cleaned_dict
    elif isinstance(obj, list):
        cleaned_list = [
            remove_empty(item, remove_empty_containers)
            for item in obj
            if not is_empty_value(item)
        ]
        cleaned_list = [item for item in cleaned_list if item is not None]
        if remove_empty_containers and not cleaned_list:
            return None
        return cleaned_list
    else:
        return obj

if __name__ == "__main__":
    with open("api_cache.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_data = remove_empty(data, remove_empty_containers=True)

    with open("api_cache.json", "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=4)

    print("清理完成，结果已保存到 output.json")
