# examples/reward_score/my_reward.py
import json, re


def _safe_last_json(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    """
    官方建议的签名：返回单样本分数 in [0,1]
    规则（示例）：
      +0.8 正确性：若最终 JSON 中 `city` 等于 ground_truth
      +0.2 结构化：只包含一个字段 {"city": <非空字符串>}
    """
    js = _safe_last_json(solution_str or "")
    sc = 0.0
    city = (js.get("city") or "").strip() if isinstance(js, dict) else ""
    gt = (ground_truth or "").strip()
    if city and gt and city.lower() == gt.lower():
        sc += 0.8
    if isinstance(js, dict) and set(js.keys()) == {"city"} and isinstance(city, str) and city:
        sc += 0.2
    return max(0.0, min(1.0, sc))
