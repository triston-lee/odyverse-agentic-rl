import yaml


def _set_by_path(cfg: dict, key_path: str, value):
    cur = cfg
    parts = key_path.split(".")
    for k in parts[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[parts[-1]] = value


def _parse_value(v: str):
    # 尽量把字符串转成 bool/int/float
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    try:
        if "." in v:
            return float(v)
        return int(v)
    except Exception:
        return v


def load_config(path: str, overrides=None):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    overrides = overrides or []
    for item in overrides:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        _set_by_path(cfg, k, _parse_value(v))
    return cfg
