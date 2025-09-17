from pathlib import Path
from typing import Dict, Any, Optional, Union
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, tb_dir: Optional[str] = None, run_dir: Optional[str] = None):
        # 兼容两种参数名：tb_dir 优先；否则用 run_dir
        base = tb_dir or (Path(run_dir) / "logs" if run_dir else "experiments/logs")
        base = str(base)
        Path(base).mkdir(parents=True, exist_ok=True)
        self.tb = SummaryWriter(base)

    def log_scalar(self, tag: str, value: Union[int, float], step: int):
        self.tb.add_scalar(tag, float(value), step)

    def log_many(self, scalars: Dict[str, Any], step: Optional[int] = None):
        if step is None and "train/step" in scalars:
            try:
                step = int(scalars["train/step"])
            except Exception:
                pass
        if step is None:
            raise ValueError("step 未指定，也没有在 scalars 里找到 'train/step'")
        for k, v in scalars.items():
            if k == "train/step":
                continue
            try:
                self.tb.add_scalar(k, float(v), step)
            except Exception:
                # 忽略非数值
                pass

    # 兼容你脚本里的 logger.log({...})
    def log(self, scalars: Dict[str, Any]):
        # 允许 {"step": x, "metric": y} 这种风格
        step = scalars.get("step", scalars.get("train/step"))
        if step is None:
            raise ValueError("logger.log(...) 需要提供 'step' 或 'train/step'")
        step = int(step)
        for k, v in scalars.items():
            if k in ("step", "train/step"):
                continue
            try:
                self.tb.add_scalar(k, float(v), step)
            except Exception:
                pass

    def close(self):
        self.tb.close()