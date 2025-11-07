from __future__ import annotations
import os, yaml
from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class Config:
    data_path: str
    pred_out_path: str
    artifacts_dir: str
    model_path: str
    threshold: float
    solver: str
    max_iter: int
    class_weight: Any
    reports_dir: str
    global_importance_path: str

    use_sql: bool = False
    db_path: str = "churn.db"


    sql: Dict[str, Any] = field(default_factory=dict)

def load_config(path: str = "config.yaml") -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)


    if "artifacts_dir" in raw:
        os.makedirs(raw["artifacts_dir"], exist_ok=True)
    if "reports_dir" in raw:
        os.makedirs(raw["reports_dir"], exist_ok=True)

    return Config(**raw)
