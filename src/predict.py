from __future__ import annotations
import joblib, pandas as pd
from src.config import load_config
from src.features import prepare_inference_frame
from src.logging_utils import get_logger

log = get_logger()

def predict_file(in_csv: str | None = None, out_csv: str | None = None, from_sql_joined: bool = False, save_back_sql: bool = False) -> None:
    cfg = load_config()
    model = joblib.load(cfg.model_path)

    # choose input source
    if getattr(cfg, "use_sql", False):
        from src.data_sql import fetch_inference_df, write_predictions_to_sql
        raw = fetch_inference_df(use_joined_view=from_sql_joined)
    else:
        in_csv = in_csv or cfg.data_path
        raw = pd.read_csv(in_csv)

    X = prepare_inference_frame(raw)
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= cfg.threshold).astype(int)

    out_csv = out_csv or cfg.pred_out_path
    out = raw.copy()
    out["churn_prob"] = proba.round(4)
    out["prediction"] = preds
    out.to_csv(out_csv, index=False)
    log.info("Predictions saved to %s", out_csv)

    if getattr(cfg, "use_sql", False) and save_back_sql:
        from src.data_sql import write_predictions_to_sql
        write_predictions_to_sql(out, table_name="predictions")

if __name__ == "__main__":
    predict_file()
