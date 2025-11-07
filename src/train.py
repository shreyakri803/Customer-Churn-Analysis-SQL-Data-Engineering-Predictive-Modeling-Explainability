from __future__ import annotations
import joblib, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.pipeline import Pipeline

from src.config import load_config
from src.features import prepare_training_frame
from src.logging_utils import get_logger

log = get_logger()

def main() -> None:
    cfg = load_config()
    if getattr(cfg, "use_sql", False):
        from src.data_sql import init_db_from_csv, fetch_training_df
        # Initialize DB & views on first run (idempotent replace)
        init_db_from_csv()
        df = fetch_training_df()
    else:
        log.info("Loading data from %s", cfg.data_path)
        df = pd.read_csv(cfg.data_path)

    X, y, pre = prepare_training_frame(df)

    clf = LogisticRegression(
        solver=cfg.solver,
        max_iter=cfg.max_iter,
        class_weight=cfg.class_weight,
        n_jobs=None,
        random_state=42
    )
    model = Pipeline([("pre", pre), ("clf", clf)])
    log.info("Fitting model...")
    model.fit(X, y)

    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= cfg.threshold).astype(int)
    auc = roc_auc_score(y, proba)
    f1 = f1_score(y, preds)
    log.info("Training AUC=%.4f | F1=%.4f | threshold=%.2f", auc, f1, cfg.threshold)
    joblib.dump(model, cfg.model_path)
    log.info("Model saved at %s", cfg.model_path)

if __name__ == "__main__":
    main()