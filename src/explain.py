# src/explain.py
from __future__ import annotations
import joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
import shap
from src.config import load_config
from src.features import prepare_inference_frame
from src.logging_utils import get_logger

log = get_logger()

def main() -> None:
    cfg = load_config()
    log.info("Loading model...")
    model = joblib.load(cfg.model_path)

    # split pipeline
    pre = model.named_steps["pre"]
    clf = model.named_steps["clf"]

    # prepare raw frame (may contain strings) -> transform to numeric
    df = pd.read_csv(cfg.data_path)
    X_raw = prepare_inference_frame(df)
    log.info("Transforming data with preprocessor...")
    X_num = pre.transform(X_raw)            # numeric sparse/dense matrix
    if hasattr(X_num, "toarray"):
        X_num = X_num.toarray()             # make it dense for SHAP

    # get one-hot feature names for nice labels
    try:
        feat_names = pre.get_feature_names_out()
    except Exception:
        feat_names = np.array([f"feat_{i}" for i in range(X_num.shape[1])])

    # Build a SHAP explainer on the classifier only, using the numeric matrix.
    # Use a small background sample for speed.
    log.info("Computing SHAP values on numeric features...")
    bg_idx = np.random.RandomState(42).choice(len(X_num), size=min(200, len(X_num)), replace=False)
    background = X_num[bg_idx]

    explainer = shap.Explainer(clf.predict_proba, background)  # permutation-based
    shap_values = explainer(X_num)  # shape: (n_samples, n_features, n_classes)

    # Global importance = mean(|shap|) for churn class=1
    sv_class1 = np.abs(shap_values.values[:, :, 1])
    importances = sv_class1.mean(axis=0)

    # Plot top 20 features as a horizontal bar chart
    top_k = 20 if importances.size >= 20 else importances.size
    order = np.argsort(importances)[::-1][:top_k]
    plt.figure(figsize=(9, 6))
    plt.barh(range(top_k), importances[order][::-1])
    plt.yticks(range(top_k), [feat_names[i] for i in order][::-1])
    plt.xlabel("Mean |SHAP value| (Impact)")
    plt.title("Global Feature Importance (Churn class = 1)")
    plt.tight_layout()
    out_path = cfg.global_importance_path
    plt.savefig(out_path, dpi=150)
    log.info("Saved global importance to %s", out_path)

if __name__ == "__main__":
    main()