# src/data_sql.py
from __future__ import annotations
import os, sqlite3, pandas as pd
from typing import Optional, Tuple
from src.config import load_config
from src.logging_utils import get_logger

log = get_logger()

def _connect(db_path: str) -> sqlite3.Connection:
    # sqlite3 is built-in; no extra install needed
    return sqlite3.connect(db_path, check_same_thread=False)

def init_db_from_csv() -> None:
    """
    1) Creates/opens churn.db
    2) Loads Customer_Data.csv into table prod_Churn (replace)
    3) Executes SQLQueries.sql to create views
    """
    cfg = load_config()
    conn = _connect(cfg.db_path)

    # 1) load CSV into table
    log.info("Loading CSV -> SQL table '%s' ...", cfg.sql["table_name"])
    df = pd.read_csv(cfg.data_path)
    df.to_sql(cfg.sql["table_name"], conn, if_exists="replace", index=False)
    log.info("Wrote %d rows to table %s", len(df), cfg.sql["table_name"])

    # 2) run SQL file (views & extras)
    sql_path = cfg.sql.get("sql_file", "SQLQueries.sql")
    if os.path.exists(sql_path):
        with open(sql_path, "r", encoding="utf-8") as f:
            script = f.read()
        log.info("Applying SQL script: %s", sql_path)
        conn.executescript(script)
        conn.commit()
    else:
        log.warning("SQL file %s not found. Views may not exist.", sql_path)

    conn.close()
    log.info("Database initialized at %s", cfg.db_path)

def fetch_training_df() -> pd.DataFrame:
    """
    Returns rows for supervised training: Churned + Stayed
    Uses view vw_ChurnData if available, else filters table.
    """
    cfg = load_config()
    conn = _connect(cfg.db_path)
    view = cfg.sql["view_churn"]
    table = cfg.sql["table_name"]

    # Prefer the view; fallback to filtering table if view missing
    try:
        q = f"SELECT * FROM {view};"
        df = pd.read_sql(q, conn)
    except Exception:
        log.warning("View %s not found. Falling back to WHERE filter on %s", view, table)
        q = f"""
        SELECT * FROM {table}
        WHERE Customer_Status IN ('Churned','Stayed');
        """
        df = pd.read_sql(q, conn)

    conn.close()
    return df

def fetch_inference_df(use_joined_view: bool = False) -> pd.DataFrame:
    """
    Returns rows for prediction. If use_joined_view=True, pulls 'Joined' rows.
    Otherwise returns the whole table (labels dropped later).
    """
    cfg = load_config()
    conn = _connect(cfg.db_path)
    table = cfg.sql["table_name"]
    if use_joined_view:
        view = cfg.sql["view_joined"]
        try:
            df = pd.read_sql(f"SELECT * FROM {view};", conn)
        except Exception:
            log.warning("View %s not found. Falling back to WHERE filter on %s", view, table)
            df = pd.read_sql(f"SELECT * FROM {table} WHERE Customer_Status='Joined';", conn)
    else:
        df = pd.read_sql(f"SELECT * FROM {table};", conn)
    conn.close()
    return df

def write_predictions_to_sql(pred_df: pd.DataFrame, table_name: str = "predictions") -> None:
    cfg = load_config()
    conn = _connect(cfg.db_path)
    pred_df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    log.info("Saved %d predictions to SQL table '%s'", len(pred_df), table_name)
