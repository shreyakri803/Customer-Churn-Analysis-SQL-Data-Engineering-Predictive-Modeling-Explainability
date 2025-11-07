import argparse
from src.train import main as train_main
from src.predict import predict_file
from src.explain import main as explain_main

def main():
    parser = argparse.ArgumentParser(
        description="Customer Churn Analysis CLI"
    )

    subparsers = parser.add_subparsers(dest="command")

    # -------------------------------
    # TRAIN
    # -------------------------------
    subparsers.add_parser(
        "train",
        help="Train the churn model using CSV or SQL depending on config.yaml"
    )

    # -------------------------------
    # PREDICT
    # -------------------------------
    predict_parser = subparsers.add_parser(
        "predict",
        help="Generate predictions"
    )
    predict_parser.add_argument(
        "--joined",
        action="store_true",
        help="Use SQL joined view to predict only 'Joined' customers"
    )
    predict_parser.add_argument(
        "--sql-save",
        action="store_true",
        help="Save predictions back to SQL database table"
    )
    predict_parser.add_argument(
        "--input",
        type=str,
        help="Custom CSV file for prediction (optional)"
    )
    predict_parser.add_argument(
        "--output",
        type=str,
        help="Custom output CSV file (optional)"
    )

    # -------------------------------
    # EXPLAIN
    # -------------------------------
    subparsers.add_parser(
        "explain",
        help="Generate SHAP global importance chart"
    )

    args = parser.parse_args()

    # -----------------------------------------
    # COMMANDS
    # -----------------------------------------
    if args.command == "train":
        print("▶️ Training model...")
        train_main()
        print("✅ Training complete.")

    elif args.command == "predict":
        print("▶️ Generating predictions...")
        predict_file(
            in_csv=args.input,
            out_csv=args.output,
            from_sql_joined=args.joined,
            save_back_sql=args.sql_save
        )
        print("✅ Prediction complete.")

    elif args.command == "explain":
        print("▶️ Generating explainability plot...")
        explain_main()
        print("✅ Explainability created at reports/global_importance.png")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()