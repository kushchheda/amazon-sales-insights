"""
main.py
-------
CLI entry point — runs the full Amazon Sales Insights pipeline.

Usage
-----
    python main.py
    python main.py --input data/amazon.csv --output outputs/
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Amazon Sales Insights: A/B Testing, EDA & LTV Pipeline"
    )
    parser.add_argument(
        "--input", default="data/amazon.csv", help="Path to raw Amazon CSV (default: data/amazon.csv)"
    )
    parser.add_argument(
        "--output", default="outputs/", help="Directory for charts and reports (default: outputs/)"
    )
    parser.add_argument(
        "--skip-eda", action="store_true", help="Skip EDA plots (faster run)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_dir = args.output

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        print("  → Place your amazon.csv in the data/ folder, or pass --input <path>")
        sys.exit(1)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- Imports (deferred to keep CLI fast) ---
    from src.data.loader import load_and_clean
    from src.analysis.eda import run_eda
    from src.ab_testing.ab_test import (
        run_engagement_ab_test,
        run_rating_ttests,
        run_chi_square_tests,
        simulate_epsilon_greedy_bandit,
    )
    from src.ltv.ltv_model import run_ltv_pipeline

    print("=" * 70)
    print("🛒  AMAZON SALES INSIGHTS PIPELINE")
    print("=" * 70)

    # 1. Load & clean
    print("\n[1/4] Loading and cleaning data...")
    df = load_and_clean(str(input_path))

    # 2. EDA
    if not args.skip_eda:
        print("\n[2/4] Running EDA...")
        run_eda(df, output_dir=output_dir)
    else:
        print("\n[2/4] EDA skipped (--skip-eda)")

    # 3. A/B Testing
    print("\n[3/4] Running A/B Testing...")
    run_engagement_ab_test(df, output_dir=output_dir)
    run_rating_ttests(df)
    run_chi_square_tests(df)
    simulate_epsilon_greedy_bandit(df)

    # 4. LTV Modeling
    print("\n[4/4] Running LTV Modeling...")
    run_ltv_pipeline(df, output_dir=output_dir)

    print("\n" + "=" * 70)
    print(f"✅  Pipeline complete. Charts saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
