from __future__ import annotations

import argparse
import json
from pathlib import Path

from .modeling import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/telco-customer.csv")
    parser.add_argument("--output", default="data/processed")
    args = parser.parse_args()
    result = train(args.input, args.output)
    summary = {
        "best_params": result["best_params"],
        "final_metrics": {
            key: value
            for key, value in result["final_metrics"].items()
            if key not in {"classification_report"}
        },
    }
    Path(args.output, "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
