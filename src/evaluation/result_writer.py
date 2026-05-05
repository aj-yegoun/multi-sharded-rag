from pathlib import Path

import pandas as pd


def save_results_csv(results: list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"[DONE] Results saved to: {output_path}")