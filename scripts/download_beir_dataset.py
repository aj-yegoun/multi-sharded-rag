from pathlib import Path
import argparse
from beir import util


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--raw-data-dir", default="data/raw/beir")
    args = parser.parse_args()

    raw_dir = Path(args.raw_data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = raw_dir / args.dataset

    if dataset_dir.exists():
        print(f"[INFO] Dataset already exists: {dataset_dir}")
        return

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{args.dataset}.zip"
    print(f"[INFO] Downloading {args.dataset} from {url}")
    util.download_and_unzip(url, str(raw_dir))
    print("[DONE]")


if __name__ == "__main__":
    main()