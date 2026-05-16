from pathlib import Path
import argparse
import csv
import json

import numpy as np
from sentence_transformers import SentenceTransformer


def load_queries(queries_path: Path) -> dict[str, str]:
    queries = {}
    with queries_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qid = str(obj.get("_id"))
            text = obj.get("text", "")
            queries[qid] = text
    return queries


def load_qrel_query_ids(qrels_path: Path, positive_only: bool = True) -> list[str]:
    qids = set()

    with qrels_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = str(row["query-id"])
            score = int(row["score"])

            if positive_only and score <= 0:
                continue

            qids.add(qid)

    return sorted(qids, key=lambda x: int(x) if x.isdigit() else x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dataset-dir", default="data/raw/beir/msmarco")
    parser.add_argument("--embedding-dir", default="data/embeddings/msmarco")
    parser.add_argument("--split", default="dev", choices=["dev", "test", "train"])
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--positive-only", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dataset_dir)
    emb_dir = Path(args.embedding_dir)
    emb_dir.mkdir(parents=True, exist_ok=True)

    queries_path = raw_dir / "queries.jsonl"
    qrels_path = raw_dir / "qrels" / f"{args.split}.tsv"

    print("=" * 80)
    print("Build query embeddings from qrels")
    print("dataset       : msmarco")
    print("split         :", args.split)
    print("queries_path  :", queries_path)
    print("qrels_path    :", qrels_path)
    print("embedding_dir :", emb_dir)
    print("=" * 80)

    queries = load_queries(queries_path)
    qids = load_qrel_query_ids(qrels_path, positive_only=args.positive_only)

    missing = [qid for qid in qids if qid not in queries]
    if missing:
        raise RuntimeError(f"{len(missing)} qids from qrels are missing in queries.jsonl. Example: {missing[:10]}")

    query_ids = qids
    query_texts = [queries[qid] for qid in query_ids]

    print("total queries in queries.jsonl :", len(queries))
    print("selected qrels query ids       :", len(query_ids))
    print("positive_only                  :", args.positive_only)

    print("Loading model:", args.model_name)
    model = SentenceTransformer(args.model_name)

    print("Encoding queries...")
    query_embeddings = model.encode(
        query_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=not args.no_normalize,
    ).astype(np.float32)

    np.save(emb_dir / "query_embeddings.npy", query_embeddings)

    with (emb_dir / "query_ids.json").open("w", encoding="utf-8") as f:
        json.dump(query_ids, f, ensure_ascii=False, indent=2)

    metadata_path = emb_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    metadata["query_split"] = args.split
    metadata["query_count"] = len(query_ids)
    metadata["query_embedding_shape"] = list(query_embeddings.shape)
    metadata["query_positive_only"] = args.positive_only
    metadata["query_model_name"] = args.model_name
    metadata["query_normalize_embeddings"] = not args.no_normalize

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("[DONE]")
    print("query_embeddings:", query_embeddings.shape)
    print("query_ids       :", len(query_ids))
    print("saved to        :", emb_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()