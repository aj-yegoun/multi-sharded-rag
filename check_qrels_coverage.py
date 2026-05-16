from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_doc_id_set(embedding_dir: Path) -> set[str]:
    doc_id_dir = embedding_dir / "doc_id_chunks"

    if not doc_id_dir.exists():
        raise FileNotFoundError(f"doc_id_chunks directory not found: {doc_id_dir}")

    doc_id_set: set[str] = set()
    duplicate_count = 0

    for p in sorted(doc_id_dir.glob("*.json")):
        ids = load_json(p)

        for doc_id in ids:
            doc_id = str(doc_id)
            if doc_id in doc_id_set:
                duplicate_count += 1
            else:
                doc_id_set.add(doc_id)

    if duplicate_count > 0:
        print("[WARN] duplicate doc ids found:", duplicate_count)

    return doc_id_set


def load_query_id_set(embedding_dir: Path) -> set[str]:
    query_ids_path = embedding_dir / "query_ids.json"

    if not query_ids_path.exists():
        raise FileNotFoundError(f"query_ids.json not found: {query_ids_path}")

    return {str(x) for x in load_json(query_ids_path)}


def parse_qrels(qrels_path: Path):
    qrel_qids: set[str] = set()
    positive_qrel_qids: set[str] = set()
    positive_doc_ids: set[str] = set()

    total_rows = 0
    total_positive_rows = 0
    total_nonpositive_rows = 0

    with qrels_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        required_columns = {"query-id", "corpus-id", "score"}
        if reader.fieldnames is None:
            raise RuntimeError(f"qrels file has no header: {qrels_path}")

        missing_columns = required_columns - set(reader.fieldnames)
        if missing_columns:
            raise RuntimeError(
                f"qrels file missing required columns {missing_columns}: {qrels_path}"
            )

        for row in reader:
            total_rows += 1

            qid = str(row["query-id"])
            did = str(row["corpus-id"])
            score = int(row["score"])

            qrel_qids.add(qid)

            if score > 0:
                total_positive_rows += 1
                positive_qrel_qids.add(qid)
                positive_doc_ids.add(did)
            else:
                total_nonpositive_rows += 1

    return {
        "total_rows": total_rows,
        "total_positive_rows": total_positive_rows,
        "total_nonpositive_rows": total_nonpositive_rows,
        "qrel_qids": qrel_qids,
        "positive_qrel_qids": positive_qrel_qids,
        "positive_doc_ids": positive_doc_ids,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check qrels coverage for chunked embeddings.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g., msmarco, fever, trec-covid")
    parser.add_argument("--split", default="test", help="qrels split, e.g., dev, test, train")
    parser.add_argument("--raw-data-dir", default="data/raw/beir")
    parser.add_argument("--embedding-root", default="data/embeddings")
    parser.add_argument(
        "--show-examples",
        type=int,
        default=20,
        help="Number of missing id examples to print.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    embedding_dir = Path(args.embedding_root) / args.dataset
    qrels_path = Path(args.raw_data_dir) / args.dataset / "qrels" / f"{args.split}.tsv"

    print("=" * 80)
    print("Qrels coverage check")
    print("dataset       :", args.dataset)
    print("split         :", args.split)
    print("embedding dir :", embedding_dir)
    print("qrels path    :", qrels_path)
    print("=" * 80)

    if not embedding_dir.exists():
        raise FileNotFoundError(f"embedding dir not found: {embedding_dir}")

    if not qrels_path.exists():
        raise FileNotFoundError(f"qrels file not found: {qrels_path}")

    doc_id_set = load_doc_id_set(embedding_dir)
    query_ids = load_query_id_set(embedding_dir)
    qrels_info = parse_qrels(qrels_path)

    qrel_qids: set[str] = qrels_info["qrel_qids"]
    positive_qrel_qids: set[str] = qrels_info["positive_qrel_qids"]
    positive_doc_ids: set[str] = qrels_info["positive_doc_ids"]

    missing_doc_ids = positive_doc_ids - doc_id_set
    missing_positive_query_ids = positive_qrel_qids - query_ids
    extra_query_ids = query_ids - qrel_qids

    rel_d_per_q = (
        qrels_info["total_positive_rows"] / len(positive_qrel_qids)
        if len(positive_qrel_qids) > 0
        else 0.0
    )

    print("doc ids in embeddings              :", len(doc_id_set))
    print("query ids in embeddings            :", len(query_ids))
    print("qrel rows                          :", qrels_info["total_rows"])
    print("qrel query ids                     :", len(qrel_qids))
    print("positive qrel query ids            :", len(positive_qrel_qids))
    print("positive qrel rows                 :", qrels_info["total_positive_rows"])
    print("non-positive qrel rows             :", qrels_info["total_nonpositive_rows"])
    print("unique positive doc ids            :", len(positive_doc_ids))
    print("Rel D/Q based on positive rows      :", f"{rel_d_per_q:.4f}")
    print("missing positive doc ids           :", len(missing_doc_ids))
    print("missing positive query ids         :", len(missing_positive_query_ids))
    print("extra query ids not in qrels        :", len(extra_query_ids))

    if missing_doc_ids:
        print(
            "missing positive doc id examples:",
            list(sorted(missing_doc_ids))[: args.show_examples],
        )

    if missing_positive_query_ids:
        print(
            "missing positive query id examples:",
            list(sorted(missing_positive_query_ids))[: args.show_examples],
        )

    if extra_query_ids:
        print(
            "extra query id examples:",
            list(sorted(extra_query_ids))[: args.show_examples],
        )

    print("=" * 80)
    if not missing_doc_ids and not missing_positive_query_ids:
        print("[PASS] qrels coverage looks valid.")
    else:
        print("[FAIL] qrels coverage has issues.")
    print("=" * 80)


if __name__ == "__main__":
    main()