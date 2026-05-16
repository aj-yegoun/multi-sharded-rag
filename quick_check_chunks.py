from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_doc_chunks(base: Path, expected_dim: int, full_scan: bool) -> tuple[int, int, bool]:
    doc_chunk_dir = base / "doc_chunks"
    id_chunk_dir = base / "doc_id_chunks"

    doc_chunks = sorted(doc_chunk_dir.glob("*.npy"))
    id_chunks = sorted(id_chunk_dir.glob("*.json"))

    print("=" * 80)
    print("[DOC CHUNK CHECK]")
    print("embedding dir   :", base)
    print("doc chunk dir   :", doc_chunk_dir)
    print("doc id chunk dir:", id_chunk_dir)
    print("doc chunk count :", len(doc_chunks))
    print("id chunk count  :", len(id_chunks))

    ok = True

    if not doc_chunk_dir.exists():
        print("[FAIL] doc_chunks directory not found:", doc_chunk_dir)
        return 0, 0, False

    if not id_chunk_dir.exists():
        print("[FAIL] doc_id_chunks directory not found:", id_chunk_dir)
        return 0, 0, False

    if len(doc_chunks) == 0:
        print("[FAIL] no doc chunk .npy files found")
        ok = False

    if len(id_chunks) == 0:
        print("[FAIL] no doc id chunk .json files found")
        ok = False

    if len(doc_chunks) != len(id_chunks):
        print("[FAIL] doc chunk count != id chunk count")
        ok = False

    total_vecs = 0
    total_ids = 0

    seen_ids: set[str] = set()
    duplicate_count = 0

    for i, (emb_path, id_path) in enumerate(zip(doc_chunks, id_chunks)):
        emb = np.load(emb_path, mmap_mode="r")
        ids = load_json(id_path)
        ids = [str(x) for x in ids]

        print(i, emb_path.name, emb.shape, id_path.name, len(ids))

        if emb.ndim != 2:
            print("[FAIL] embedding is not 2D:", emb_path, emb.shape)
            ok = False

        if emb.ndim == 2 and emb.shape[1] != expected_dim:
            print(
                "[FAIL] embedding dim mismatch:",
                emb_path,
                "expected =",
                expected_dim,
                "actual =",
                emb.shape,
            )
            ok = False

        if emb.shape[0] != len(ids):
            print("[FAIL] ROW/ID MISMATCH:", emb_path, id_path)
            ok = False

        if full_scan:
            # mmap array에 대해 chunk 단위로 finite 여부 확인
            arr = np.asarray(emb)
            if not np.isfinite(arr).all():
                print("[FAIL] NaN or inf found in:", emb_path)
                ok = False

        for doc_id in ids:
            if doc_id in seen_ids:
                duplicate_count += 1
            else:
                seen_ids.add(doc_id)

        total_vecs += emb.shape[0]
        total_ids += len(ids)

    print("-" * 80)
    print("total vectors:", total_vecs)
    print("total ids    :", total_ids)
    print("unique ids   :", len(seen_ids))
    print("duplicates   :", duplicate_count)

    if total_vecs != total_ids:
        print("[FAIL] total vectors != total ids")
        ok = False

    if duplicate_count > 0:
        print("[FAIL] duplicate doc ids found")
        ok = False

    return total_vecs, total_ids, ok


def check_query_embeddings(base: Path, expected_dim: int, full_scan: bool) -> tuple[int, bool]:
    query_emb_path = base / "query_embeddings.npy"
    query_ids_path = base / "query_ids.json"

    print("=" * 80)
    print("[QUERY CHECK]")
    print("query embeddings:", query_emb_path)
    print("query ids       :", query_ids_path)

    ok = True

    if not query_emb_path.exists():
        print("[FAIL] query_embeddings.npy not found:", query_emb_path)
        return 0, False

    if not query_ids_path.exists():
        print("[FAIL] query_ids.json not found:", query_ids_path)
        return 0, False

    q_emb = np.load(query_emb_path, mmap_mode="r")
    q_ids = [str(x) for x in load_json(query_ids_path)]

    print("query embeddings:", q_emb.shape)
    print("query ids       :", len(q_ids))

    if q_emb.ndim != 2:
        print("[FAIL] query embedding is not 2D:", q_emb.shape)
        ok = False

    if q_emb.ndim == 2 and q_emb.shape[1] != expected_dim:
        print(
            "[FAIL] query embedding dim mismatch:",
            "expected =",
            expected_dim,
            "actual =",
            q_emb.shape,
        )
        ok = False

    if q_emb.shape[0] != len(q_ids):
        print("[FAIL] query embedding row count != query id count")
        ok = False

    if len(q_ids) != len(set(q_ids)):
        print("[FAIL] duplicate query ids found:", len(q_ids) - len(set(q_ids)))
        ok = False

    if full_scan:
        arr = np.asarray(q_emb)
        if not np.isfinite(arr).all():
            print("[FAIL] NaN or inf found in query embeddings")
            ok = False

    return len(q_ids), ok


def check_metadata(base: Path) -> None:
    print("=" * 80)
    print("[METADATA CHECK]")

    metadata_path = base / "metadata.json"
    partial_path = base / "metadata_partial.json"

    if metadata_path.exists():
        print("metadata.json exists:", metadata_path)
        try:
            metadata = load_json(metadata_path)
            for key in [
                "dataset_name",
                "split",
                "model_name",
                "chunk_size",
                "total_corpus_docs",
                "total_encoded",
                "query_count",
            ]:
                if key in metadata:
                    print(f"{key}: {metadata[key]}")
        except Exception as e:
            print("[WARN] failed to read metadata.json:", e)
    else:
        print("[WARN] metadata.json not found")

    if partial_path.exists():
        print("metadata_partial.json exists:", partial_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick check chunked BEIR embeddings.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g., msmarco, fever, trec-covid")
    parser.add_argument("--embedding-root", default="data/embeddings")
    parser.add_argument("--expected-dim", type=int, default=384)
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Check NaN/inf by scanning all embedding arrays. Slower for large datasets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base = Path(args.embedding_root) / args.dataset

    print("=" * 80)
    print("Quick check chunked embeddings")
    print("dataset       :", args.dataset)
    print("embedding root:", args.embedding_root)
    print("embedding dir :", base)
    print("expected dim  :", args.expected_dim)
    print("full scan     :", args.full_scan)
    print("=" * 80)

    doc_vecs, doc_ids, doc_ok = check_doc_chunks(
        base=base,
        expected_dim=args.expected_dim,
        full_scan=args.full_scan,
    )

    query_count, query_ok = check_query_embeddings(
        base=base,
        expected_dim=args.expected_dim,
        full_scan=args.full_scan,
    )

    check_metadata(base)

    print("=" * 80)
    print("[SUMMARY]")
    print("dataset      :", args.dataset)
    print("doc vectors  :", doc_vecs)
    print("doc ids      :", doc_ids)
    print("query count  :", query_count)
    print("doc check    :", "PASS" if doc_ok else "FAIL")
    print("query check  :", "PASS" if query_ok else "FAIL")

    if doc_ok and query_ok:
        print("[OVERALL PASS]")
    else:
        print("[OVERALL FAIL]")
    print("=" * 80)


if __name__ == "__main__":
    main()