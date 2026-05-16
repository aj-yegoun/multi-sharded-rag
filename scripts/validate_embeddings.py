"""
Validate BEIR embedding artifacts for the multi-sharded RAG project.

Run from the project root, for example:

python scripts/validate_embeddings.py \
  --dataset msmarco \
  --raw-data-dir data/raw/beir \
  --embedding-dir data/embeddings/msmarco

Expected embedding directory files:
- doc_embeddings.npy
- query_embeddings.npy
- doc_ids.json
- query_ids.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def human_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def load_beir_dataset(dataset: str, raw_data_dir: Path, split: str):
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    raw_data_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = raw_data_dir / dataset

    if not dataset_dir.exists():
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        print(f"[INFO] Dataset directory not found. Downloading {dataset} from:")
        print(f"       {url}")
        data_path = util.download_and_unzip(url, str(raw_data_dir))
    else:
        data_path = str(dataset_dir)

    print(f"[INFO] Loading BEIR dataset: dataset={dataset}, split={split}")
    return GenericDataLoader(data_folder=data_path).load(split=split)


def check_duplicate_ids(ids: list[str], name: str) -> int:
    unique_count = len(set(ids))
    duplicate_count = len(ids) - unique_count
    status = "PASS" if duplicate_count == 0 else "FAIL"
    print(f"[{status}] {name} duplicate check: {duplicate_count} duplicates")
    return duplicate_count


def check_shape(emb: np.ndarray, ids: list[str], name: str) -> bool:
    ok = emb.ndim == 2 and emb.shape[0] == len(ids)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name} shape/id length: shape={emb.shape}, ids={len(ids)}")
    return ok


def check_finite_and_norms(
    emb: np.ndarray,
    name: str,
    expected_norm: float | None,
    norm_tolerance: float,
    full_scan: bool,
    chunk_size: int,
) -> dict[str, Any]:
    n = emb.shape[0]
    if n == 0:
        print(f"[FAIL] {name} is empty")
        return {"finite": False, "empty": True}

    if full_scan:
        ranges = [(start, min(start + chunk_size, n)) for start in range(0, n, chunk_size)]
        scan_label = "full"
    else:
        # Sample first, middle, last rows plus up to 2048 evenly spaced rows.
        sample_count = min(2048, n)
        indices = np.linspace(0, n - 1, sample_count, dtype=np.int64)
        block = np.asarray(emb[indices])
        finite = bool(np.isfinite(block).all())
        norms = np.linalg.norm(block.astype(np.float32, copy=False), axis=1)
        print(f"[{'PASS' if finite else 'FAIL'}] {name} finite check: sampled {sample_count} rows")
        _print_norm_summary(name, norms, expected_norm, norm_tolerance, sampled=True)
        return {
            "finite": finite,
            "empty": False,
            "norm_min": float(norms.min()),
            "norm_mean": float(norms.mean()),
            "norm_max": float(norms.max()),
            "sampled": True,
        }

    finite = True
    norm_min = math.inf
    norm_max = -math.inf
    norm_sum = 0.0
    norm_count = 0

    for start, end in ranges:
        block = np.asarray(emb[start:end])
        if not np.isfinite(block).all():
            finite = False
        norms = np.linalg.norm(block.astype(np.float32, copy=False), axis=1)
        norm_min = min(norm_min, float(norms.min()))
        norm_max = max(norm_max, float(norms.max()))
        norm_sum += float(norms.sum())
        norm_count += len(norms)

    norm_mean = norm_sum / max(norm_count, 1)
    print(f"[{'PASS' if finite else 'FAIL'}] {name} finite check: {scan_label} scan, rows={n}")
    _print_norm_summary(
        name,
        np.array([norm_min, norm_mean, norm_max], dtype=np.float32),
        expected_norm,
        norm_tolerance,
        sampled=False,
        explicit=(norm_min, norm_mean, norm_max),
    )
    return {
        "finite": finite,
        "empty": False,
        "norm_min": norm_min,
        "norm_mean": norm_mean,
        "norm_max": norm_max,
        "sampled": False,
    }


def _print_norm_summary(
    name: str,
    norms: np.ndarray,
    expected_norm: float | None,
    norm_tolerance: float,
    sampled: bool,
    explicit: tuple[float, float, float] | None = None,
) -> None:
    if explicit is None:
        norm_min = float(norms.min())
        norm_mean = float(norms.mean())
        norm_max = float(norms.max())
    else:
        norm_min, norm_mean, norm_max = explicit

    print(
        f"[INFO] {name} norm summary"
        f" ({'sampled' if sampled else 'full'}):"
        f" min={norm_min:.6f}, mean={norm_mean:.6f}, max={norm_max:.6f}"
    )

    if expected_norm is not None:
        ok = (
            abs(norm_min - expected_norm) <= norm_tolerance
            and abs(norm_mean - expected_norm) <= norm_tolerance
            and abs(norm_max - expected_norm) <= norm_tolerance
        )
        print(
            f"[{'PASS' if ok else 'WARN'}] {name} norm ~= {expected_norm}"
            f" with tolerance {norm_tolerance}"
        )


def check_dataset_coverage(
    corpus: dict,
    queries: dict,
    qrels: dict,
    doc_ids: list[str],
    query_ids: list[str],
) -> dict[str, Any]:
    corpus_ids = set(corpus.keys())
    query_set = set(queries.keys())
    doc_id_set = set(doc_ids)
    query_id_set = set(query_ids)

    missing_doc_ids = doc_id_set - corpus_ids
    missing_corpus_ids = corpus_ids - doc_id_set
    missing_query_ids = query_id_set - query_set
    missing_queries = query_set - query_id_set

    print(f"[{'PASS' if not missing_doc_ids else 'FAIL'}] doc_ids all exist in corpus: missing={len(missing_doc_ids)}")
    print(f"[{'PASS' if not missing_corpus_ids else 'FAIL'}] corpus docs all embedded: missing={len(missing_corpus_ids)}")
    print(f"[{'PASS' if not missing_query_ids else 'FAIL'}] query_ids all exist in queries: missing={len(missing_query_ids)}")
    print(f"[{'PASS' if not missing_queries else 'FAIL'}] query texts all embedded: missing={len(missing_queries)}")

    qrel_query_ids = set(qrels.keys())
    qrel_queries_missing_embedding = qrel_query_ids - query_id_set

    total_positive_qrels = 0
    positive_qrels_missing_doc = 0
    qrels_with_no_embedded_positive_doc = 0

    for qid, rels in qrels.items():
        positive_docs = [doc_id for doc_id, rel in rels.items() if rel > 0]
        total_positive_qrels += len(positive_docs)
        missing_for_query = [doc_id for doc_id in positive_docs if doc_id not in doc_id_set]
        positive_qrels_missing_doc += len(missing_for_query)
        if positive_docs and len(missing_for_query) == len(positive_docs):
            qrels_with_no_embedded_positive_doc += 1

    qrel_doc_coverage = (
        1.0 - positive_qrels_missing_doc / total_positive_qrels
        if total_positive_qrels else 0.0
    )

    print(f"[INFO] qrels queries: {len(qrel_query_ids)}")
    print(f"[{'PASS' if not qrel_queries_missing_embedding else 'FAIL'}] qrels query ids embedded: missing={len(qrel_queries_missing_embedding)}")
    print(f"[INFO] positive qrels: {total_positive_qrels}")
    print(f"[{'PASS' if positive_qrels_missing_doc == 0 else 'FAIL'}] qrels positive doc ids embedded: missing={positive_qrels_missing_doc}, coverage={qrel_doc_coverage:.6f}")
    print(f"[{'PASS' if qrels_with_no_embedded_positive_doc == 0 else 'FAIL'}] qrels with no embedded positive doc: {qrels_with_no_embedded_positive_doc}")

    rel_dq = total_positive_qrels / len(qrel_query_ids) if qrel_query_ids else 0.0
    print(f"[INFO] Rel D/Q from loaded qrels: {rel_dq:.4f}")

    return {
        "missing_doc_ids": len(missing_doc_ids),
        "missing_corpus_ids": len(missing_corpus_ids),
        "missing_query_ids": len(missing_query_ids),
        "missing_queries": len(missing_queries),
        "qrel_queries_missing_embedding": len(qrel_queries_missing_embedding),
        "total_positive_qrels": total_positive_qrels,
        "positive_qrels_missing_doc": positive_qrels_missing_doc,
        "qrel_doc_coverage": qrel_doc_coverage,
        "qrels_with_no_embedded_positive_doc": qrels_with_no_embedded_positive_doc,
        "rel_dq": rel_dq,
    }


def topk_sanity_check(
    doc_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    doc_ids: list[str],
    query_ids: list[str],
    qrels: dict,
    num_queries: int,
    top_k: int,
    chunk_size: int,
) -> None:
    if num_queries <= 0:
        print("[SKIP] top-k sanity check disabled")
        return

    doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    query_id_to_index = {query_id: idx for idx, query_id in enumerate(query_ids)}

    candidate_qids = [
        qid for qid, rels in qrels.items()
        if qid in query_id_to_index and any(rel > 0 and doc_id in doc_id_to_index for doc_id, rel in rels.items())
    ]
    candidate_qids = candidate_qids[:num_queries]

    if not candidate_qids:
        print("[WARN] No qrels queries available for top-k sanity check")
        return

    print(f"[INFO] Running top-{top_k} sanity check for {len(candidate_qids)} queries...")
    hits = 0
    for qid in candidate_qids:
        qidx = query_id_to_index[qid]
        qvec = np.asarray(query_embeddings[qidx], dtype=np.float32)
        top_scores = np.full(top_k, -np.inf, dtype=np.float32)
        top_indices = np.full(top_k, -1, dtype=np.int64)

        for start in range(0, doc_embeddings.shape[0], chunk_size):
            end = min(start + chunk_size, doc_embeddings.shape[0])
            block = np.asarray(doc_embeddings[start:end], dtype=np.float32)
            scores = block @ qvec

            if len(scores) <= top_k:
                local_idx = np.argsort(-scores)
            else:
                local_idx = np.argpartition(-scores, top_k - 1)[:top_k]
                local_idx = local_idx[np.argsort(-scores[local_idx])]

            combined_scores = np.concatenate([top_scores, scores[local_idx]])
            combined_indices = np.concatenate([top_indices, start + local_idx])
            keep = np.argsort(-combined_scores)[:top_k]
            top_scores = combined_scores[keep]
            top_indices = combined_indices[keep]

        retrieved_doc_ids = {doc_ids[i] for i in top_indices if i >= 0}
        relevant_doc_ids = {doc_id for doc_id, rel in qrels[qid].items() if rel > 0}
        hit = bool(retrieved_doc_ids & relevant_doc_ids)
        hits += int(hit)
        print(f"  qid={qid}: hit@{top_k}={hit}, positives={len(relevant_doc_ids)}")

    print(f"[INFO] sanity hit@{top_k}: {hits}/{len(candidate_qids)}")
    print("[NOTE] This is only a rough embedding sanity check, not a required pass/fail test.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="BEIR dataset name, e.g., msmarco, fever, scifact")
    parser.add_argument("--split", default="test", help="BEIR split to load, default: test")
    parser.add_argument("--raw-data-dir", default="data/raw/beir")
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--expected-dim", type=int, default=None, help="Expected embedding dimension, e.g., 384")
    parser.add_argument("--expected-norm", type=float, default=1.0, help="Expected L2 norm if normalized; use --expected-norm -1 to disable")
    parser.add_argument("--norm-tolerance", type=float, default=0.02)
    parser.add_argument("--full-scan", action="store_true", help="Scan all rows for finite values/norms; default samples rows")
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--sanity-topk", type=int, default=0, help="Run top-k sanity check; 0 disables")
    parser.add_argument("--num-sanity-queries", type=int, default=5)
    args = parser.parse_args()

    raw_data_dir = Path(args.raw_data_dir)
    embedding_dir = Path(args.embedding_dir)

    doc_emb_path = embedding_dir / "doc_embeddings.npy"
    query_emb_path = embedding_dir / "query_embeddings.npy"
    doc_ids_path = embedding_dir / "doc_ids.json"
    query_ids_path = embedding_dir / "query_ids.json"

    print("=" * 80)
    print("Embedding validation")
    print(f"dataset       : {args.dataset}")
    print(f"split         : {args.split}")
    print(f"raw_data_dir  : {raw_data_dir}")
    print(f"embedding_dir : {embedding_dir}")
    print("=" * 80)

    for path in [doc_emb_path, query_emb_path, doc_ids_path, query_ids_path]:
        require_file(path)
        print(f"[PASS] found {path} ({human_bytes(path.stat().st_size)})")

    # mmap_mode keeps large MS-MARCO embeddings from being loaded all at once.
    doc_embeddings = np.load(doc_emb_path, mmap_mode="r")
    query_embeddings = np.load(query_emb_path, mmap_mode="r")
    doc_ids = load_json(doc_ids_path)
    query_ids = load_json(query_ids_path)

    print("-" * 80)
    print(f"[INFO] doc_embeddings: shape={doc_embeddings.shape}, dtype={doc_embeddings.dtype}")
    print(f"[INFO] query_embeddings: shape={query_embeddings.shape}, dtype={query_embeddings.dtype}")
    print(f"[INFO] num doc_ids: {len(doc_ids)}")
    print(f"[INFO] num query_ids: {len(query_ids)}")

    ok = True
    ok &= check_shape(doc_embeddings, doc_ids, "doc_embeddings")
    ok &= check_shape(query_embeddings, query_ids, "query_embeddings")

    if args.expected_dim is not None:
        doc_dim_ok = doc_embeddings.ndim == 2 and doc_embeddings.shape[1] == args.expected_dim
        query_dim_ok = query_embeddings.ndim == 2 and query_embeddings.shape[1] == args.expected_dim
        print(f"[{'PASS' if doc_dim_ok else 'FAIL'}] doc embedding dim == {args.expected_dim}")
        print(f"[{'PASS' if query_dim_ok else 'FAIL'}] query embedding dim == {args.expected_dim}")
        ok &= doc_dim_ok and query_dim_ok

    ok &= check_duplicate_ids(doc_ids, "doc_ids") == 0
    ok &= check_duplicate_ids(query_ids, "query_ids") == 0

    expected_norm = None if args.expected_norm < 0 else args.expected_norm
    print("-" * 80)
    doc_norm_stats = check_finite_and_norms(
        doc_embeddings,
        "doc_embeddings",
        expected_norm,
        args.norm_tolerance,
        args.full_scan,
        args.chunk_size,
    )
    query_norm_stats = check_finite_and_norms(
        query_embeddings,
        "query_embeddings",
        expected_norm,
        args.norm_tolerance,
        args.full_scan,
        args.chunk_size,
    )
    ok &= bool(doc_norm_stats.get("finite")) and bool(query_norm_stats.get("finite"))

    print("-" * 80)
    corpus, queries, qrels = load_beir_dataset(args.dataset, raw_data_dir, args.split)
    print(f"[INFO] loaded corpus={len(corpus)}, queries={len(queries)}, qrels_queries={len(qrels)}")
    coverage_stats = check_dataset_coverage(corpus, queries, qrels, doc_ids, query_ids)

    ok &= coverage_stats["missing_doc_ids"] == 0
    ok &= coverage_stats["missing_corpus_ids"] == 0
    ok &= coverage_stats["missing_query_ids"] == 0
    ok &= coverage_stats["missing_queries"] == 0
    ok &= coverage_stats["qrel_queries_missing_embedding"] == 0
    ok &= coverage_stats["positive_qrels_missing_doc"] == 0

    print("-" * 80)
    topk_sanity_check(
        doc_embeddings=doc_embeddings,
        query_embeddings=query_embeddings,
        doc_ids=doc_ids,
        query_ids=query_ids,
        qrels=qrels,
        num_queries=args.num_sanity_queries if args.sanity_topk > 0 else 0,
        top_k=args.sanity_topk,
        chunk_size=args.chunk_size,
    )

    print("=" * 80)
    if ok:
        print("[OVERALL PASS] Embedding artifacts look consistent and usable.")
    else:
        print("[OVERALL FAIL/WARN] Some checks failed. Review messages above before continuing.")
    print("=" * 80)


if __name__ == "__main__":
    main()
