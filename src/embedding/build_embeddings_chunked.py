from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_numpy(arr: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    np.save(path, arr)


def build_doc_text(doc: dict) -> str:
    title = doc.get("title", "") or ""
    text = doc.get("text", "") or ""
    return f"{title} {text}".strip()


def iter_corpus_items(corpus: dict) -> Iterable[tuple[str, str]]:
    for doc_id, doc in corpus.items():
        yield str(doc_id), build_doc_text(doc)


def download_beir_dataset_if_needed(dataset_name: str, raw_data_dir: str | Path) -> Path:
    raw_data_dir = ensure_dir(raw_data_dir)
    dataset_dir = raw_data_dir / dataset_name

    if dataset_dir.exists():
        print(f"[INFO] Using existing dataset directory: {dataset_dir}")
        return dataset_dir

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    print(f"[INFO] Dataset directory not found: {dataset_dir}")
    print(f"[INFO] Downloading {dataset_name} from {url}")

    downloaded_path = util.download_and_unzip(url, str(raw_data_dir))
    return Path(downloaded_path)


def load_beir_dataset(
    dataset_name: str,
    raw_data_dir: str | Path,
    split: str,
):
    dataset_dir = download_beir_dataset_if_needed(dataset_name, raw_data_dir)

    print(f"[INFO] Loading BEIR dataset")
    print(f"       dataset_dir: {dataset_dir}")
    print(f"       split      : {split}")

    corpus, queries, qrels = GenericDataLoader(data_folder=str(dataset_dir)).load(split=split)

    corpus = {str(k): v for k, v in corpus.items()}
    queries = {str(k): v for k, v in queries.items()}
    qrels = {
        str(qid): {str(doc_id): int(score) for doc_id, score in doc_scores.items()}
        for qid, doc_scores in qrels.items()
    }

    return corpus, queries, qrels, dataset_dir


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    return embeddings.astype(np.float32)


def save_doc_chunk(
    output_dir: Path,
    chunk_idx: int,
    doc_ids: list[str],
    embeddings: np.ndarray,
) -> dict:
    chunk_name = f"chunk_{chunk_idx:05d}"
    emb_path = output_dir / "doc_chunks" / f"{chunk_name}.npy"
    ids_path = output_dir / "doc_id_chunks" / f"{chunk_name}.json"

    save_numpy(embeddings, emb_path)
    save_json(doc_ids, ids_path)

    return {
        "chunk_idx": chunk_idx,
        "embedding_path": str(emb_path),
        "doc_ids_path": str(ids_path),
        "num_docs": len(doc_ids),
        "shape": list(embeddings.shape),
    }


def chunk_exists(output_dir: Path, chunk_idx: int) -> bool:
    chunk_name = f"chunk_{chunk_idx:05d}"
    emb_path = output_dir / "doc_chunks" / f"{chunk_name}.npy"
    ids_path = output_dir / "doc_id_chunks" / f"{chunk_name}.json"
    return emb_path.exists() and ids_path.exists()


def load_existing_chunk_meta(output_dir: Path, chunk_idx: int) -> dict | None:
    chunk_name = f"chunk_{chunk_idx:05d}"
    emb_path = output_dir / "doc_chunks" / f"{chunk_name}.npy"
    ids_path = output_dir / "doc_id_chunks" / f"{chunk_name}.json"

    if not emb_path.exists() or not ids_path.exists():
        return None

    emb = np.load(emb_path, mmap_mode="r")
    ids = load_json(ids_path)

    return {
        "chunk_idx": chunk_idx,
        "embedding_path": str(emb_path),
        "doc_ids_path": str(ids_path),
        "num_docs": len(ids),
        "shape": list(emb.shape),
        "skipped_by_resume": True,
    }


def select_query_ids_from_qrels(
    qrels: dict[str, dict[str, int]],
    positive_only: bool,
) -> list[str]:
    selected = []

    for qid, doc_scores in qrels.items():
        if positive_only:
            if any(score > 0 for score in doc_scores.values()):
                selected.append(str(qid))
        else:
            selected.append(str(qid))

    return sorted(selected, key=lambda x: int(x) if x.isdigit() else x)


def build_query_embeddings(
    model: SentenceTransformer,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    output_dir: Path,
    batch_size: int,
    normalize_embeddings: bool = True,
    positive_only: bool = False,
) -> dict:
    query_ids = select_query_ids_from_qrels(qrels, positive_only=positive_only)

    missing_query_ids = [qid for qid in query_ids if qid not in queries]
    if missing_query_ids:
        raise RuntimeError(
            f"{len(missing_query_ids)} qids from qrels are missing in queries. "
            f"Examples: {missing_query_ids[:10]}"
        )

    query_texts = [queries[qid] for qid in query_ids]

    print(f"[INFO] Encoding queries")
    print(f"       selected queries: {len(query_ids)}")
    print(f"       positive_only   : {positive_only}")

    start = time.time()
    query_embeddings = encode_texts(
        model=model,
        texts=query_texts,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
    )
    elapsed = time.time() - start

    save_numpy(query_embeddings, output_dir / "query_embeddings.npy")
    save_json(query_ids, output_dir / "query_ids.json")

    print(f"[DONE] Query embeddings saved: {query_embeddings.shape}")
    print(f"       elapsed: {elapsed:.2f}s")

    return {
        "query_count": len(query_ids),
        "query_embedding_shape": list(query_embeddings.shape),
        "query_elapsed_sec": elapsed,
        "query_positive_only": positive_only,
    }


def build_corpus_embeddings_chunked(
    dataset_name: str,
    raw_data_dir: str,
    output_dir: str,
    split: str,
    model_name: str,
    batch_size: int,
    chunk_size: int,
    normalize_embeddings: bool,
    max_docs: int | None,
    build_docs: bool,
    build_queries: bool,
    positive_only_queries: bool,
    resume: bool,
) -> None:
    output_dir = ensure_dir(output_dir)
    ensure_dir(output_dir / "doc_chunks")
    ensure_dir(output_dir / "doc_id_chunks")

    print("=" * 80)
    print("Build chunked embeddings")
    print(f"dataset_name          : {dataset_name}")
    print(f"split                 : {split}")
    print(f"raw_data_dir          : {raw_data_dir}")
    print(f"output_dir            : {output_dir}")
    print(f"model_name            : {model_name}")
    print(f"batch_size            : {batch_size}")
    print(f"chunk_size            : {chunk_size}")
    print(f"normalize_embeddings  : {normalize_embeddings}")
    print(f"max_docs              : {max_docs}")
    print(f"build_docs            : {build_docs}")
    print(f"build_queries         : {build_queries}")
    print(f"positive_only_queries : {positive_only_queries}")
    print(f"resume                : {resume}")
    print("=" * 80)

    corpus, queries, qrels, dataset_dir = load_beir_dataset(
        dataset_name=dataset_name,
        raw_data_dir=raw_data_dir,
        split=split,
    )

    print(f"[INFO] Corpus docs : {len(corpus)}")
    print(f"[INFO] Queries     : {len(queries)}")
    print(f"[INFO] qrels queries: {len(qrels)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading model: {model_name}")

    model = SentenceTransformer(model_name, device=device)

    metadata = {
        "dataset_name": dataset_name,
        "dataset_dir": str(dataset_dir),
        "split": split,
        "model_name": model_name,
        "batch_size": batch_size,
        "chunk_size": chunk_size,
        "normalize_embeddings": normalize_embeddings,
        "total_corpus_docs": len(corpus),
        "total_queries_loaded": len(queries),
        "total_qrels_queries": len(qrels),
        "max_docs": max_docs,
        "build_docs": build_docs,
        "build_queries": build_queries,
        "positive_only_queries": positive_only_queries,
        "device": device,
        "chunks": [],
    }

    if build_queries:
        query_meta = build_query_embeddings(
            model=model,
            queries=queries,
            qrels=qrels,
            output_dir=output_dir,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            positive_only=positive_only_queries,
        )
        metadata.update(query_meta)

    if not build_docs:
        save_json(metadata, output_dir / "metadata.json")
        print("[DONE] Query embeddings only. Document embedding skipped.")
        return

    current_doc_ids: list[str] = []
    current_texts: list[str] = []
    chunk_idx = 0
    total_encoded = 0
    start_all = time.time()

    effective_total_docs = len(corpus) if max_docs is None else min(max_docs, len(corpus))
    expected_chunks = math.ceil(effective_total_docs / chunk_size)

    print(f"[INFO] Effective docs to encode: {effective_total_docs}")
    print(f"[INFO] Expected chunks          : {expected_chunks}")

    for doc_idx, (doc_id, doc_text) in enumerate(iter_corpus_items(corpus)):
        if max_docs is not None and total_encoded + len(current_doc_ids) >= max_docs:
            break

        current_doc_ids.append(doc_id)
        current_texts.append(doc_text)

        if len(current_doc_ids) >= chunk_size:
            if resume and chunk_exists(output_dir, chunk_idx):
                existing_meta = load_existing_chunk_meta(output_dir, chunk_idx)
                if existing_meta is None:
                    raise RuntimeError(f"Resume check failed for chunk {chunk_idx}")

                print(f"[SKIP] chunk={chunk_idx:05d} already exists")
                metadata["chunks"].append(existing_meta)
                total_encoded += existing_meta["num_docs"]

                current_doc_ids = []
                current_texts = []
                chunk_idx += 1
                save_json(metadata, output_dir / "metadata_partial.json")
                continue

            print(
                f"[INFO] Encoding chunk {chunk_idx:05d} "
                f"docs={len(current_doc_ids)} "
                f"total_before={total_encoded}"
            )

            start = time.time()
            embeddings = encode_texts(
                model=model,
                texts=current_texts,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings,
            )
            elapsed = time.time() - start

            chunk_meta = save_doc_chunk(
                output_dir=output_dir,
                chunk_idx=chunk_idx,
                doc_ids=current_doc_ids,
                embeddings=embeddings,
            )
            chunk_meta["elapsed_sec"] = elapsed
            chunk_meta["docs_per_sec"] = len(current_doc_ids) / elapsed if elapsed > 0 else 0
            metadata["chunks"].append(chunk_meta)

            total_encoded += len(current_doc_ids)

            print(
                f"[DONE] chunk={chunk_idx:05d}, "
                f"total_encoded={total_encoded}, "
                f"elapsed={elapsed:.2f}s, "
                f"throughput={chunk_meta['docs_per_sec']:.2f} docs/sec"
            )

            current_doc_ids = []
            current_texts = []
            chunk_idx += 1

            save_json(metadata, output_dir / "metadata_partial.json")

    if current_doc_ids:
        if resume and chunk_exists(output_dir, chunk_idx):
            existing_meta = load_existing_chunk_meta(output_dir, chunk_idx)
            if existing_meta is None:
                raise RuntimeError(f"Resume check failed for final chunk {chunk_idx}")

            print(f"[SKIP] final chunk={chunk_idx:05d} already exists")
            metadata["chunks"].append(existing_meta)
            total_encoded += existing_meta["num_docs"]
        else:
            print(
                f"[INFO] Encoding final chunk {chunk_idx:05d} "
                f"docs={len(current_doc_ids)} "
                f"total_before={total_encoded}"
            )

            start = time.time()
            embeddings = encode_texts(
                model=model,
                texts=current_texts,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings,
            )
            elapsed = time.time() - start

            chunk_meta = save_doc_chunk(
                output_dir=output_dir,
                chunk_idx=chunk_idx,
                doc_ids=current_doc_ids,
                embeddings=embeddings,
            )
            chunk_meta["elapsed_sec"] = elapsed
            chunk_meta["docs_per_sec"] = len(current_doc_ids) / elapsed if elapsed > 0 else 0
            metadata["chunks"].append(chunk_meta)

            total_encoded += len(current_doc_ids)

            print(
                f"[DONE] final chunk={chunk_idx:05d}, "
                f"total_encoded={total_encoded}, "
                f"elapsed={elapsed:.2f}s, "
                f"throughput={chunk_meta['docs_per_sec']:.2f} docs/sec"
            )

    elapsed_all = time.time() - start_all
    metadata["total_encoded"] = total_encoded
    metadata["total_elapsed_sec"] = elapsed_all
    metadata["overall_docs_per_sec"] = total_encoded / elapsed_all if elapsed_all > 0 else 0
    metadata["num_chunks_saved"] = len(metadata["chunks"])

    save_json(metadata, output_dir / "metadata.json")

    print("=" * 80)
    print("[DONE] Chunked embeddings saved.")
    print(f"dataset              : {dataset_name}")
    print(f"split                : {split}")
    print(f"total_encoded        : {total_encoded}")
    print(f"num_chunks_saved     : {len(metadata['chunks'])}")
    print(f"elapsed              : {elapsed_all / 3600:.2f} hours")
    print(f"overall throughput   : {metadata['overall_docs_per_sec']:.2f} docs/sec")
    print(f"output_dir           : {output_dir}")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build chunked BEIR embeddings.")

    parser.add_argument("--dataset", required=True, help="BEIR dataset name, e.g., msmarco, fever, trec-covid")
    parser.add_argument("--split", default="test", help="BEIR qrels split, e.g., dev, test, train")
    parser.add_argument("--raw-data-dir", default="data/raw/beir")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--max-docs", type=int, default=None)

    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--skip-docs", action="store_true", help="Only build query embeddings.")
    parser.add_argument("--skip-queries", action="store_true", help="Only build document embeddings.")
    parser.add_argument(
        "--positive-only-queries",
        action="store_true",
        help="Build query embeddings only for qrels queries that have at least one positive score.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = f"data/embeddings/{args.dataset}"

    build_corpus_embeddings_chunked(
        dataset_name=args.dataset,
        raw_data_dir=args.raw_data_dir,
        output_dir=output_dir,
        split=args.split,
        model_name=args.model_name,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        normalize_embeddings=not args.no_normalize,
        max_docs=args.max_docs,
        build_docs=not args.skip_docs,
        build_queries=not args.skip_queries,
        positive_only_queries=args.positive_only_queries,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()