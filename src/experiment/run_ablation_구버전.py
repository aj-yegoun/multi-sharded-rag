from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml
from sklearn.cluster import MiniBatchKMeans

from src.data.load_beir import load_beir_dataset
from src.embedding.chunked_embedding_store import load_embedding_bundle
from src.evaluation.document_recall import (
    document_recall_at_k_within_selected_shards,
    get_positive_relevant_docs,
    oracle_relevant_document_coverage,
)
from src.evaluation.group_recall import group_recall_at_g, is_group_drop_error
from src.evaluation.relevant_shard import build_relevant_shards
from src.evaluation.result_writer import save_results_csv
from src.evaluation.shard_recall import shard_recall_at_b
from src.experiment.run_pilot import build_doc_to_shard, print_shard_size_summary
from src.grouping.build_groups import build_groups_from_shard_centroids, print_group_summary
from src.ranking.group_multi_ranker import rank_shards_group_multi_centroid
from src.ranking.group_single_ranker import rank_shards_group_single_centroid
from src.ranking.multi_centroid_ranker import rank_shards_multi_centroid
from src.ranking.single_centroid_ranker import rank_shards_single_centroid


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_qrels(qrels: dict) -> dict[str, dict[str, int]]:
    return {
        str(query_id): {str(doc_id): int(rel) for doc_id, rel in doc_rels.items()}
        for query_id, doc_rels in qrels.items()
    }


def iter_embedding_batches(
    doc_embeddings,
    doc_ids: list[str],
    batch_size: int,
) -> Iterator[tuple[np.ndarray, list[str], int]]:
    """
    Yield (batch_embeddings, batch_doc_ids, global_start_index).

    For chunked embeddings, this uses the original chunks directly.
    For flat memmap embeddings, it slices by batch_size.
    """
    if hasattr(doc_embeddings, "iter_chunks"):
        for chunk_embeddings, chunk_doc_ids, start_idx in doc_embeddings.iter_chunks():
            yield chunk_embeddings, chunk_doc_ids, start_idx
        return

    n = len(doc_ids)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield doc_embeddings[start:end], doc_ids[start:end], start


def get_doc_matrix(doc_embeddings, doc_id_to_index: dict[str, int], doc_ids: list[str]) -> np.ndarray:
    if hasattr(doc_embeddings, "get_by_doc_ids"):
        return doc_embeddings.get_by_doc_ids(doc_ids)
    indices = [doc_id_to_index[str(doc_id)] for doc_id in doc_ids]
    return doc_embeddings[indices]


def build_random_shards_streaming(
    doc_ids: list[str],
    num_shards: int,
    seed: int,
) -> dict[int, list[str]]:
    rng = random.Random(seed)
    shuffled = list(doc_ids)
    rng.shuffle(shuffled)

    shards: dict[int, list[str]] = {sid: [] for sid in range(num_shards)}
    for idx, doc_id in enumerate(shuffled):
        shards[idx % num_shards].append(str(doc_id))
    return shards


def _fit_minibatch_kmeans_streaming(
    doc_embeddings,
    doc_ids: list[str],
    n_clusters: int,
    seed: int,
    batch_size: int,
    label: str,
) -> MiniBatchKMeans:
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")
    if n_clusters > len(doc_ids):
        raise ValueError(f"n_clusters({n_clusters}) cannot be larger than num_docs({len(doc_ids)})")

    effective_batch_size = max(batch_size, n_clusters * 4)
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=effective_batch_size,
        n_init="auto",
        reassignment_ratio=0.01,
    )

    print(f"[INFO] Fitting MiniBatchKMeans for {label}: n_clusters={n_clusters}")
    seen = 0
    for batch_embeddings, _batch_doc_ids, _start_idx in iter_embedding_batches(
        doc_embeddings, doc_ids, effective_batch_size
    ):
        if len(batch_embeddings) == 0:
            continue
        kmeans.partial_fit(batch_embeddings)
        seen += len(batch_embeddings)
        if seen % max(effective_batch_size * 20, 1) == 0:
            print(f"[INFO] {label}: partial_fit seen={seen}/{len(doc_ids)}")

    return kmeans


def build_topic_based_shards_streaming(
    doc_ids: list[str],
    doc_embeddings,
    num_shards: int,
    seed: int,
    batch_size: int,
) -> dict[int, list[str]]:
    """
    Topic-based sharding without materializing the full embedding matrix.
    It runs MiniBatchKMeans in a first pass and predicts labels in a second pass.
    """
    kmeans = _fit_minibatch_kmeans_streaming(
        doc_embeddings=doc_embeddings,
        doc_ids=doc_ids,
        n_clusters=num_shards,
        seed=seed,
        batch_size=batch_size,
        label=f"topic_shards_{num_shards}",
    )

    shards: dict[int, list[str]] = {sid: [] for sid in range(num_shards)}
    print(f"[INFO] Assigning documents to topic-based shards: num_shards={num_shards}")
    for batch_embeddings, batch_doc_ids, _start_idx in iter_embedding_batches(doc_embeddings, doc_ids, batch_size):
        labels = kmeans.predict(batch_embeddings)
        for doc_id, label in zip(batch_doc_ids, labels):
            shards[int(label)].append(str(doc_id))

    empty = [sid for sid, ids in shards.items() if not ids]
    if empty:
        raise RuntimeError(f"Some topic-based shards are empty: {empty}")
    return shards


def build_heterogeneous_shards_streaming(
    doc_ids: list[str],
    doc_embeddings,
    num_shards: int,
    seed: int,
    batch_size: int,
    num_topics: int | None = None,
) -> dict[int, list[str]]:
    """
    Heterogeneous sharding without materializing the full embedding matrix.

    First, documents are clustered into topic clusters with MiniBatchKMeans.
    Then each topic cluster is distributed round-robin over shards.
    """
    if num_topics is None:
        num_topics = num_shards

    kmeans = _fit_minibatch_kmeans_streaming(
        doc_embeddings=doc_embeddings,
        doc_ids=doc_ids,
        n_clusters=num_topics,
        seed=seed,
        batch_size=batch_size,
        label=f"heterogeneous_topics_{num_topics}",
    )

    topic_to_docs: dict[int, list[str]] = defaultdict(list)
    print(f"[INFO] Assigning documents to temporary topics: num_topics={num_topics}")
    for batch_embeddings, batch_doc_ids, _start_idx in iter_embedding_batches(doc_embeddings, doc_ids, batch_size):
        labels = kmeans.predict(batch_embeddings)
        for doc_id, label in zip(batch_doc_ids, labels):
            topic_to_docs[int(label)].append(str(doc_id))

    rng = random.Random(seed)
    shards: dict[int, list[str]] = {sid: [] for sid in range(num_shards)}

    for topic_id, topic_doc_ids in topic_to_docs.items():
        topic_doc_ids = list(topic_doc_ids)
        rng.shuffle(topic_doc_ids)
        start_offset = topic_id % num_shards
        for idx, doc_id in enumerate(topic_doc_ids):
            shard_id = (start_offset + idx) % num_shards
            shards[shard_id].append(doc_id)

    empty = [sid for sid, ids in shards.items() if not ids]
    if empty:
        raise RuntimeError(f"Some heterogeneous shards are empty: {empty}")
    return shards


def build_shards_streaming(
    shard_type: str,
    doc_ids: list[str],
    doc_embeddings,
    num_shards: int,
    seed: int,
    batch_size: int,
) -> dict[int, list[str]]:
    shard_type = shard_type.lower()
    if shard_type == "random":
        return build_random_shards_streaming(doc_ids, num_shards, seed)
    if shard_type == "topic_based":
        return build_topic_based_shards_streaming(doc_ids, doc_embeddings, num_shards, seed, batch_size)
    if shard_type == "heterogeneous":
        return build_heterogeneous_shards_streaming(doc_ids, doc_embeddings, num_shards, seed, batch_size)
    raise ValueError(f"Unknown shard_type: {shard_type}")


def compute_single_centroids_streaming(
    shards: dict[int, list[str]],
    doc_ids: list[str],
    doc_embeddings,
    batch_size: int,
) -> dict[int, np.ndarray]:
    """Compute shard mean centroids by streaming over document embedding chunks."""
    doc_to_shard = build_doc_to_shard(shards)
    num_shards = len(shards)
    dim = int(doc_embeddings.shape[1])

    sums = np.zeros((num_shards, dim), dtype=np.float64)
    counts = np.zeros((num_shards,), dtype=np.int64)

    print("[INFO] Computing single centroids by streaming")
    for batch_embeddings, batch_doc_ids, _start_idx in iter_embedding_batches(doc_embeddings, doc_ids, batch_size):
        shard_indices = np.asarray([doc_to_shard[str(doc_id)] for doc_id in batch_doc_ids], dtype=np.int64)
        np.add.at(sums, shard_indices, batch_embeddings)
        np.add.at(counts, shard_indices, 1)

    shard_centroids: dict[int, np.ndarray] = {}
    for shard_id in range(num_shards):
        if counts[shard_id] == 0:
            raise RuntimeError(f"Shard {shard_id} has no documents")
        centroid = sums[shard_id] / counts[shard_id]
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        shard_centroids[shard_id] = centroid.astype(np.float32)

    return shard_centroids


def compute_multi_centroids_memory_safe(
    shards: dict[int, list[str]],
    doc_ids: list[str],
    doc_embeddings,
    k: int,
    seed: int,
    minibatch_size: int,
) -> dict[int, np.ndarray]:
    """
    Compute multi-centroids shard by shard.

    This avoids loading the full corpus embedding matrix. It only materializes one
    shard's document matrix at a time. For very large shards, MiniBatchKMeans is
    used instead of vanilla KMeans.
    """
    doc_id_to_index = {str(doc_id): idx for idx, doc_id in enumerate(doc_ids)}
    shard_multi_centroids: dict[int, np.ndarray] = {}

    for shard_id, shard_doc_ids in shards.items():
        shard_doc_ids = [str(x) for x in shard_doc_ids]
        actual_k = min(k, len(shard_doc_ids))
        shard_matrix = get_doc_matrix(doc_embeddings, doc_id_to_index, shard_doc_ids)

        if actual_k <= 1:
            centroid = shard_matrix.mean(axis=0, keepdims=True)
            norm = np.linalg.norm(centroid, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            shard_multi_centroids[shard_id] = (centroid / norm).astype(np.float32)
            continue

        mb_kmeans = MiniBatchKMeans(
            n_clusters=actual_k,
            random_state=seed,
            batch_size=min(max(minibatch_size, actual_k * 16), len(shard_doc_ids)),
            n_init="auto",
            reassignment_ratio=0.01,
        )
        mb_kmeans.fit(shard_matrix)
        centroids = mb_kmeans.cluster_centers_
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        shard_multi_centroids[shard_id] = (centroids / norms).astype(np.float32)

    return shard_multi_centroids


def centroid_scoring_cost(
    method: str,
    num_shards: int,
    centroids_per_shard: int,
    num_groups: int | None = None,
    num_selected_groups: int | None = None,
    groups: dict[int, list[int]] | None = None,
) -> int:
    if method == "single_centroid":
        return num_shards
    if method == "multi_centroid":
        return num_shards * centroids_per_shard

    if method in {"group_single_centroid", "group_multi_centroid"}:
        if num_groups is None or num_selected_groups is None or groups is None:
            raise ValueError("Grouping cost requires num_groups, num_selected_groups, and groups.")
        avg_group_size = sum(len(v) for v in groups.values()) / len(groups)
        expected_candidate_shards = int(round(avg_group_size * num_selected_groups))
        if method == "group_single_centroid":
            return num_groups + expected_candidate_shards
        return num_groups + expected_candidate_shards * centroids_per_shard

    raise ValueError(f"Unknown method: {method}")


def evaluate_ranked_results(
    *,
    dataset_name: str,
    shard_type: str,
    method: str,
    num_shards: int,
    top_b_values: list[int],
    seed: int,
    query_embeddings: np.ndarray,
    query_ids: list[str],
    qrels: dict,
    relevant_shards_by_query: dict[str, set[int]],
    shards: dict[int, list[str]],
    doc_ids: list[str],
    doc_embeddings,
    rank_fn,
    centroids_per_shard: int,
    num_groups: int | None = None,
    num_selected_groups: int | None = None,
    groups: dict[int, list[int]] | None = None,
    shard_to_group: dict[int, int] | None = None,
    document_recall_k_values: list[int] | None = None,
    document_recall_batch_size: int = 65536,
) -> list[dict]:
    query_id_to_index = {str(query_id): idx for idx, query_id in enumerate(query_ids)}
    doc_id_to_index = {str(doc_id): idx for idx, doc_id in enumerate(doc_ids)}
    document_recall_k_values = document_recall_k_values or []

    results: list[dict] = []

    for top_b in top_b_values:
        shard_recalls: list[float] = []
        oracle_coverages: list[float] = []
        selected_shard_counts: list[int] = []
        doc_recalls_by_k: dict[int, list[float]] = {k: [] for k in document_recall_k_values}
        group_recalls: list[float] = []
        group_drop_errors: list[float] = []

        for query_id, relevant_shards in relevant_shards_by_query.items():
            query_id = str(query_id)
            if query_id not in query_id_to_index:
                continue

            query_embedding = query_embeddings[query_id_to_index[query_id]]
            rank_output = rank_fn(query_embedding)

            if isinstance(rank_output, tuple):
                ranked, selected_groups = rank_output
            else:
                ranked = rank_output
                selected_groups = None

            selected_shards = [shard_id for shard_id, _score in ranked[:top_b]]
            selected_shard_counts.append(len(selected_shards))
            relevant_docs = get_positive_relevant_docs(qrels, query_id)

            shard_recalls.append(
                shard_recall_at_b(
                    selected_shards=selected_shards,
                    relevant_shards=relevant_shards,
                )
            )

            oracle_coverages.append(
                oracle_relevant_document_coverage(
                    selected_shards=selected_shards,
                    relevant_docs=relevant_docs,
                    shards=shards,
                )
            )

            for k_doc in document_recall_k_values:
                doc_recalls_by_k[k_doc].append(
                    document_recall_at_k_within_selected_shards(
                        query_embedding=query_embedding,
                        selected_shards=selected_shards,
                        relevant_docs=relevant_docs,
                        shards=shards,
                        doc_id_to_index=doc_id_to_index,
                        doc_embeddings=doc_embeddings,
                        k=k_doc,
                        batch_size=document_recall_batch_size,
                    )
                )

            if selected_groups is not None:
                assert shard_to_group is not None
                group_recalls.append(
                    group_recall_at_g(
                        selected_groups=selected_groups,
                        relevant_shards=relevant_shards,
                        shard_to_group=shard_to_group,
                    )
                )
                group_drop_errors.append(
                    float(
                        is_group_drop_error(
                            selected_groups=selected_groups,
                            relevant_shards=relevant_shards,
                            shard_to_group=shard_to_group,
                        )
                    )
                )

        row = {
            "dataset": dataset_name,
            "shard_type": shard_type,
            "method": method,
            "num_shards": num_shards,
            "num_groups": num_groups,
            "num_selected_groups": num_selected_groups,
            "centroids_per_shard": centroids_per_shard,
            "top_b": top_b,
            "mean_shard_recall": float(np.mean(shard_recalls)) if shard_recalls else 0.0,
            "mean_oracle_doc_coverage": float(np.mean(oracle_coverages)) if oracle_coverages else 0.0,
            "mean_selected_shard_count": float(np.mean(selected_shard_counts)) if selected_shard_counts else 0.0,
            "accessed_shard_ratio": (float(np.mean(selected_shard_counts)) / num_shards) if selected_shard_counts else 0.0,
            "accessed_group_ratio": (num_selected_groups / num_groups) if num_groups else None,
            "centroid_scoring_cost": centroid_scoring_cost(
                method=method,
                num_shards=num_shards,
                centroids_per_shard=centroids_per_shard,
                num_groups=num_groups,
                num_selected_groups=num_selected_groups,
                groups=groups,
            ),
            "num_queries": len(shard_recalls),
            "seed": seed,
        }

        for k_doc, values in doc_recalls_by_k.items():
            row[f"mean_doc_recall_at_{k_doc}_within_selected_shards"] = (
                float(np.mean(values)) if values else 0.0
            )

        if group_recalls:
            row["mean_group_recall"] = float(np.mean(group_recalls))
            row["group_drop_error_rate"] = float(np.mean(group_drop_errors))
        else:
            row["mean_group_recall"] = None
            row["group_drop_error_rate"] = None

        results.append(row)
        print(
            f"[RESULT] dataset={dataset_name}, shard_type={shard_type}, method={method}, "
            f"k={centroids_per_shard}, top-B={top_b}, "
            f"ShardRecall={row['mean_shard_recall']:.4f}, "
            f"OracleCoverage={row['mean_oracle_doc_coverage']:.4f}, "
            f"GroupDrop={row['group_drop_error_rate']}"
        )

    return results


def run_ablation(config_path: str | Path) -> None:
    cfg = load_config(config_path)

    dataset_name = cfg["dataset"]["name"]
    raw_data_dir = cfg["dataset"].get("raw_data_dir", "data/raw/beir")
    embedding_dir = Path(cfg["dataset"]["embedding_dir"])

    result_dir = Path(cfg["output"]["result_dir"])
    filename_prefix = cfg["output"].get("filename_prefix", f"{dataset_name}_ablation")

    seed = int(cfg["experiment"].get("seed", 42))
    num_shard_values = list(cfg["experiment"]["num_shards"])
    shard_types = list(cfg["experiment"]["shard_types"])
    top_b_values = list(cfg["experiment"]["top_b_values"])
    centroid_k_values = list(cfg["experiment"].get("centroid_k_values", [2, 4, 8]))
    embedding_batch_size = int(cfg["experiment"].get("embedding_batch_size", 65536))
    multi_centroid_batch_size = int(cfg["experiment"].get("multi_centroid_batch_size", 8192))

    grouping_cfg = cfg["experiment"].get("grouping", {})
    grouping_enabled = bool(grouping_cfg.get("enabled", False))
    num_group_values = list(grouping_cfg.get("num_groups", []))
    num_selected_group_values = list(grouping_cfg.get("num_selected_groups", []))

    document_recall_k_values = list(cfg.get("metrics", {}).get("document_recall_k_values", []))
    document_recall_batch_size = int(cfg.get("metrics", {}).get("document_recall_batch_size", 65536))

    print(f"[INFO] Loading dataset: {dataset_name}")
    corpus, queries, qrels = load_beir_dataset(dataset_name, raw_data_dir)
    qrels = normalize_qrels(qrels)
    print(f"[INFO] corpus={len(corpus)}, queries={len(queries)}, qrels={len(qrels)}")

    print(f"[INFO] Loading embeddings from {embedding_dir}")
    bundle = load_embedding_bundle(embedding_dir)
    doc_embeddings = bundle.doc_embeddings
    query_embeddings = bundle.query_embeddings
    doc_ids = [str(x) for x in bundle.doc_ids]
    query_ids = [str(x) for x in bundle.query_ids]

    print(f"[INFO] doc_embeddings shape: {doc_embeddings.shape}")
    print(f"[INFO] query_embeddings shape: {query_embeddings.shape}")
    print(f"[INFO] num doc_ids: {len(doc_ids)}")
    print(f"[INFO] num query_ids: {len(query_ids)}")

    if len(doc_ids) != doc_embeddings.shape[0]:
        raise ValueError("doc_ids/doc_embeddings mismatch")
    if len(query_ids) != query_embeddings.shape[0]:
        raise ValueError("query_ids/query_embeddings mismatch")

    all_results: list[dict] = []

    for num_shards in num_shard_values:
        for shard_type in shard_types:
            print("=" * 80)
            print(f"[INFO] Running shard_type={shard_type}, num_shards={num_shards}")

            shards = build_shards_streaming(
                shard_type=shard_type,
                doc_ids=doc_ids,
                doc_embeddings=doc_embeddings,
                num_shards=num_shards,
                seed=seed,
                batch_size=embedding_batch_size,
            )
            print_shard_size_summary(shards)

            doc_to_shard = build_doc_to_shard(shards)
            relevant_shards_by_query = build_relevant_shards(qrels=qrels, doc_to_shard=doc_to_shard)
            print(f"[INFO] num queries with relevant shards: {len(relevant_shards_by_query)}")

            shard_centroids = compute_single_centroids_streaming(
                shards=shards,
                doc_ids=doc_ids,
                doc_embeddings=doc_embeddings,
                batch_size=embedding_batch_size,
            )

            all_results.extend(
                evaluate_ranked_results(
                    dataset_name=dataset_name,
                    shard_type=shard_type,
                    method="single_centroid",
                    num_shards=num_shards,
                    top_b_values=top_b_values,
                    seed=seed,
                    query_embeddings=query_embeddings,
                    query_ids=query_ids,
                    qrels=qrels,
                    relevant_shards_by_query=relevant_shards_by_query,
                    shards=shards,
                    doc_ids=doc_ids,
                    doc_embeddings=doc_embeddings,
                    rank_fn=lambda q, sc=shard_centroids: rank_shards_single_centroid(q, sc),
                    centroids_per_shard=1,
                    document_recall_k_values=document_recall_k_values,
                    document_recall_batch_size=document_recall_batch_size,
                )
            )

            multi_centroids_by_k = {}
            for k in centroid_k_values:
                print(f"[INFO] Computing multi-centroids: k={k}")
                multi_centroids_by_k[k] = compute_multi_centroids_memory_safe(
                    shards=shards,
                    doc_ids=doc_ids,
                    doc_embeddings=doc_embeddings,
                    k=k,
                    seed=seed,
                    minibatch_size=multi_centroid_batch_size,
                )

                all_results.extend(
                    evaluate_ranked_results(
                        dataset_name=dataset_name,
                        shard_type=shard_type,
                        method="multi_centroid",
                        num_shards=num_shards,
                        top_b_values=top_b_values,
                        seed=seed,
                        query_embeddings=query_embeddings,
                        query_ids=query_ids,
                        qrels=qrels,
                        relevant_shards_by_query=relevant_shards_by_query,
                        shards=shards,
                        doc_ids=doc_ids,
                        doc_embeddings=doc_embeddings,
                        rank_fn=lambda q, smc=multi_centroids_by_k[k]: rank_shards_multi_centroid(q, smc),
                        centroids_per_shard=k,
                        document_recall_k_values=document_recall_k_values,
                        document_recall_batch_size=document_recall_batch_size,
                    )
                )

            if grouping_enabled:
                for num_groups in num_group_values:
                    if num_groups >= num_shards:
                        print(f"[WARN] Skipping num_groups={num_groups} because num_groups >= num_shards")
                        continue

                    print(f"[INFO] Building groups: num_groups={num_groups}")
                    groups, shard_to_group, group_centroids = build_groups_from_shard_centroids(
                        shard_centroids=shard_centroids,
                        num_groups=num_groups,
                        seed=seed,
                    )
                    print_group_summary(groups)

                    for num_selected_groups in num_selected_group_values:
                        if num_selected_groups > num_groups:
                            print(
                                f"[WARN] Skipping num_selected_groups={num_selected_groups} "
                                f"because it is larger than num_groups={num_groups}"
                            )
                            continue

                        all_results.extend(
                            evaluate_ranked_results(
                                dataset_name=dataset_name,
                                shard_type=shard_type,
                                method="group_single_centroid",
                                num_shards=num_shards,
                                top_b_values=top_b_values,
                                seed=seed,
                                query_embeddings=query_embeddings,
                                query_ids=query_ids,
                                qrels=qrels,
                                relevant_shards_by_query=relevant_shards_by_query,
                                shards=shards,
                                doc_ids=doc_ids,
                                doc_embeddings=doc_embeddings,
                                rank_fn=lambda q, ng=num_selected_groups: rank_shards_group_single_centroid(
                                    query_embedding=q,
                                    group_centroids=group_centroids,
                                    groups=groups,
                                    shard_centroids=shard_centroids,
                                    num_selected_groups=ng,
                                ),
                                centroids_per_shard=1,
                                num_groups=num_groups,
                                num_selected_groups=num_selected_groups,
                                groups=groups,
                                shard_to_group=shard_to_group,
                                document_recall_k_values=document_recall_k_values,
                                document_recall_batch_size=document_recall_batch_size,
                            )
                        )

                        for k in centroid_k_values:
                            all_results.extend(
                                evaluate_ranked_results(
                                    dataset_name=dataset_name,
                                    shard_type=shard_type,
                                    method="group_multi_centroid",
                                    num_shards=num_shards,
                                    top_b_values=top_b_values,
                                    seed=seed,
                                    query_embeddings=query_embeddings,
                                    query_ids=query_ids,
                                    qrels=qrels,
                                    relevant_shards_by_query=relevant_shards_by_query,
                                    shards=shards,
                                    doc_ids=doc_ids,
                                    doc_embeddings=doc_embeddings,
                                    rank_fn=lambda q, ng=num_selected_groups, smc=multi_centroids_by_k[k]: rank_shards_group_multi_centroid(
                                        query_embedding=q,
                                        group_centroids=group_centroids,
                                        groups=groups,
                                        shard_multi_centroids=smc,
                                        num_selected_groups=ng,
                                    ),
                                    centroids_per_shard=k,
                                    num_groups=num_groups,
                                    num_selected_groups=num_selected_groups,
                                    groups=groups,
                                    shard_to_group=shard_to_group,
                                    document_recall_k_values=document_recall_k_values,
                                    document_recall_batch_size=document_recall_batch_size,
                                )
                            )

            partial_output = result_dir / f"{filename_prefix}_partial.csv"
            save_results_csv(all_results, partial_output)

    output_path = result_dir / f"{filename_prefix}_results.csv"
    save_results_csv(all_results, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to ablation yaml config")
    args = parser.parse_args()
    run_ablation(args.config)
