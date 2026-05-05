from pathlib import Path

import numpy as np

from src.data.io_utils import load_json, load_numpy
from src.data.load_beir import load_beir_dataset
from src.evaluation.relevant_shard import build_relevant_shards
from src.evaluation.result_writer import save_results_csv
from src.evaluation.shard_recall import shard_recall_at_b
from src.ranking.multi_centroid_ranker import rank_shards_multi_centroid
from src.ranking.single_centroid_ranker import rank_shards_single_centroid
from src.representation.multi_centroid import compute_multi_centroids
from src.representation.single_centroid import compute_single_centroids
from src.sharding.heterogeneous_shard import make_heterogeneous_shards
from src.sharding.random_shard import make_random_shards
from src.sharding.topic_shard import make_topic_based_shards


def build_doc_to_shard(shards: dict[int, list[str]]) -> dict[str, int]:
    """
    doc_id -> shard_id mapping 생성.
    shard type과 무관하게 공통으로 사용한다.
    """
    doc_to_shard = {}

    for shard_id, shard_doc_ids in shards.items():
        for doc_id in shard_doc_ids:
            doc_to_shard[doc_id] = shard_id

    return doc_to_shard


def build_shards(
    shard_type: str,
    doc_ids: list[str],
    doc_embeddings: np.ndarray,
    num_shards: int,
    seed: int,
) -> dict[int, list[str]]:
    """
    shard_type에 따라 shard를 생성한다.
    """
    if shard_type == "random":
        return make_random_shards(
            doc_ids=doc_ids,
            num_shards=num_shards,
            seed=seed,
        )

    if shard_type == "topic_based":
        return make_topic_based_shards(
            doc_ids=doc_ids,
            doc_embeddings=doc_embeddings,
            num_shards=num_shards,
            seed=seed,
        )

    if shard_type == "heterogeneous":
        return make_heterogeneous_shards(
            doc_ids=doc_ids,
            doc_embeddings=doc_embeddings,
            num_shards=num_shards,
            num_topics=num_shards,
            seed=seed,
        )

    raise ValueError(f"Unknown shard_type: {shard_type}")


def evaluate_single_centroid(
    dataset_name: str,
    shard_type: str,
    num_shards: int,
    top_b_values: list[int],
    seed: int,
    query_embeddings: np.ndarray,
    query_ids: list[str],
    relevant_shards_by_query: dict[str, set[int]],
    shard_centroids: dict[int, np.ndarray],
) -> list[dict]:
    """
    Single-Centroid ranking 평가.
    """
    query_id_to_index = {query_id: idx for idx, query_id in enumerate(query_ids)}
    results = []

    print("[INFO] Running Single-Centroid ranking and Shard Recall@B...")

    for top_b in top_b_values:
        recalls = []

        for query_id, relevant_shards in relevant_shards_by_query.items():
            if query_id not in query_id_to_index:
                continue

            query_idx = query_id_to_index[query_id]
            query_embedding = query_embeddings[query_idx]

            ranked = rank_shards_single_centroid(
                query_embedding=query_embedding,
                shard_centroids=shard_centroids,
            )

            selected_shards = [shard_id for shard_id, _score in ranked[:top_b]]

            recall = shard_recall_at_b(
                selected_shards=selected_shards,
                relevant_shards=relevant_shards,
            )

            recalls.append(recall)

        mean_recall = float(np.mean(recalls)) if recalls else 0.0

        row = {
            "dataset": dataset_name,
            "shard_type": shard_type,
            "method": "single_centroid",
            "num_shards": num_shards,
            "num_groups": None,
            "centroids_per_shard": 1,
            "top_b": top_b,
            "mean_shard_recall": mean_recall,
            "accessed_shard_ratio": top_b / num_shards,
            "num_queries": len(recalls),
            "seed": seed,
        }

        results.append(row)

        print(
            f"[RESULT] shard_type={shard_type}, method=single_centroid, "
            f"top-B={top_b}, mean Shard Recall@B={mean_recall:.4f}, "
            f"num_queries={len(recalls)}"
        )

    return results


def evaluate_multi_centroid(
    dataset_name: str,
    shard_type: str,
    num_shards: int,
    top_b_values: list[int],
    centroid_k_values: list[int],
    seed: int,
    query_embeddings: np.ndarray,
    query_ids: list[str],
    relevant_shards_by_query: dict[str, set[int]],
    shards: dict[int, list[str]],
    doc_ids: list[str],
    doc_embeddings: np.ndarray,
) -> list[dict]:
    """
    Multi-Centroid ranking 평가.
    """
    query_id_to_index = {query_id: idx for idx, query_id in enumerate(query_ids)}
    results = []

    print("[INFO] Running Multi-Centroid ranking and Shard Recall@B...")

    for k in centroid_k_values:
        print(f"[INFO] Computing multi centroids: k={k}")

        shard_multi_centroids = compute_multi_centroids(
            shards=shards,
            doc_ids=doc_ids,
            doc_embeddings=doc_embeddings,
            k=k,
            seed=seed,
        )

        for top_b in top_b_values:
            recalls = []

            for query_id, relevant_shards in relevant_shards_by_query.items():
                if query_id not in query_id_to_index:
                    continue

                query_idx = query_id_to_index[query_id]
                query_embedding = query_embeddings[query_idx]

                ranked = rank_shards_multi_centroid(
                    query_embedding=query_embedding,
                    shard_multi_centroids=shard_multi_centroids,
                )

                selected_shards = [shard_id for shard_id, _score in ranked[:top_b]]

                recall = shard_recall_at_b(
                    selected_shards=selected_shards,
                    relevant_shards=relevant_shards,
                )

                recalls.append(recall)

            mean_recall = float(np.mean(recalls)) if recalls else 0.0

            row = {
                "dataset": dataset_name,
                "shard_type": shard_type,
                "method": "multi_centroid",
                "num_shards": num_shards,
                "num_groups": None,
                "centroids_per_shard": k,
                "top_b": top_b,
                "mean_shard_recall": mean_recall,
                "accessed_shard_ratio": top_b / num_shards,
                "num_queries": len(recalls),
                "seed": seed,
            }

            results.append(row)

            print(
                f"[RESULT] shard_type={shard_type}, method=multi_centroid, "
                f"k={k}, top-B={top_b}, "
                f"mean Shard Recall@B={mean_recall:.4f}, "
                f"num_queries={len(recalls)}"
            )

    return results


def print_shard_size_summary(shards: dict[int, list[str]]) -> None:
    sizes = [len(docs) for docs in shards.values()]

    print("[INFO] Shard size summary")
    print(f"  num shards: {len(sizes)}")
    print(f"  min size: {min(sizes)}")
    print(f"  max size: {max(sizes)}")
    print(f"  mean size: {float(np.mean(sizes)):.2f}")


def run_scifact_pilot():
    dataset_name = "scifact"

    raw_data_dir = "data/raw/beir"
    embedding_dir = Path("data/embeddings/scifact")
    result_dir = Path("results/pilot/scifact")

    num_shards = 32
    top_b_values = [1, 3, 5]
    centroid_k_values = [2, 4, 8]
    seed = 42

    shard_type = "heterogeneous"

    print("[INFO] Loading SciFact dataset...")
    corpus, queries, qrels = load_beir_dataset(dataset_name, raw_data_dir)

    print("[INFO] Loading embeddings...")
    doc_embeddings = load_numpy(embedding_dir / "doc_embeddings.npy")
    query_embeddings = load_numpy(embedding_dir / "query_embeddings.npy")
    doc_ids = load_json(embedding_dir / "doc_ids.json")
    query_ids = load_json(embedding_dir / "query_ids.json")

    print(f"[INFO] doc_embeddings shape: {doc_embeddings.shape}")
    print(f"[INFO] query_embeddings shape: {query_embeddings.shape}")
    print(f"[INFO] num doc_ids: {len(doc_ids)}")
    print(f"[INFO] num query_ids: {len(query_ids)}")

    print(f"[INFO] Building shards: shard_type={shard_type}")
    shards = build_shards(
        shard_type=shard_type,
        doc_ids=doc_ids,
        doc_embeddings=doc_embeddings,
        num_shards=num_shards,
        seed=seed,
    )

    print_shard_size_summary(shards)

    doc_to_shard = build_doc_to_shard(shards)

    print("[INFO] Building relevant shard mapping...")
    relevant_shards_by_query = build_relevant_shards(
        qrels=qrels,
        doc_to_shard=doc_to_shard,
    )

    print(f"[INFO] num queries with relevant shards: {len(relevant_shards_by_query)}")

    print("[INFO] Computing single centroids...")
    shard_centroids = compute_single_centroids(
        shards=shards,
        doc_ids=doc_ids,
        doc_embeddings=doc_embeddings,
    )

    results = []

    single_results = evaluate_single_centroid(
        dataset_name=dataset_name,
        shard_type=shard_type,
        num_shards=num_shards,
        top_b_values=top_b_values,
        seed=seed,
        query_embeddings=query_embeddings,
        query_ids=query_ids,
        relevant_shards_by_query=relevant_shards_by_query,
        shard_centroids=shard_centroids,
    )
    results.extend(single_results)

    multi_results = evaluate_multi_centroid(
        dataset_name=dataset_name,
        shard_type=shard_type,
        num_shards=num_shards,
        top_b_values=top_b_values,
        centroid_k_values=centroid_k_values,
        seed=seed,
        query_embeddings=query_embeddings,
        query_ids=query_ids,
        relevant_shards_by_query=relevant_shards_by_query,
        shards=shards,
        doc_ids=doc_ids,
        doc_embeddings=doc_embeddings,
    )
    results.extend(multi_results)

    output_path = result_dir / f"scifact_{shard_type}_single_multi_results.csv"
    save_results_csv(results, output_path)


if __name__ == "__main__":
    run_scifact_pilot()