import numpy as np


def rank_groups(
    query_embedding: np.ndarray,
    group_centroids: dict[int, np.ndarray],
) -> list[tuple[int, float]]:
    """
    query와 group centroid 간 similarity로 group ranking을 수행한다.
    """
    ranked = []

    for group_id, centroid in group_centroids.items():
        score = float(np.dot(query_embedding, centroid))
        ranked.append((group_id, score))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def rank_shards_group_single_centroid(
    query_embedding: np.ndarray,
    group_centroids: dict[int, np.ndarray],
    groups: dict[int, list[int]],
    shard_centroids: dict[int, np.ndarray],
    num_selected_groups: int = 1,
) -> tuple[list[tuple[int, float]], list[int]]:
    """
    Grouping + Single-Centroid ranking.

    Returns
    -------
    ranked_shards:
        [(shard_id, score), ...] score 내림차순
    selected_groups:
        선택된 group id 목록
    """
    ranked_groups = rank_groups(
        query_embedding=query_embedding,
        group_centroids=group_centroids,
    )

    selected_groups = [
        group_id for group_id, _score in ranked_groups[:num_selected_groups]
    ]

    candidate_shards = []

    for group_id in selected_groups:
        candidate_shards.extend(groups[group_id])

    ranked_shards = []

    for shard_id in candidate_shards:
        score = float(np.dot(query_embedding, shard_centroids[shard_id]))
        ranked_shards.append((shard_id, score))

    ranked_shards.sort(key=lambda x: x[1], reverse=True)

    return ranked_shards, selected_groups