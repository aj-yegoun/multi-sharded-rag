import numpy as np

from src.ranking.group_single_ranker import rank_groups


def rank_shards_group_multi_centroid(
    query_embedding: np.ndarray,
    group_centroids: dict[int, np.ndarray],
    groups: dict[int, list[int]],
    shard_multi_centroids: dict[int, np.ndarray],
    num_selected_groups: int = 1,
) -> tuple[list[tuple[int, float]], list[int]]:
    """
    Grouping + Multi-Centroid ranking.

    Group selection은 group centroid 기반으로 수행하고,
    선택된 group 내부에서만 Multi-Centroid shard ranking을 수행한다.

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
        centroids = shard_multi_centroids[shard_id]
        scores = centroids @ query_embedding
        score = float(np.max(scores))

        ranked_shards.append((shard_id, score))

    ranked_shards.sort(key=lambda x: x[1], reverse=True)

    return ranked_shards, selected_groups