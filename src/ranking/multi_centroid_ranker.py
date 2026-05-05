import numpy as np


def rank_shards_multi_centroid(
    query_embedding: np.ndarray,
    shard_multi_centroids: dict[int, np.ndarray],
) -> list[tuple[int, float]]:
    """
    query와 shard 내부 local centroids 간 similarity 중 max 값을
    shard score로 사용하여 ranking한다.

    Returns
    -------
    ranked:
        [(shard_id, score), ...] score 내림차순
    """
    ranked = []

    for shard_id, centroids in shard_multi_centroids.items():
        scores = centroids @ query_embedding
        score = float(np.max(scores))
        ranked.append((shard_id, score))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked