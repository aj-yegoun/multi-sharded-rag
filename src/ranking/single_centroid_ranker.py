import numpy as np


def rank_shards_single_centroid(
    query_embedding: np.ndarray,
    shard_centroids: dict[int, np.ndarray],
) -> list[tuple[int, float]]:
    """
    query와 shard centroid 간 cosine similarity로 shard ranking을 수행한다.

    Returns
    -------
    ranked:
        [(shard_id, score), ...] score 내림차순
    """
    ranked = []

    for shard_id, centroid in shard_centroids.items():
        score = float(np.dot(query_embedding, centroid))
        ranked.append((shard_id, score))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked