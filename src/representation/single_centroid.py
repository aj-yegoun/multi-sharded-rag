import numpy as np


def compute_single_centroids(
    shards: dict[int, list[str]],
    doc_ids: list[str],
    doc_embeddings: np.ndarray,
) -> dict[int, np.ndarray]:
    """
    shard별 Single-Centroid를 계산한다.
    """
    doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

    shard_centroids = {}

    for shard_id, shard_doc_ids in shards.items():
        indices = [doc_id_to_index[doc_id] for doc_id in shard_doc_ids]
        shard_matrix = doc_embeddings[indices]

        centroid = shard_matrix.mean(axis=0)

        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        shard_centroids[shard_id] = centroid.astype(np.float32)

    return shard_centroids