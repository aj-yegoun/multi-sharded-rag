import numpy as np
from sklearn.cluster import KMeans


def compute_multi_centroids(
    shards: dict[int, list[str]],
    doc_ids: list[str],
    doc_embeddings: np.ndarray,
    k: int = 4,
    seed: int = 42,
) -> dict[int, np.ndarray]:
    """
    shard별 Multi-Centroid를 계산한다.

    각 shard 내부 문서 embedding을 KMeans로 clustering하고,
    각 cluster center를 local centroid로 사용한다.

    Returns
    -------
    shard_multi_centroids:
        shard_id -> np.ndarray, shape = (actual_k, dim)
    """
    doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

    shard_multi_centroids = {}

    for shard_id, shard_doc_ids in shards.items():
        indices = [doc_id_to_index[doc_id] for doc_id in shard_doc_ids]
        shard_matrix = doc_embeddings[indices]

        actual_k = min(k, len(shard_doc_ids))

        if actual_k <= 1:
            centroid = shard_matrix.mean(axis=0, keepdims=True)
            norm = np.linalg.norm(centroid, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            centroid = centroid / norm
            shard_multi_centroids[shard_id] = centroid.astype(np.float32)
            continue

        kmeans = KMeans(
            n_clusters=actual_k,
            random_state=seed,
            n_init="auto",
        )

        kmeans.fit(shard_matrix)
        centroids = kmeans.cluster_centers_

        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        centroids = centroids / norms

        shard_multi_centroids[shard_id] = centroids.astype(np.float32)

    return shard_multi_centroids