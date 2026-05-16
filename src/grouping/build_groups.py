from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans


def build_groups_from_shard_centroids(
    shard_centroids: dict[int, np.ndarray],
    num_groups: int = 4,
    seed: int = 42,
) -> tuple[dict[int, list[int]], dict[int, int], dict[int, np.ndarray]]:
    """
    shard single-centroid를 기준으로 shard들을 group으로 묶는다.

    Parameters
    ----------
    shard_centroids:
        shard_id -> centroid vector
    num_groups:
        생성할 group 수
    seed:
        KMeans random_state

    Returns
    -------
    groups:
        group_id -> list[shard_id]
    shard_to_group:
        shard_id -> group_id
    group_centroids:
        group_id -> group centroid vector
    """
    shard_ids = sorted(shard_centroids.keys())
    centroid_matrix = np.stack([shard_centroids[shard_id] for shard_id in shard_ids])

    if num_groups <= 0:
        raise ValueError("num_groups must be positive.")

    if num_groups > len(shard_ids):
        raise ValueError(
            f"num_groups({num_groups}) cannot be larger than num_shards({len(shard_ids)})."
        )

    kmeans = KMeans(
        n_clusters=num_groups,
        random_state=seed,
        n_init=10,
    )

    labels = kmeans.fit_predict(centroid_matrix)

    groups = defaultdict(list)
    shard_to_group = {}

    for shard_id, label in zip(shard_ids, labels):
        group_id = int(label)
        groups[group_id].append(shard_id)
        shard_to_group[shard_id] = group_id

    groups = dict(groups)

    group_centroids = {}

    for group_id, group_shard_ids in groups.items():
        group_matrix = np.stack(
            [shard_centroids[shard_id] for shard_id in group_shard_ids]
        )

        centroid = group_matrix.mean(axis=0)

        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        group_centroids[group_id] = centroid.astype(np.float32)

    return groups, shard_to_group, group_centroids


def print_group_summary(groups: dict[int, list[int]]) -> None:
    sizes = [len(shard_ids) for shard_ids in groups.values()]

    print("[INFO] Group size summary")
    print(f"  num groups: {len(groups)}")
    print(f"  min size: {min(sizes)}")
    print(f"  max size: {max(sizes)}")
    print(f"  mean size: {float(np.mean(sizes)):.2f}")