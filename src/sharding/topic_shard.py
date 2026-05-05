from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans


def make_topic_based_shards(
    doc_ids: list[str],
    doc_embeddings: np.ndarray,
    num_shards: int = 32,
    seed: int = 42,
) -> dict[int, list[str]]:
    """
    문서 embedding을 KMeans로 clustering하여 topic-based shard를 구성한다.

    같은 cluster에 속한 문서들은 의미적으로 유사하다고 보고,
    하나의 cluster를 하나의 shard로 사용한다.

    Parameters
    ----------
    doc_ids:
        문서 id 목록. doc_embeddings의 row 순서와 일치해야 한다.
    doc_embeddings:
        문서 embedding matrix, shape = (num_docs, dim)
    num_shards:
        생성할 shard 수
    seed:
        KMeans random_state

    Returns
    -------
    shards:
        shard_id -> list[doc_id]
    """
    if len(doc_ids) != len(doc_embeddings):
        raise ValueError(
            f"doc_ids length and doc_embeddings length mismatch: "
            f"{len(doc_ids)} vs {len(doc_embeddings)}"
        )

    if num_shards <= 0:
        raise ValueError("num_shards must be positive.")

    if num_shards > len(doc_ids):
        raise ValueError(
            f"num_shards({num_shards}) cannot be larger than num_docs({len(doc_ids)})."
        )

    print(f"[INFO] Building topic-based shards with KMeans: num_shards={num_shards}")

    kmeans = KMeans(
        n_clusters=num_shards,
        random_state=seed,
        n_init=10,
    )

    labels = kmeans.fit_predict(doc_embeddings)

    shards = defaultdict(list)

    for doc_id, label in zip(doc_ids, labels):
        shard_id = int(label)
        shards[shard_id].append(doc_id)

    shards = dict(shards)

    # 혹시 모를 empty shard 검사
    if len(shards) != num_shards:
        missing = set(range(num_shards)) - set(shards.keys())
        raise RuntimeError(f"Some topic-based shards are empty: {missing}")

    return shards


def build_doc_to_shard(shards: dict[int, list[str]]) -> dict[str, int]:
    """
    doc_id -> shard_id mapping을 만든다.
    """
    doc_to_shard = {}

    for shard_id, shard_doc_ids in shards.items():
        for doc_id in shard_doc_ids:
            doc_to_shard[doc_id] = shard_id

    return doc_to_shard