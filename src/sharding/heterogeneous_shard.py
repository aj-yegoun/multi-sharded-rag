import random
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans


def make_heterogeneous_shards(
    doc_ids: list[str],
    doc_embeddings: np.ndarray,
    num_shards: int = 32,
    num_topics: int | None = None,
    seed: int = 42,
) -> dict[int, list[str]]:
    """
    Heterogeneous shard를 구성한다.

    먼저 문서를 topic cluster로 나눈 뒤,
    각 topic cluster의 문서를 여러 shard에 분산시킨다.
    따라서 하나의 shard 안에 다양한 topic의 문서가 섞이게 된다.

    Parameters
    ----------
    doc_ids:
        문서 id 목록. doc_embeddings의 row 순서와 일치해야 한다.
    doc_embeddings:
        문서 embedding matrix, shape = (num_docs, dim)
    num_shards:
        생성할 shard 수
    num_topics:
        문서를 먼저 나눌 topic cluster 수.
        None이면 num_shards와 동일하게 둔다.
    seed:
        KMeans 및 shuffle에 사용할 seed

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

    if num_topics is None:
        num_topics = num_shards

    if num_topics <= 0:
        raise ValueError("num_topics must be positive.")

    if num_topics > len(doc_ids):
        raise ValueError(
            f"num_topics({num_topics}) cannot be larger than num_docs({len(doc_ids)})."
        )

    print(
        f"[INFO] Building heterogeneous shards: "
        f"num_shards={num_shards}, num_topics={num_topics}"
    )

    kmeans = KMeans(
        n_clusters=num_topics,
        random_state=seed,
        n_init=10,
    )

    labels = kmeans.fit_predict(doc_embeddings)

    topic_to_docs = defaultdict(list)

    for doc_id, label in zip(doc_ids, labels):
        topic_to_docs[int(label)].append(doc_id)

    rng = random.Random(seed)

    shards = defaultdict(list)

    # 각 topic의 문서를 여러 shard에 분산
    for topic_id, topic_doc_ids in topic_to_docs.items():
        shuffled_docs = list(topic_doc_ids)
        rng.shuffle(shuffled_docs)

        # topic마다 시작 shard를 다르게 해서 특정 shard에 몰리지 않게 함
        start_offset = topic_id % num_shards

        for idx, doc_id in enumerate(shuffled_docs):
            shard_id = (start_offset + idx) % num_shards
            shards[shard_id].append(doc_id)

    # 모든 shard id가 존재하도록 보정
    shards = {shard_id: shards[shard_id] for shard_id in range(num_shards)}

    empty_shards = [shard_id for shard_id, docs in shards.items() if len(docs) == 0]
    if empty_shards:
        raise RuntimeError(f"Some heterogeneous shards are empty: {empty_shards}")

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