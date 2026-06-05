import random
from collections import defaultdict


def make_random_shards(
    doc_ids: list[str],
    num_shards: int = 32,
    seed: int = 42,
) -> dict[int, list[str]]:
    """
    doc_ids를 랜덤하게 num_shards개의 shard로 분할한다.
    pilot 실험용으로, 본 실험에서는 사용을 하지 않는다.
    """
    rng = random.Random(seed)
    shuffled_doc_ids = list(doc_ids)
    rng.shuffle(shuffled_doc_ids)

    shards = defaultdict(list)

    for idx, doc_id in enumerate(shuffled_doc_ids):
        shard_id = idx % num_shards
        shards[shard_id].append(doc_id)

    return dict(shards)


def build_doc_to_shard(shards: dict[int, list[str]]) -> dict[str, int]:
    """
    doc_id -> shard_id mapping을 만든다.
    """
    doc_to_shard = {}

    for shard_id, shard_doc_ids in shards.items():
        for doc_id in shard_doc_ids:
            doc_to_shard[doc_id] = shard_id

    return doc_to_shard