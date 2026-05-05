def build_relevant_shards(
    qrels: dict,
    doc_to_shard: dict[str, int],
) -> dict[str, set[int]]:
    """
    qrels를 query_id -> relevant shard set으로 변환한다.

    qrels:
        query_id -> {doc_id: relevance}
    """
    relevant_shards = {}

    for query_id, doc_rels in qrels.items():
        shard_set = set()

        for doc_id, rel in doc_rels.items():
            if rel <= 0:
                continue

            if doc_id in doc_to_shard:
                shard_set.add(doc_to_shard[doc_id])

        if shard_set:
            relevant_shards[query_id] = shard_set

    return relevant_shards