def shard_recall_at_b(
    selected_shards: list[int],
    relevant_shards: set[int],
) -> float:
    """
    Shard Recall@B = |selected_shards ∩ relevant_shards| / |relevant_shards|
    """
    if not relevant_shards:
        return 0.0

    selected_set = set(selected_shards)
    hit_count = len(selected_set & relevant_shards)

    return hit_count / len(relevant_shards)