def group_recall_at_g(
    selected_groups: list[int],
    relevant_shards: set[int],
    shard_to_group: dict[int, int],
) -> float:
    """
    relevant shard들이 속한 group 중 selected_groups에 포함된 비율을 계산한다.

    예:
        relevant_shards = {2, 5, 9}
        shard_to_group = {2: 0, 5: 1, 9: 1}
        relevant_groups = {0, 1}

        selected_groups = [1]
        group_recall = 1 / 2
    """
    if not relevant_shards:
        return 0.0

    relevant_groups = {
        shard_to_group[shard_id]
        for shard_id in relevant_shards
        if shard_id in shard_to_group
    }

    if not relevant_groups:
        return 0.0

    selected_group_set = set(selected_groups)
    hit_count = len(selected_group_set & relevant_groups)

    return hit_count / len(relevant_groups)


def is_group_drop_error(
    selected_groups: list[int],
    relevant_shards: set[int],
    shard_to_group: dict[int, int],
) -> bool:
    """
    relevant shard가 속한 group이 하나도 선택되지 않았는지 확인한다.
    """
    if not relevant_shards:
        return False

    relevant_groups = {
        shard_to_group[shard_id]
        for shard_id in relevant_shards
        if shard_id in shard_to_group
    }

    if not relevant_groups:
        return False

    selected_group_set = set(selected_groups)

    return len(selected_group_set & relevant_groups) == 0