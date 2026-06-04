from __future__ import annotations

import numpy as np


def get_positive_relevant_docs(qrels: dict, query_id: str) -> set[str]:
    """
    qrels에서 relevance > 0인 relevant doc_id set을 반환한다.
    id type mismatch를 피하기 위해 query_id/doc_id는 문자열로 통일한다.
    """
    doc_rels = qrels.get(str(query_id), {})
    return {str(doc_id) for doc_id, rel in doc_rels.items() if rel > 0}


def oracle_relevant_document_coverage(
    selected_shards: list[int],
    relevant_docs: set[str],
    shards: dict[int, list[str]],
) -> float:
    if not relevant_docs:
        return 0.0

    selected_doc_ids = set()
    for shard_id in selected_shards:
        selected_doc_ids.update(str(doc_id) for doc_id in shards.get(shard_id, []))

    return len(selected_doc_ids & relevant_docs) / len(relevant_docs)


def _get_doc_matrix(doc_embeddings, doc_id_to_index: dict[str, int], doc_ids: list[str]) -> np.ndarray:
    if hasattr(doc_embeddings, "get_by_doc_ids"):
        return doc_embeddings.get_by_doc_ids(doc_ids)
    indices = [doc_id_to_index[str(doc_id)] for doc_id in doc_ids]
    return doc_embeddings[indices]


def document_recall_at_k_within_selected_shards(
    query_embedding: np.ndarray,
    selected_shards: list[int],
    relevant_docs: set[str],
    shards: dict[int, list[str]],
    doc_id_to_index: dict[str, int],
    doc_embeddings,
    k: int,
    batch_size: int = 65536,
) -> float:
    """
    Document Recall@K within Selected Shards.

    The candidate documents can be large in MS-MARCO/FEVER. Therefore this
    function scores candidates in batches instead of materializing one huge
    candidate matrix.
    """
    if not relevant_docs:
        return 0.0

    candidate_doc_ids: list[str] = []
    for shard_id in selected_shards:
        candidate_doc_ids.extend(str(doc_id) for doc_id in shards.get(shard_id, []))

    if not candidate_doc_ids:
        return 0.0

    top_k = min(k, len(candidate_doc_ids))
    if top_k <= 0:
        return 0.0

    # Keep only the current global top-k candidates while streaming batches.
    best_scores = np.empty((0,), dtype=np.float32)
    best_doc_ids: list[str] = []

    for start in range(0, len(candidate_doc_ids), batch_size):
        batch_doc_ids = candidate_doc_ids[start : start + batch_size]
        batch_matrix = _get_doc_matrix(doc_embeddings, doc_id_to_index, batch_doc_ids)
        scores = np.asarray(batch_matrix @ query_embedding, dtype=np.float32)

        merged_scores = np.concatenate([best_scores, scores])
        merged_doc_ids = best_doc_ids + batch_doc_ids

        keep = min(top_k, len(merged_doc_ids))
        if keep <= 0:
            continue

        top_indices = np.argpartition(-merged_scores, keep - 1)[:keep]
        best_scores = merged_scores[top_indices]
        best_doc_ids = [merged_doc_ids[int(i)] for i in top_indices]

    top_doc_ids = set(best_doc_ids)
    return len(top_doc_ids & relevant_docs) / len(relevant_docs)
