from pathlib import Path
import json
import numpy as np


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_embeddings_from_dir(embedding_dir: str | Path):
    """
    Supports both formats:

    1. Flat format:
       doc_embeddings.npy
       doc_ids.json
       query_embeddings.npy
       query_ids.json

    2. Chunked document format:
       doc_chunks/chunk_00000.npy
       doc_chunk_ids/chunk_00000.json
       query_embeddings.npy
       query_ids.json
    """
    embedding_dir = Path(embedding_dir)

    query_embeddings_path = embedding_dir / "query_embeddings.npy"
    query_ids_path = embedding_dir / "query_ids.json"

    if not query_embeddings_path.exists():
        raise FileNotFoundError(f"Missing query embeddings: {query_embeddings_path}")
    if not query_ids_path.exists():
        raise FileNotFoundError(f"Missing query ids: {query_ids_path}")

    query_embeddings = np.load(query_embeddings_path)
    query_ids = load_json(query_ids_path)

    flat_doc_embeddings_path = embedding_dir / "doc_embeddings.npy"
    flat_doc_ids_path = embedding_dir / "doc_ids.json"

    if flat_doc_embeddings_path.exists() and flat_doc_ids_path.exists():
        doc_embeddings = np.load(flat_doc_embeddings_path)
        doc_ids = load_json(flat_doc_ids_path)

        return {
            "doc_embeddings": doc_embeddings,
            "doc_ids": doc_ids,
            "query_embeddings": query_embeddings,
            "query_ids": query_ids,
        }

    doc_chunks_dir = embedding_dir / "doc_chunks"
    doc_chunk_ids_dir = embedding_dir / "doc_chunk_ids"

    if not doc_chunks_dir.exists():
        raise FileNotFoundError(
            f"Missing both flat doc_embeddings.npy and chunked dir: {doc_chunks_dir}"
        )
    if not doc_chunk_ids_dir.exists():
        raise FileNotFoundError(f"Missing doc_chunk_ids dir: {doc_chunk_ids_dir}")

    chunk_paths = sorted(doc_chunks_dir.glob("chunk_*.npy"))

    if not chunk_paths:
        raise FileNotFoundError(f"No chunk_*.npy files found in {doc_chunks_dir}")

    doc_embedding_chunks = []
    doc_ids = []

    for chunk_path in chunk_paths:
        chunk_name = chunk_path.stem
        ids_path = doc_chunk_ids_dir / f"{chunk_name}.json"

        if not ids_path.exists():
            raise FileNotFoundError(f"Missing id file for {chunk_path.name}: {ids_path}")

        emb = np.load(chunk_path)
        ids = load_json(ids_path)

        if len(emb) != len(ids):
            raise ValueError(
                f"Embedding/id length mismatch in {chunk_name}: "
                f"{len(emb)} embeddings vs {len(ids)} ids"
            )

        doc_embedding_chunks.append(emb)
        doc_ids.extend(ids)

    doc_embeddings = np.concatenate(doc_embedding_chunks, axis=0)

    return {
        "doc_embeddings": doc_embeddings,
        "doc_ids": doc_ids,
        "query_embeddings": query_embeddings,
        "query_ids": query_ids,
    }