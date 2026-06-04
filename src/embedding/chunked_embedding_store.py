from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from src.data.io_utils import load_json


@dataclass(frozen=True)
class EmbeddingChunk:
    path: Path
    ids_path: Path
    start: int
    end: int
    embeddings: np.ndarray
    doc_ids: list[str]


@dataclass
class EmbeddingBundle:
    doc_embeddings: "ChunkedEmbeddingStore | np.ndarray"
    doc_ids: list[str]
    query_embeddings: np.ndarray
    query_ids: list[str]


class ChunkedEmbeddingStore:
    """
    Chunked document embedding reader.

    Supported directory format:
      embedding_dir/
        doc_chunks/chunk_00000.npy
        doc_chunks/chunk_00001.npy
        doc_chunk_ids/chunk_00000.json
        doc_chunk_ids/chunk_00001.json

    This class avoids np.concatenate over all document embeddings.
    It exposes enough ndarray-like behavior for the experiment code:
      - len(store)
      - store.shape
      - store[indices]
      - store.iter_chunks()
      - store.get_by_doc_ids(doc_ids)
    """

    def __init__(self, embedding_dir: str | Path):
        self.embedding_dir = Path(embedding_dir)
        self.doc_chunks_dir = self.embedding_dir / "doc_chunks"
        self.doc_chunk_ids_dir = self.embedding_dir / "doc_chunk_ids"

        if not self.doc_chunks_dir.exists():
            raise FileNotFoundError(f"Missing doc_chunks directory: {self.doc_chunks_dir}")
        if not self.doc_chunk_ids_dir.exists():
            raise FileNotFoundError(f"Missing doc_chunk_ids directory: {self.doc_chunk_ids_dir}")

        chunk_paths = sorted(self.doc_chunks_dir.glob("chunk_*.npy"))
        if not chunk_paths:
            raise FileNotFoundError(f"No chunk_*.npy files found in {self.doc_chunks_dir}")

        self.chunks: list[EmbeddingChunk] = []
        self.doc_ids: list[str] = []
        self.doc_id_to_index: dict[str, int] = {}

        start = 0
        dim: int | None = None

        for chunk_path in chunk_paths:
            ids_path = self.doc_chunk_ids_dir / f"{chunk_path.stem}.json"
            if not ids_path.exists():
                raise FileNotFoundError(f"Missing id file for {chunk_path.name}: {ids_path}")

            embeddings = np.load(chunk_path, mmap_mode="r")
            chunk_doc_ids = [str(x) for x in load_json(ids_path)]

            if embeddings.ndim != 2:
                raise ValueError(f"Expected 2D embeddings in {chunk_path}, got {embeddings.shape}")
            if embeddings.shape[0] != len(chunk_doc_ids):
                raise ValueError(
                    f"Embedding/id length mismatch in {chunk_path.name}: "
                    f"{embeddings.shape[0]} embeddings vs {len(chunk_doc_ids)} ids"
                )

            if dim is None:
                dim = int(embeddings.shape[1])
            elif int(embeddings.shape[1]) != dim:
                raise ValueError(
                    f"Embedding dim mismatch in {chunk_path.name}: {embeddings.shape[1]} vs {dim}"
                )

            end = start + embeddings.shape[0]
            self.chunks.append(
                EmbeddingChunk(
                    path=chunk_path,
                    ids_path=ids_path,
                    start=start,
                    end=end,
                    embeddings=embeddings,
                    doc_ids=chunk_doc_ids,
                )
            )

            for local_idx, doc_id in enumerate(chunk_doc_ids):
                global_idx = start + local_idx
                if doc_id in self.doc_id_to_index:
                    raise ValueError(f"Duplicate doc_id found in embedding chunks: {doc_id}")
                self.doc_id_to_index[doc_id] = global_idx

            self.doc_ids.extend(chunk_doc_ids)
            start = end

        self._shape = (start, dim or 0)

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def dtype(self):
        return self.chunks[0].embeddings.dtype

    def __len__(self) -> int:
        return self._shape[0]

    def iter_chunks(self) -> Iterator[tuple[np.ndarray, list[str], int]]:
        """Yield (chunk_embeddings, chunk_doc_ids, global_start_index)."""
        for chunk in self.chunks:
            yield chunk.embeddings, chunk.doc_ids, chunk.start

    def __getitem__(self, indices):
        if isinstance(indices, slice):
            index_array = np.arange(*indices.indices(len(self)), dtype=np.int64)
            return self._get_by_indices(index_array)

        if isinstance(indices, (int, np.integer)):
            index_array = np.asarray([int(indices)], dtype=np.int64)
            return self._get_by_indices(index_array)[0]

        index_array = np.asarray(indices, dtype=np.int64)
        return self._get_by_indices(index_array)

    def _get_by_indices(self, index_array: np.ndarray) -> np.ndarray:
        if index_array.ndim != 1:
            raise ValueError(f"Only 1D index arrays are supported, got shape {index_array.shape}")
        if len(index_array) == 0:
            return np.empty((0, self.shape[1]), dtype=self.dtype)
        if np.any(index_array < 0) or np.any(index_array >= len(self)):
            raise IndexError("Embedding index out of range")

        output = np.empty((len(index_array), self.shape[1]), dtype=self.dtype)

        # Preserve the original requested order.
        for chunk in self.chunks:
            mask = (index_array >= chunk.start) & (index_array < chunk.end)
            if not np.any(mask):
                continue
            local_indices = index_array[mask] - chunk.start
            output[np.where(mask)[0]] = chunk.embeddings[local_indices]

        return output

    def get_by_doc_ids(self, doc_ids: Sequence[str]) -> np.ndarray:
        indices = [self.doc_id_to_index[str(doc_id)] for doc_id in doc_ids]
        return self[np.asarray(indices, dtype=np.int64)]


def _load_query_embeddings_and_ids(embedding_dir: Path) -> tuple[np.ndarray, list[str]]:
    query_embeddings_path = embedding_dir / "query_embeddings.npy"
    query_ids_path = embedding_dir / "query_ids.json"

    if not query_embeddings_path.exists():
        raise FileNotFoundError(f"Missing query embeddings: {query_embeddings_path}")
    if not query_ids_path.exists():
        raise FileNotFoundError(f"Missing query ids: {query_ids_path}")

    query_embeddings = np.load(query_embeddings_path, mmap_mode="r")
    query_ids = [str(x) for x in load_json(query_ids_path)]

    if query_embeddings.shape[0] != len(query_ids):
        raise ValueError(
            f"query_embeddings/query_ids length mismatch: "
            f"{query_embeddings.shape[0]} embeddings vs {len(query_ids)} ids"
        )

    return query_embeddings, query_ids


def load_embedding_bundle(embedding_dir: str | Path) -> EmbeddingBundle:
    """
    Load embeddings in either flat or chunked format.

    Flat format:
      doc_embeddings.npy
      doc_ids.json
      query_embeddings.npy
      query_ids.json

    Chunked format:
      doc_chunks/*.npy
      doc_chunk_ids/*.json
      query_embeddings.npy
      query_ids.json
    """
    embedding_dir = Path(embedding_dir)

    query_embeddings, query_ids = _load_query_embeddings_and_ids(embedding_dir)

    flat_doc_embeddings_path = embedding_dir / "doc_embeddings.npy"
    flat_doc_ids_path = embedding_dir / "doc_ids.json"

    if flat_doc_embeddings_path.exists() and flat_doc_ids_path.exists():
        doc_embeddings = np.load(flat_doc_embeddings_path, mmap_mode="r")
        doc_ids = [str(x) for x in load_json(flat_doc_ids_path)]

        if doc_embeddings.shape[0] != len(doc_ids):
            raise ValueError(
                f"doc_embeddings/doc_ids length mismatch: "
                f"{doc_embeddings.shape[0]} embeddings vs {len(doc_ids)} ids"
            )

        return EmbeddingBundle(
            doc_embeddings=doc_embeddings,
            doc_ids=doc_ids,
            query_embeddings=query_embeddings,
            query_ids=query_ids,
        )

    store = ChunkedEmbeddingStore(embedding_dir)
    return EmbeddingBundle(
        doc_embeddings=store,
        doc_ids=store.doc_ids,
        query_embeddings=query_embeddings,
        query_ids=query_ids,
    )
