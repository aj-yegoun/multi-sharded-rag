from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.data.io_utils import ensure_dir, save_json, save_numpy
from src.data.load_beir import load_beir_dataset


def build_doc_texts(corpus: dict) -> tuple[list[str], list[str]]:
    """
    BEIR corpus를 embedding 입력용 text list로 변환한다.

    Returns
    -------
    doc_ids:
        doc id 목록
    doc_texts:
        title + text 형태의 문서 문자열 목록
    """
    doc_ids = []
    doc_texts = []

    for doc_id, doc in corpus.items():
        title = doc.get("title", "") or ""
        text = doc.get("text", "") or ""

        full_text = f"{title} {text}".strip()

        doc_ids.append(doc_id)
        doc_texts.append(full_text)

    return doc_ids, doc_texts


def build_query_texts(queries: dict) -> tuple[list[str], list[str]]:
    """
    BEIR queries를 embedding 입력용 text list로 변환한다.
    """
    query_ids = []
    query_texts = []

    for query_id, query_text in queries.items():
        query_ids.append(query_id)
        query_texts.append(query_text)

    return query_ids, query_texts


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 32,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    """
    sentence-transformers 모델로 text list를 embedding matrix로 변환한다.
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )

    return embeddings.astype(np.float32)


def build_and_save_scifact_embeddings(
    raw_data_dir: str = "data/raw/beir",
    embedding_dir: str = "data/embeddings/scifact",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    normalize_embeddings: bool = True,
) -> None:
    dataset_name = "scifact"

    print("[INFO] Loading SciFact dataset...")
    corpus, queries, qrels = load_beir_dataset(dataset_name, raw_data_dir)

    print("[INFO] Building document/query text lists...")
    doc_ids, doc_texts = build_doc_texts(corpus)
    query_ids, query_texts = build_query_texts(queries)

    print(f"[INFO] Num docs: {len(doc_ids)}")
    print(f"[INFO] Num queries: {len(query_ids)}")

    print(f"[INFO] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print("[INFO] Encoding documents...")
    doc_embeddings = encode_texts(
        model=model,
        texts=doc_texts,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
    )

    print("[INFO] Encoding queries...")
    query_embeddings = encode_texts(
        model=model,
        texts=query_texts,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
    )

    embedding_dir = ensure_dir(embedding_dir)

    print("[INFO] Saving embeddings and id mappings...")
    save_numpy(doc_embeddings, embedding_dir / "doc_embeddings.npy")
    save_numpy(query_embeddings, embedding_dir / "query_embeddings.npy")
    save_json(doc_ids, embedding_dir / "doc_ids.json")
    save_json(query_ids, embedding_dir / "query_ids.json")

    print("=" * 60)
    print("[DONE] SciFact embeddings saved.")
    print(f"doc_embeddings shape: {doc_embeddings.shape}")
    print(f"query_embeddings shape: {query_embeddings.shape}")
    print(f"Saved to: {Path(embedding_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    build_and_save_scifact_embeddings()