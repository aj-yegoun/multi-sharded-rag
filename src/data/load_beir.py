from pathlib import Path

from beir import util
from beir.datasets.data_loader import GenericDataLoader


def load_beir_dataset(dataset_name: str, raw_data_dir: str):
    """
    BEIR 데이터셋을 다운로드/로드한다.

    Parameters
    ----------
    dataset_name : str
        예: "scifact", "nfcorpus"
    raw_data_dir : str
        BEIR 데이터셋을 저장할 상위 디렉토리

    Returns
    -------
    corpus : dict
        doc_id -> {"title": ..., "text": ...}
    queries : dict
        query_id -> query text
    qrels : dict
        query_id -> {doc_id: relevance}
    """
    raw_data_dir = Path(raw_data_dir)
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = raw_data_dir / dataset_name

    if not dataset_dir.exists():
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        print(f"[INFO] Downloading {dataset_name} from {url}")
        data_path = util.download_and_unzip(url, str(raw_data_dir))
    else:
        print(f"[INFO] Using existing dataset directory: {dataset_dir}")
        data_path = str(dataset_dir)

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    return corpus, queries, qrels


def print_dataset_summary(dataset_name: str, corpus: dict, queries: dict, qrels: dict):
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Num corpus docs: {len(corpus)}")
    print(f"Num queries: {len(queries)}")
    print(f"Num qrels queries: {len(qrels)}")

    example_doc_id = next(iter(corpus.keys()))
    example_query_id = next(iter(queries.keys()))

    print("-" * 60)
    print(f"Example doc id: {example_doc_id}")
    print(f"Example doc: {corpus[example_doc_id]}")
    print("-" * 60)
    print(f"Example query id: {example_query_id}")
    print(f"Example query: {queries[example_query_id]}")
    print("-" * 60)
    print(f"Example qrels for query {example_query_id}:")
    print(qrels.get(example_query_id, {}))
    print("=" * 60)


if __name__ == "__main__":
    dataset_name = "scifact"
    raw_data_dir = "data/raw/beir"

    corpus, queries, qrels = load_beir_dataset(dataset_name, raw_data_dir)
    print_dataset_summary(dataset_name, corpus, queries, qrels)