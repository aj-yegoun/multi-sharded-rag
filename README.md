# multi-sharded-rag
Optimizing the Number of Accessed Shards through Effective Shard Ranking in Multi-Sharded RAG

## 1.
연구 목적본 프로젝트는 Multi-Sharded RAG 환경에서 shard ranking 방식에 따른 relevant shard 보존성을 비교하기 위한 실험 코드이다.현재 연구의 핵심 질문은 다음과 같다.> 동일한 개수의 shard만 접근할 때, Multi-Centroid representation은 Single-Centroid representation보다 정답 문서를 포함한 shard를 더 자주 후보군 안에 남기는가?파일럿 실험에서는 먼저 SciFact 데이터셋을 사용하여 전체 실험 파이프라인이 정상적으로 동작하는지 검증한다.초기 파일럿의 목표는 다음과 같다.```textSciFact 로드→ document/query embedding 생성→ random shard 구성→ Single-Centroid 계산→ query별 shard ranking→ Shard Recall@B 계산→ 결과 CSV 저장

## 2. 실험 환경
본 프로젝트는 Python 기반으로 작성한다.
주요 라이브러리는 다음과 같다.
| Library               | Role                                  |
| --------------------- | ------------------------------------- |
| numpy                 | embedding matrix 및 vector 연산        |
| pandas                | 실험 결과 CSV 저장 및 분석               |
| scikit-learn          | KMeans, cosine similarity, clustering |
| scipy                 | 거리 계산 및 sparse matrix 관련 보조     |
| torch                 | sentence-transformers backend         |
| sentence-transformers | document/query embedding 생성          |
| beir                  | BEIR benchmark 데이터셋 로드            |
| tqdm                  | 진행률 표시                             |
| pyyaml                | YAML config 로드                       |
| matplotlib            | 결과 시각화                             |
| jupyter, ipykernel    | notebook 기반 결과 확인                 |

설치 방법:
pip install -r requirements.txt

## 3. 디렉토리 구조
rag-shard-ranking/
├─ README.md
├─ requirements.txt
├─ .gitignore
│
├─ configs/
│  ├─ pilot/
│  │  └─ scifact_random_single.yaml
│  └─ full/
│     ├─ scifact_ablation.yaml
│     ├─ nfcorpus_ablation.yaml
│     └─ beir_ablation.yaml
│
├─ data/
│  ├─ raw/
│  │  └─ beir/
│  ├─ processed/
│  │  ├─ scifact/
│  │  └─ nfcorpus/
│  ├─ embeddings/
│  │  ├─ scifact/
│  │  └─ nfcorpus/
│  └─ shards/
│     ├─ scifact/
│     └─ nfcorpus/
│
├─ results/
│  ├─ pilot/
│  │  └─ scifact/
│  ├─ full/
│  │  ├─ scifact/
│  │  └─ nfcorpus/
│  └─ figures/
│
├─ logs/
│  ├─ pilot/
│  └─ full/
│
├─ scripts/
│  ├─ run_pilot_scifact.ps1
│  ├─ run_pilot_scifact.sh
│  ├─ run_full_scifact.ps1
│  └─ run_full_scifact.sh
│
├─ notebooks/
│  ├─ inspect_dataset.ipynb
│  ├─ inspect_embeddings.ipynb
│  └─ analyze_results.ipynb
│
├─ src/
│  ├─ __init__.py
│  │
│  ├─ data/
│  │  ├─ __init__.py
│  │  ├─ load_beir.py
│  │  ├─ inspect_dataset.py
│  │  └─ io_utils.py
│  │
│  ├─ embedding/
│  │  ├─ __init__.py
│  │  ├─ build_embeddings.py
│  │  └─ embedding_store.py
│  │
│  ├─ sharding/
│  │  ├─ __init__.py
│  │  ├─ random_shard.py
│  │  ├─ topic_shard.py
│  │  ├─ heterogeneous_shard.py
│  │  └─ shard_store.py
│  │
│  ├─ representation/
│  │  ├─ __init__.py
│  │  ├─ single_centroid.py
│  │  ├─ multi_centroid.py
│  │  └─ centroid_store.py
│  │
│  ├─ grouping/
│  │  ├─ __init__.py
│  │  ├─ build_groups.py
│  │  └─ group_store.py
│  │
│  ├─ ranking/
│  │  ├─ __init__.py
│  │  ├─ single_centroid_ranker.py
│  │  ├─ multi_centroid_ranker.py
│  │  ├─ group_single_ranker.py
│  │  └─ group_multi_ranker.py
│  │
│  ├─ evaluation/
│  │  ├─ __init__.py
│  │  ├─ relevant_shard.py
│  │  ├─ shard_recall.py
│  │  ├─ group_recall.py
│  │  └─ result_writer.py
│  │
│  ├─ experiment/
│  │  ├─ __init__.py
│  │  ├─ config.py
│  │  ├─ runner.py
│  │  └─ run_pilot.py
│  │
│  └─ utils/
│     ├─ __init__.py
│     ├─ seed.py
│     ├─ logging_utils.py
│     └─ path_utils.py
│
└─ tests/
   ├─ test_shard.py
   ├─ test_centroid.py
   └─ test_metrics.py

## 4. 주요 디렉토리 설명
### configs/
실험 설정 파일을 저장한다.
파일럿 실험 설정은 configs/pilot/에 저장하고, 본 실험 설정은 configs/full/에 저장한다.

### data/
원본 데이터, 전처리 데이터, embedding, shard 구성 결과를 저장한다.
- data/raw/beir/
BEIR 원본 데이터셋 저장 위치이다.
- data/processed/
전처리된 corpus, query, qrels 등의 저장 위치이다.
- data/embeddings/
문서 embedding, query embedding, id mapping 저장 위치이다.
- data/shards/
shard partition 결과 저장 위치이다.

### results/
실험 결과 CSV를 저장한다.
파일럿 결과는 results/pilot/ 아래에 저장하고, 본 실험 결과는 results/full/ 아래에 저장한다.

### logs/
실험 실행 로그를 저장한다.

### scripts/
반복 실행용 shell script 또는 PowerShell script를 저장한다.

#### notebooks/
데이터 확인, embedding 확인, 결과 분석용 notebook을 저장한다.
핵심 실험 코드는 notebook이 아니라 src/ 아래 Python module로 관리한다.

### src/
실험 코드 본체이다.
기능별로 다음과 같이 분리한다.
| Directory             | Role                                         |
| --------------------- | -------------------------------------------  |
| `src/data/`           | BEIR 데이터셋 로드, 저장/로드 유틸               |
| `src/embedding/`      | document/query embedding 생성 및 저장          |
| `src/sharding/`       | random, topic-based, heterogeneous shard 구성 |
| `src/representation/` | Single-Centroid, Multi-Centroid 계산          |
| `src/grouping/`       | shard grouping 구성                           |
| `src/ranking/`        | 각 setting별 shard ranking                    |
| `src/evaluation/`     | Shard Recall@B, Group Recall 등 metric 계산   |
| `src/experiment/`     | 전체 실험 실행                                 |
| `src/utils/`          | seed, logging, path 관련 유틸                  |


### tests/
핵심 함수의 동작을 확인하는 테스트 코드를 저장한다.

## 5. 현재 파일럿 실험 범위
현재 파일럿 실험에서는 전체 4가지 setting 중 가장 기본이 되는 Single-Centroid baseline을 먼저 구현한다.
초기 파일럿 범위:
- Dataset: SciFact
- Shard type: random
- Num shards: 32
- Representation: Single-Centroid
- Ranking score: cosine similarity
- top-B: 1, 3, 5
- Metric: Shard Recall@B
이후 확장 순서는 다음과 같다.
1. Single-Centroid
2. Multi-Centroid
3. Grouping + Single-Centroid 
4. Grouping + Multi-Centroid
5. topic-based shard
6. heterogeneous shard
7. Group-level recall analysis

## 6. 실행 방법
### 6.1 SciFact 데이터셋 로드 확인
python -m src.data.load_beir
정상 실행 시 다음 정보가 출력된다.
Dataset: scifactNum corpus docs: 5183Num queries: 300Num qrels queries: 300

### 6.2 SciFact embedding 생성
python -m src.embedding.build_embeddings
정상 실행 시 다음 파일들이 생성된다.
data/embeddings/scifact/doc_embeddings.npy
data/embeddings/scifact/query_embeddings.npy
data/embeddings/scifact/doc_ids.json
data/embeddings/scifact/query_ids.json

## 7. 주의 사항
embedding matrix와 id mapping의 순서는 반드시 보존되어야 한다.
즉, 다음 관계가 항상 유지되어야 한다.

> doc_embeddings[i]   == doc_ids[i]에 해당하는 문서 embedding
> 
> query_embeddings[j] == query_ids[j]에 해당하는 query embedding

이 mapping이 깨지면 shard 구성, centroid 계산, recall 계산 결과가 모두 잘못된다.
또한 data/, results/, logs/ 디렉토리는 용량이 커질 수 있으므로 Git에 올리지 않는다.
