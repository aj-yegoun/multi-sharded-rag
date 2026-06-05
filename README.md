# Multi-Sharded RAG Shard Selection

Multi-Sharded RAG 환경에서 shard representation 방식과 grouping 구조가 relevant shard 보존 성능 및 scoring cost에 미치는 영향을 분석하는 실험 코드입니다.

본 연구는 전체 shard를 모두 검색하지 않고, 제한된 수의 shard만 접근하는 selective shard retrieval 상황을 가정합니다.
이때 핵심 목표는 retrieval cost를 줄이면서도 relevant document가 포함된 shard를 최대한 보존하는 것입니다.

---

## 1. 연구 목적

대규모 문서 집합을 사용하는 RAG 시스템에서는 문서 컬렉션이 여러 개의 shard로 나뉘어 저장될 수 있습니다. 이때 모든 shard를 검색하면 relevant document를 놓칠 가능성은 낮지만, shard 수가 증가할수록 retrieval cost와 latency가 커집니다.

따라서 본 연구는 다음 질문을 중심으로 진행됩니다.

> 제한된 수의 shard만 접근할 때, relevant document를 포함한 shard를 어떻게 더 잘 보존할 수 있는가?

특히 다음 두 요소를 중심으로 shard selection 성능을 분석합니다.

1. **Centroid Representation**

   * shard를 하나의 평균 centroid로 표현할 것인가
   * shard 내부를 여러 local centroid로 나누어 표현할 것인가

2. **Grouping**

   * 전체 shard를 직접 ranking할 것인가
   * 유사한 shard를 group으로 묶고, 선택된 group 내부에서만 ranking할 것인가

---

## 2. 핵심 아이디어

### 2.1 Single-Centroid

Single-Centroid는 shard 내부의 모든 document embedding을 평균내어 shard를 하나의 centroid vector로 표현합니다.

이 방식은 계산이 단순하고 scoring cost가 낮지만, shard 내부에 여러 topic이 섞여 있을 경우 query와 관련된 local semantic region이 평균에 묻힐 수 있습니다.

### 2.2 Multi-Centroid

Multi-Centroid는 shard 내부의 document embedding을 clustering하여 여러 개의 local centroid로 표현합니다.

query와 shard의 score는 해당 shard의 여러 centroid 중 query와 가장 가까운 centroid를 기준으로 계산합니다.

이를 통해 shard 내부의 다양한 semantic region을 더 세밀하게 반영할 수 있는지 확인합니다.

### 2.3 Grouping

Grouping은 유사한 shard들을 상위 group으로 묶은 뒤, query와 관련 가능성이 높은 group을 먼저 선택하는 방식입니다.

이후 선택된 group 내부에서만 shard ranking을 수행합니다.

Grouping의 목적은 coverage를 직접 높이는 것보다는, coverage 손실을 제한하면서 centroid scoring cost를 줄일 수 있는지 확인하는 데 있습니다.

---

## 3. 비교 실험 설정

본 연구는 다음 4가지 shard selection 방식을 비교합니다.

| Representation  | No Grouping     | Grouping                   |
| --------------- | --------------- | -------------------------- |
| Single-Centroid | Single-Centroid | Grouping + Single-Centroid |
| Multi-Centroid  | Multi-Centroid  | Grouping + Multi-Centroid  |

각 설정의 역할은 다음과 같습니다.

| Method                     | Description                                               |
| -------------------------- | --------------------------------------------------------- |
| Single-Centroid            | 각 shard를 하나의 평균 centroid로 표현하고 ranking                    |
| Multi-Centroid             | 각 shard를 여러 local centroid로 표현하고 ranking                  |
| Grouping + Single-Centroid | group-level selection 이후 single-centroid 기반 shard ranking |
| Grouping + Multi-Centroid  | group-level selection 이후 multi-centroid 기반 shard ranking  |

---

## 4. 데이터셋 및 실험 조건

본 실험에서는 BEIR 계열 데이터셋을 사용합니다.

| Item                 | Setting                     |
| -------------------- | --------------------------- |
| Dataset              | TREC-COVID, FEVER, MS-MARCO |
| Number of Shards     | 32                          |
| Shard Type           | topic-based, heterogeneous  |
| Ranking Score        | cosine similarity           |
| Effectiveness Metric | Oracle Document Coverage@K  |
| Cost Metric          | Centroid Scoring Cost       |

### Shard 구성 방식

| Shard Type    | Description                                    |
| ------------- | ---------------------------------------------- |
| topic-based   | embedding clustering을 통해 유사한 문서끼리 같은 shard에 배치 |
| heterogeneous | 서로 다른 topic의 문서가 섞이도록 shard 구성                 |

---

## 5. 평가 지표

### 5.1 Oracle Document Coverage@K

선택된 shard 집합이 relevant document를 얼마나 잘 보존하는지를 측정합니다.

즉, Top-B shard selection 이후에도 정답 문서가 검색 가능한 후보 공간 안에 남아 있는지를 평가합니다.

### 5.2 Centroid Scoring Cost

query 하나를 처리할 때 비교해야 하는 centroid 수를 의미합니다.

예를 들어 shard가 32개이고 shard당 centroid가 8개라면, grouping을 사용하지 않는 Multi-Centroid 방식의 centroid scoring cost는 다음과 같습니다.

```text
32 shards × 8 centroids = 256
```

Grouping을 사용하면 선택된 group 내부의 shard만 scoring하므로 cost를 줄일 수 있습니다.

---

## 6. Repository 구조

```text
multi-sharded-rag/
├─ README.md
├─ requirements.txt
│
├─ configs/
│  ├─ pilot/
│  │  └─ scifact_random_single.yaml
│  └─ full/
│     ├─ scifact_ablation.yaml
│     ├─ nfcorpus_ablation.yaml
│     └─ beir_ablation.yaml
│
├─ scripts/
│  ├─ run_pilot_scifact.sh
│  ├─ run_pilot_scifact.ps1
│  ├─ run_full_scifact.sh
│  └─ run_full_scifact.ps1
│
├─ src/
│  ├─ data/
│  │  ├─ load_beir.py
│  │  ├─ inspect_dataset.py
│  │  └─ io_utils.py
│  │
│  ├─ embedding/
│  │  ├─ build_embeddings.py
│  │  └─ embedding_store.py
│  │
│  ├─ sharding/
│  │  ├─ random_shard.py
│  │  ├─ topic_shard.py
│  │  └─ heterogeneous_shard.py
│  │
│  ├─ representation/
│  │  ├─ single_centroid.py
│  │  └─ multi_centroid.py
│  │
│  ├─ grouping/
│  │  └─ build_groups.py
│  │
│  ├─ ranking/
│  │  ├─ single_centroid_ranker.py
│  │  ├─ multi_centroid_ranker.py
│  │  ├─ group_single_ranker.py
│  │  └─ group_multi_ranker.py
│  │
│  ├─ evaluation/
│  │  ├─ relevant_shard.py
│  │  ├─ shard_recall.py
│  │  ├─ group_recall.py
│  │  └─ result_writer.py
│  │
│  ├─ experiment/
│  │  ├─ config.py
│  │  ├─ runner.py
│  │  └─ run_pilot.py
│  │
│  └─ utils/
│     ├─ seed.py
│     ├─ logging_utils.py
│     └─ path_utils.py
│
└─ tests/
   ├─ test_shard.py
   ├─ test_centroid.py
   └─ test_metrics.py
```

---

## 7. 주요 모듈 설명

| Directory             | Role                                                            |
| --------------------- | --------------------------------------------------------------- |
| `src/data/`           | BEIR 데이터셋 로드 및 입출력 유틸                                           |
| `src/embedding/`      | document/query embedding 생성                                     |
| `src/sharding/`       | random, topic-based, heterogeneous shard 구성                     |
| `src/representation/` | Single-Centroid, Multi-Centroid 계산                              |
| `src/grouping/`       | shard grouping 구성                                               |
| `src/ranking/`        | 각 setting별 shard ranking                                        |
| `src/evaluation/`     | relevant shard mapping, recall, coverage, group-level metric 계산 |
| `src/experiment/`     | 전체 실험 실행                                                        |

---

## 8. 설치 방법

Python 가상환경을 생성한 뒤 필요한 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

주요 dependency는 다음과 같습니다.

| Library                 | Role                                  |
| ----------------------- | ------------------------------------- |
| `numpy`                 | embedding matrix 및 vector 연산          |
| `pandas`                | 실험 결과 저장 및 분석                         |
| `scikit-learn`          | KMeans, cosine similarity, clustering |
| `torch`                 | embedding model backend               |
| `sentence-transformers` | document/query embedding 생성           |
| `beir`                  | BEIR benchmark 데이터셋 로드                |
| `pyyaml`                | YAML config 로드                        |

---

## 9. 실행 방법

### 9.1 데이터셋 로드 확인

```bash
python -m src.data.load_beir
```

정상 실행 시 BEIR 데이터셋의 corpus, query, qrels 정보가 출력됩니다.

### 9.2 Embedding 생성

```bash
python -m src.embedding.build_embeddings
```

실행 후 다음과 같은 파일이 생성됩니다.

```text
data/embeddings/{dataset}/doc_embeddings.npy
data/embeddings/{dataset}/query_embeddings.npy
data/embeddings/{dataset}/doc_ids.json
data/embeddings/{dataset}/query_ids.json
```

---

## 10. 실험 파이프라인

전체 실험 흐름은 다음과 같습니다.

```text
BEIR dataset load
→ document/query embedding 생성 또는 로드
→ shard partition 구성
→ relevant shard mapping 생성
→ centroid representation 계산
→ optional grouping
→ query별 shard ranking
→ Top-B shard selection
→ coverage / cost metric 계산
→ result CSV 저장
```

---

## 11. 현재까지의 주요 관찰

현재 실험에서 확인한 주요 경향은 다음과 같습니다.

1. **Multi-Centroid는 일부 조건에서 Single-Centroid보다 높은 coverage를 보였습니다.**

   * shard 내부의 local semantic signal을 더 세밀하게 반영할 수 있기 때문으로 해석됩니다.
   * 다만 모든 dataset과 shard 구성에서 일관되게 개선되지는 않았습니다.

2. **Centroid 수 증가는 항상 coverage 향상으로 이어지지 않았습니다.**

   * topic-based shard에서는 centroid 수 증가에 따라 coverage가 완만히 증가하는 경우가 있었습니다.
   * heterogeneous shard에서는 centroid 수 증가가 일관된 성능 향상으로 이어지지 않았습니다.

3. **Grouping은 coverage를 일부 감소시킬 수 있지만, scoring cost를 줄였습니다.**

   * group-level pruning 과정에서 일부 relevant shard가 제외될 위험이 있습니다.
   * 하지만 Multi-Centroid처럼 scoring cost가 큰 방식에서는 Grouping의 cost 감소 효과가 더 크게 나타났습니다.

---

## 12. 주의 사항

### Embedding과 ID mapping

embedding matrix와 id mapping의 순서는 반드시 유지되어야 합니다.

```text
doc_embeddings[i]   == doc_ids[i]에 해당하는 document embedding
query_embeddings[j] == query_ids[j]에 해당하는 query embedding
```

이 mapping이 깨지면 shard 구성, centroid 계산, ranking, coverage 계산 결과가 모두 잘못될 수 있습니다.

### Git에 포함하지 않는 파일

다음 디렉토리는 용량이 커지므로 Git에 올리지 않습니다.

```text
data/
results/
logs/
```

---

## 13. Project Status

현재 구현 및 분석 범위는 다음과 같습니다.

* BEIR 데이터셋 로드
* document/query embedding 생성
* random, topic-based, heterogeneous shard 구성
* Single-Centroid / Multi-Centroid representation
* Grouping 기반 shard selection
* query별 shard ranking
* coverage 및 scoring cost 계산
* 결과 CSV 저장 및 분석
