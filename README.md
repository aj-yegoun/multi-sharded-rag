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

## 3. 주요 디렉토리 설명
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


