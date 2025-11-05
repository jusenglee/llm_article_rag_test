# LLM Article RAG 테스트 프로젝트

## 개요
이 저장소는 대규모 논문 데이터셋을 기반으로 검색 증강 생성(RAG) 파이프라인을 구성하고, FastAPI 웹 애플리케이션을 통해 질의 응답을 제공하는 예제 프로젝트입니다. 논문 메타데이터/본문을 Qdrant에 적재한 뒤, LLM으로 질의를 확장하고 검색 결과를 재랭킹하여 근거가 포함된 답변을 생성합니다. 

## 디렉터리 구조
```
.
├── main.py                # FastAPI 앱 진입점
├── query_pipeline.py      # RAG 검색, 프롬프트 구성, Triton 추론 로직
├── qdrant_rag_script/     # 데이터 다운로드 및 벡터 DB 적재 스크립트
├── data/                  # 토크나이저 및 샘플 데이터
└── README.md
```

## 주요 컴포넌트
### 1. FastAPI 애플리케이션 (`main.py`)
- 애플리케이션 시작 시 `build_rag_objects()`를 호출하여 Qdrant 클라이언트, 임베딩 모델, LlamaIndex 검색기를 초기화합니다.
- `/` 엔드포인트는 기본 HTML 템플릿을 렌더링하며, `/query` 엔드포인트가 실제 질문을 처리합니다.
- 질문 입력 시 LLM이 RAG 수행 여부를 먼저 판단하고(`decide_rag_needed`), 필요하면 질의를 확장한 뒤 재랭킹과 컨텍스트 구성, 최종 응답 생성을 수행합니다.

### 2. RAG 파이프라인 (`query_pipeline.py`)
- **Triton 추론**: `triton_infer`가 gRPC 기반 Triton Inference Server로 프롬프트를 보내고 스트리밍 응답을 수집합니다.
- **벡터 검색**: Qdrant에서 검색(`dense_retrieve_hybrid`) 후 RapidFuzz 기반 키워드 부스팅과 RRF 재랭킹(`rrf_rerank`)을 수행합니다.
- **게이트 판단**: 검색 스코어, 질의 패턴, 키워드 품질을 기준으로 RAG 수행 여부를 결정합니다.
- **컨텍스트 생성**: 검색 결과에서 문서 스니펫과 출처 매핑을 만들어 프롬프트를 구성하며, 토큰 예산을 초과하지 않도록 잘라냅니다.

### 3. 데이터 적재 스크립트 (`qdrant_rag_script/`)
- `01_make_dataset.py`: Hugging Face `allenai/peS2o` 데이터셋을 스트리밍으로 내려받아 JSONL 샘플을 생성합니다.
- `02_ingest_to_qdrant.py`: 논문 본문을 청크로 나누고 HuggingFace 임베딩을 계산하여 Qdrant에 삽입합니다. 재시작 가능한 상태 파일을 이용해 긴 적재 작업을 지원합니다.

## 환경 변수 및 설정
주요 설정은 환경 변수로 제어합니다. 필요 시 `.env` 파일을 만들어 FastAPI 또는 스크립트 실행 전에 로드하세요.

| 이름 | 기본값 | 설명 |
| --- | --- | --- |
| `QDRANT_HOST` | `localhost` | Qdrant gRPC 호스트 |
| `QDRANT_URL` | `localhost:6333` | Qdrant HTTP URL |
| `QDRANT_COLLECTION` | `peS2o_rag` | 검색에 사용할 컬렉션 이름 |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | LlamaIndex 임베딩 모델 |
| `TRITON_URL` | `localhost:8001` | Triton Inference Server 주소 |
| `TRITON_MODEL` | `gemma_vllm_0` | Triton 상의 모델 이름 |
| `TOKENIZER_ID` | `./` | 토크나이저 경로 또는 ID |
| `MILVUS_COLLECTION` | `peS2o_rag` | (Milvus 적재 시) 컬렉션 이름 |

## 설치 및 실행
1. **의존성 설치**
   ```bash
   pip install -r requirements.txt  # requirements 파일이 없다면 아래 패키지를 수동 설치
   pip install fastapi uvicorn[standard] qdrant-client llama-index transformers tritonclient[grpc] rapidfuzz tqdm orjson sentence-transformers networkx scikit-learn pandas matplotlib
   ```

2. **환경 변수 설정**
   - Qdrant, Triton, (선택) Milvus 인프라 주소를 올바르게 지정합니다.
   - `TOKENIZER_ID`가 로컬 토크나이저 디렉터리(`data/`)를 가리키도록 설정할 수 있습니다.

3. **FastAPI 서버 실행**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8080
   ```
   브라우저에서 `http://localhost:8080`에 접속해 질의를 테스트합니다.

4. **CLI 모드 테스트**
   ```bash
   python query_pipeline.py
   ```
   터미널에서 질문을 입력하면 RAG 파이프라인이 순차적으로 실행됩니다.

5. **데이터 준비 (선택)**
   ```bash
   # Hugging Face 데이터셋 샘플링
   python qdrant_rag_script/01_make_dataset.py

   # Qdrant로 청크 적재
   python qdrant_rag_script/02_ingest_to_qdrant.py
   ```

## 개발 참고 사항
- `query_pipeline.py`는 GPU 환경을 전제로 Hugging Face 임베딩 모델과 Triton 서버를 사용합니다. 로컬 CPU 환경에서 실행할 경우 `device="cpu"`로 변경하거나 경량 모델을 선택해야 합니다.
- RAG 게이트 로직은 RapidFuzz를 활용한 키워드 매칭 점수, 검색 스코어, 질의 유형 휴리스틱에 기반합니다. 필요 시 스코어 임계값(`SCORE_THRESHOLD`) 등을 조정하세요.
- FastAPI 앱은 `static/`, `templates/` 디렉터리가 있다고 가정하고 있으므로 배포 시 해당 리소스를 준비해야 합니다.

## 라이선스
프로젝트에 대한 명시적 라이선스 파일이 없으므로 사용 전에 소유자와 협의하세요.
