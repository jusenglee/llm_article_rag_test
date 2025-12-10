# LLM Article RAG 테스트 프로젝트

## 프로젝트 개요
이 저장소는 논문/학술 데이터를 활용한 검색 증강 생성(RAG) 실험용 코드베이스입니다. FastAPI 웹 앱에서 사용자의 질문을 받아 **두 개의 RAG 스택(A/B)**으로 검색·재랭킹·컨텍스트 생성을 수행한 뒤, Triton Inference Server에 연결된 LLM으로 답변을 스트리밍합니다. 질문이 RAG를 필요로 하는지 먼저 판별한 뒤에만 검색을 수행하도록 설계되어 있어 GPU·검색 리소스를 효율적으로 사용할 수 있습니다.

주요 특징:
- **이중 스택 비교**: A 스택(multilingual-e5-large-instruct)과 B 스택(multilingual-e5-large)을 동시에 실행하여 검색 품질을 비교.
- **게이트 로직**: 캐주얼 대화/사실성 질문을 구분해 불필요한 검색을 건너뜀.
- **하이브리드 검색**: 확장 질의 + 키워드 기반의 dense/lexical 검색 후 RRF 재랭킹.
- **스트리밍 응답**: Triton gRPC와 SSE를 조합해 웹 UI에서 실시간 생성 결과를 확인.

## 디렉터리 구조
```
.
├── main.py               # FastAPI 엔트리포인트 및 SSE 스트리밍 라우트
├── rag_pipeline.py       # 게이트 판정, RAG 단일/이중 실행 로직
├── retrieval.py          # 쿼리 확장, 하이브리드 검색, 컨텍스트 생성
├── rag_store.py          # Qdrant/임베딩/Retriever 초기화 및 캐싱
├── triton_client.py      # Triton gRPC 클라이언트 및 토크나이저 헬퍼
├── rag_types.py          # RAG 결과 데이터 클래스 정의
├── bench_logger.py       # A/B 결과를 JSONL로 로그 파일에 기록
├── qdrant_rag_script/    # 학술 데이터 다운로드 및 Qdrant 적재 스크립트
├── templates/index.html  # 간단한 웹 UI 템플릿
└── data/                 # 토크나이저 또는 샘플 데이터(필요 시 채워 넣음)
```

## 사전 준비
- Python 3.10 이상 권장
- GPU 가속 환경(CUDA)과 Triton Inference Server 접속 가능 주소
- Qdrant 서버(또는 호환 gRPC 엔드포인트)와 학술 문서 컬렉션이 준비되어 있어야 합니다.

## 환경 변수
`settings.py`에서 기본값을 정의합니다. 서비스 환경에 맞게 오버라이드하세요.

| 이름 | 기본값 | 설명 |
| --- | --- | --- |
| `QDRANT_HOST` | `211.241.177.73` | A 스택 Qdrant 호스트(gRPC) |
| `QDRANT_PORT` | `6334` | A 스택 Qdrant gRPC 포트 |
| `QDRANT_COLLECTION` | `e5_instruct_rag_100_000` | A 스택 컬렉션 이름 |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large-instruct` | A 스택 임베딩 모델 |
| `QDRANT_HOST_B` | `QDRANT_HOST`와 동일 | B 스택 Qdrant 호스트(gRPC) |
| `QDRANT_PORT_B` | `QDRANT_PORT`와 동일 | B 스택 Qdrant gRPC 포트 |
| `QDRANT_COLLECTION_B` | `e5_rag_100_000` | B 스택 컬렉션 이름 |
| `EMBEDDING_MODEL_B` | `intfloat/multilingual-e5-large` | B 스택 임베딩 모델 |
| `TRITON_URL` | `211.241.177.73:8001` | Triton Inference Server 주소 |
| `TRITON_MODEL` | `gemma_vllm_0` | 기본 LLM 모델 이름 |
| `RAG_BENCH_LOG_DIR` | `./rag_bench_logs` | A/B 로그 저장 경로 |

## 설치
의존성 관리 도구가 없다면 `pip`로 주요 패키지를 설치합니다.
```bash
pip install fastapi uvicorn[standard] qdrant-client llama-index \
           transformers tritonclient[grpc] rapidfuzz tqdm orjson \
           sentence-transformers networkx scikit-learn pandas matplotlib
```

## 실행 방법
1. **환경 변수 설정**: Qdrant/Triton 주소와 토크나이저 경로(`data/`)를 맞게 설정합니다.
2. **FastAPI 서버 기동**:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8080
   ```
3. **웹 UI 접근**: 브라우저에서 `http://localhost:8080`을 열어 질문을 입력합니다. 모델 선택(`gpt`/`gemma`)과 함께 `/query/stream` SSE 엔드포인트를 통해 생성 결과가 실시간으로 표시됩니다.

## 데이터 적재(선택)
Qdrant 컬렉션이 비어 있다면 `qdrant_rag_script/`를 사용해 샘플을 준비할 수 있습니다.
```bash
# Hugging Face에서 학술 데이터셋 샘플 생성
python qdrant_rag_script/01_make_dataset.py

# 생성된 청크를 Qdrant에 업로드
python qdrant_rag_script/02_ingest_to_qdrant.py
```

## 참고 사항
- `rag_store.py`는 Qdrant/임베딩/Retriever 객체를 싱글톤으로 캐싱하여 여러 FastAPI 워커 간에도 재사용합니다.
- `decide_rag_needed()`가 캐주얼 대화를 감지하면 검색 단계를 생략하고 순수 채팅 모드로 전환합니다.
- `bench_logger.py`를 실험 코드에서 호출하면 A/B 스택 결과를 JSONL로 축적하여 오프라인 분석에 활용할 수 있습니다.

