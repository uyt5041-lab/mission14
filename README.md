# Mission 14 — 근거를 검증하는 연말정산 RAG

국세청의 **2024년 귀속 연말정산 신고안내**를 대상으로 LangChain 기반 RAG를 구현하는 프로젝트입니다. 단순히 답변을 생성하는 데서 끝내지 않고, 검색 근거가 맞았는지 먼저 평가한 뒤 답변 품질을 평가합니다.

## 이번 저장소의 원칙

1. 검색 결과와 생성 답변을 분리해서 관찰합니다.
2. PDF 페이지, 섹션, 청크 ID를 모든 근거에 남깁니다.
3. 표의 `종전 / 개정 / 적용시기` 관계를 보존합니다.
4. 청크 크기는 감으로 정하지 않고 Hit@k와 MRR로 고릅니다.
5. 문서에 없는 질문은 추측하지 않고 범위 밖이라고 답합니다.

## 목표 구조

```text
PDF → 페이지 추출/정제 → metadata → token-aware chunking
    → Dense(KURE-v1) + BM25 → weighted RRF → reranker
    → 근거 5개 → Qwen 4-bit → 답변 + PDF 페이지
```

## 파일 안내

- `PLAN_CHECKLIST.md`: 실행 순서, 완료 조건, 실행 기록
- `notebooks/mission14_rag_colab.ipynb`: Colab에서 위에서 아래로 실행하는 메인 노트북
- `src/rag_core.py`: 정제, 청킹, 검색, 융합, 평가, LCEL 생성 체인
- `data/evaluation_qa.json`: 15개 평가 질문 초안
- `data/metadata_rules.json`: 원문 확인 후 입력하는 part/section/printed-page 범위
- `requirements-colab.txt`: Colab 의존성
- `tests/test_rag_core.py`: 네트워크/GPU 없이 돌리는 핵심 로직 테스트

## 빠른 시작

1. GitHub에서 `notebooks/mission14_rag_colab.ipynb`를 Colab으로 엽니다.
2. 런타임을 GPU로 변경합니다.
3. 노트북을 위에서 아래로 실행합니다.
4. PDF 다운로드가 막히면 노트북의 업로드 fallback을 사용합니다.
5. 각 단계가 통과할 때마다 `PLAN_CHECKLIST.md`의 실행 기록을 갱신합니다.

공식 원문 기본 URL은 국세청 다운로드 링크를 사용하며, 노트북 상단에서 다른 미션 제공 URL이나 Drive 파일 경로로 바꿀 수 있습니다.

## 현재 상태

저장소 골격과 실행 코드에 대한 정적 검증까지 완료한 상태입니다. PDF 전체 파싱, 임베딩, Qwen 생성은 Colab GPU 실행 결과가 있어야 완료로 표시합니다. 현재 체크포인트는 [PLAN_CHECKLIST.md](PLAN_CHECKLIST.md)를 기준으로 판단합니다.
