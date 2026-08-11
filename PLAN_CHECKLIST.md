# Mission 14 실행 플랜 및 체크리스트

최종 목표는 “답변이 그럴듯하다”가 아니라, **정답 근거를 찾았고 그 근거만으로 답했다는 사실을 재현 가능한 결과로 증명하는 것**입니다.

## 상태 표기

- `[x]` 구현 또는 실행 결과를 확인함
- `[ ]` 아직 실행 증빙이 없음
- 체크박스 옆의 **완료 조건**을 통과해야 `[x]`로 바꿈
- 코드 작성만 끝난 GPU 단계는 완료로 표시하지 않음

## 최종 합격 기준

- 같은 런타임에서 새로 시작해 노트북 전체가 순서대로 실행됨
- 426쪽 내외의 PDF가 정상 로드되고 페이지/섹션/청크 metadata가 보존됨
- 평가 질문의 gold page가 사람이 확인되어 있음
- C1/C2/C3 청킹 실험의 Hit@1, Hit@5, MRR이 기록됨
- Basic RAG와 Advanced RAG를 같은 평가셋으로 비교함
- 최종 답변에 PDF 페이지와 근거가 표시됨
- 범위 밖 질문 2개를 문서 밖이라고 안전하게 처리함
- 실패 사례 2개 이상에서 원인과 개선을 연결함

---

## Phase 0. 저장소와 재현 환경

- [x] 시스템 구조와 실험 순서를 문서화했다.
- [x] Colab용 의존성 목록을 만들었다.
- [x] 랜덤 시드와 기본 설정을 한 곳에 모았다.
- [x] 국세청 원문 URL과 로컬/Drive fallback 경로를 분리했다.
- [ ] 새 Colab GPU 런타임에서 설치 셀이 오류 없이 끝났다.
- [ ] `torch`, `transformers`, `langchain`, GPU 이름을 실행 결과로 남겼다.

**완료 조건:** 새 Colab 런타임에서 setup 및 environment 셀을 재실행할 수 있고 버전 정보가 출력된다.

## Phase 1. PDF 로드와 무결성 확인

- [x] PDF 다운로드, `%PDF` signature, SHA-256, 페이지 수 검사 코드를 작성했다.
- [x] URL 실패 시 파일 업로드 fallback을 작성했다.
- [ ] 실제 파일 SHA-256을 기록했다.
- [ ] 페이지 수가 예상 범위(400~450)인지 확인했다.
- [ ] 첫 페이지, 개정세법 표 페이지, 본문 페이지를 눈으로 비교했다.

**완료 조건:** PDF의 hash와 페이지 수가 출력되고, 최소 3개 대표 페이지의 추출 텍스트가 원문과 대응한다.

## Phase 2. 텍스트 정제와 metadata

- [x] 반복 line 탐지와 보수적 cleaning 함수를 작성했다.
- [x] `pdf_page`, `printed_page`, `section`, `content_type` metadata 구조를 만들었다.
- [ ] 반복 header/footer 제거 전후 샘플을 저장했다.
- [ ] 금액·비율·조항·`종전/개정/적용시기`가 보존되는지 확인했다.
- [ ] printed page offset 또는 추출 규칙을 실제 PDF에서 확정했다.

**완료 조건:** 정제 전후 비교에서 반복 문구는 줄고 세법상 의미가 있는 기호와 조건은 유지된다.

## Phase 3. 핵심 표 보정

- [x] 표 보정 문서를 일반 본문과 합칠 수 있는 데이터 구조를 작성했다.
- [ ] 개정세법 요약 표의 대상 PDF page 목록을 확정했다.
- [ ] 월세액 공제 표를 `[항목]/[종전]/[개정]/[적용시기]` 문장으로 변환했다.
- [ ] 공제율·한도, 비거주자, 제출기한 표를 같은 방식으로 보정했다.
- [ ] 최소 5개 표 문서를 사람이 원문과 대조했다.

**완료 조건:** 표 질문에서 종전 값과 개정 값을 뒤섞지 않는 독립 `table` 문서가 5개 이상 존재한다.

## Phase 4. 평가 질문과 gold evidence

- [x] 15개 평가 질문 유형과 schema를 만들었다.
- [x] 범위 밖 질문 2개를 포함했다.
- [ ] 각 in-scope 질문의 `expected_answer`를 원문 기준으로 확정했다.
- [ ] 각 in-scope 질문의 `gold_pdf_pages`를 직접 확인했다.
- [ ] gold page가 복수인 질문은 필수/허용 page를 구분했다.

**완료 조건:** in-scope 질문에 빈 gold page가 없고, 근거 위치를 제3자가 PDF에서 재확인할 수 있다.

## Phase 5. 청킹 실험

- [x] C1(300/50), C2(500/80), C3(800/120) 설정을 코드화했다.
- [x] 한국어 문서용 separator와 section header prepend를 구현했다.
- [ ] 각 설정의 청크 수와 token 길이 분포를 기록했다.
- [ ] 과도하게 짧거나 긴 청크 샘플을 확인했다.
- [ ] C1/C2/C3 각각의 Dense index를 만들었다.

**완료 조건:** 세 설정이 동일한 정제 문서에서 만들어지고 청크 수·길이·샘플이 비교 가능하다.

## Phase 6. Dense baseline과 검색 평가

- [x] KURE-v1 normalized embedding과 Chroma 저장/재사용 코드를 작성했다.
- [x] Hit@1, Hit@5, MRR 계산 함수를 작성했다.
- [ ] C1/C2/C3의 동일 질문별 검색 결과를 저장했다.
- [ ] 청킹별 Hit@1, Hit@5, MRR, 평균 latency를 기록했다.
- [ ] metric과 실패 사례를 근거로 best chunk를 선택했다.

**중단 기준:** Hit@5가 낮으면 LLM 연결 전에 gold page, PDF 추출, 표 문서, query를 먼저 점검한다.

**완료 조건:** best chunk 선택 근거가 숫자와 실패 사례로 남아 있다.

## Phase 7. Basic RAG

- [x] 검색 결과를 rank/page/section/score/snippet으로 표시하는 구조를 작성했다.
- [x] 문서 근거만 사용하도록 하는 prompt와 LCEL 생성 체인을 작성했다.
- [x] Qwen 4-bit NF4 로딩 코드를 작성했다.
- [ ] 대표 질문 3개에 대해 검색 근거와 답변을 저장했다.
- [ ] 근거는 맞고 답변이 틀린 generation failure를 별도로 분류했다.
- [ ] OOM 없이 embedding model과 LLM을 순차 로드했다.

**완료 조건:** 질문 → 근거표 → 답변 → page citation이 한 함수 호출에서 재현된다.

## Phase 8. Hybrid retrieval과 reranking

- [x] BM25와 한국어/숫자 보존 tokenizer를 구현했다.
- [x] weighted Reciprocal Rank Fusion을 구현했다.
- [x] cross-encoder reranking 연결 함수를 구현했다.
- [ ] Dense 15 + BM25 15 → RRF top 12 결과를 저장했다.
- [ ] Reranker 적용 전후 순위 변화를 확인했다.
- [ ] neighbor expansion이 필요한 실제 실패 사례를 1개 이상 확인했다.

**완료 조건:** 동일 질문셋에서 Dense와 Hybrid+Rerank의 Hit@k/MRR 차이가 기록된다.

## Phase 9. 최종 비교와 보고서

- [ ] E0(No RAG), E1(Basic), E4(Advanced)를 같은 질문으로 실행했다.
- [ ] 정확성·근거 충실성·조건/예외·완전성·citation rubric을 채점했다.
- [ ] 청크 길이 histogram을 만들었다.
- [ ] 청킹별 Hit@5와 retriever별 MRR을 시각화했다.
- [ ] No RAG/Basic/Advanced answer score를 비교했다.
- [ ] 실패 사례 2개 이상과 개선 원인을 작성했다.
- [ ] 노트북 첫머리 Executive Summary에 최종 수치를 반영했다.

**완료 조건:** 제출자가 노트북 첫 화면만 보고도 최종 구조, best 설정, 핵심 성과, 한계를 이해할 수 있다.

---

## 실행 기록

| 날짜(UTC) | 단계 | 상태 | 증빙/메모 |
|---|---|---|---|
| 2026-08-11 | 저장소 초기화 | 완료 | README, 체크리스트, 노트북, core module, 평가셋 생성 |
| 2026-08-11 | 정적 검증 | 완료 | Python compile, JSON parse, 노트북 15개 code-cell parse, unittest 4/4 통과 |
| 2026-08-11 | 리뷰 P1/P2 보강 | 완료 | 숫자 보존, unique chunk ID, metadata fingerprint, hard gate, multi-fact metric, Qwen chat template |
| 2026-08-11 | Kaggle 실행판 작성 | 완료 | run contract, resume, Drive backup, COMPLETE lock, post-run Drive notebook sync 코드 |
| 2026-08-11 | 보강판 정적 검증 | 완료 | unittest 15/15, Python compile, Colab/Kaggle code-cell parse, gate 순서, secret scan |
| 미실행 | Colab 환경/전체 PDF | 대기 | GPU 런타임과 실제 PDF 실행 결과 필요 |

## Kaggle 전환 체크리스트

- [x] `/kaggle/input`과 `/kaggle/working` 경로를 사용한다.
- [x] Inspection과 Build run을 분리했다.
- [x] Gate가 청킹과 인덱싱보다 먼저 실행된다.
- [x] CUDA가 없을 때 최종 transformer를 CPU로 조용히 전환하지 않는다.
- [x] config hash와 PDF fingerprint가 다른 resume를 막는다.
- [x] Drive checkpoint archive의 경로 탈출을 검사한다.
- [x] Drive upload의 file ID와 size를 기록한다.
- [x] Kaggle 실행본을 기존 Drive notebook file ID에 update하는 스크립트를 작성했다.
- [ ] 실제 Kaggle GPU에서 Inspection Version을 저장했다.
- [ ] 원문 확인값을 metadata/gold evidence에 반영했다.
- [ ] 실제 Kaggle GPU에서 Build Version을 저장했다.
- [ ] Drive checkpoint upload와 restore를 실제로 검증했다.
- [ ] Kaggle source/output을 pull하고 Drive notebook readback 검증을 통과했다.

## 다음 실행 순서

1. Colab에서 Phase 0 설치/환경 셀 실행
2. Phase 1 PDF hash·페이지 수·대표 페이지 확인
3. Phase 2 반복 문구와 printed page 규칙 확정
4. Phase 4 gold page부터 채운 뒤 청킹/검색 실험 시작

평가셋의 gold page가 비어 있는 동안에는 RAG 답변을 많이 생성하지 않습니다. 먼저 검색 평가의 기준부터 고정해야 이후 개선 결과를 믿을 수 있습니다.
