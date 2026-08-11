# Mission 14 — 근거를 검증하는 연말정산 RAG

국세청의 **2024년 귀속 연말정산 신고안내**를 대상으로 LangChain RAG를 구현합니다. 답변을 먼저 생성하지 않고, 검색 근거를 사람의 gold evidence와 비교한 뒤 생성 품질을 평가합니다.

## 바로 실행

- [Kaggle 실행 notebook](notebooks/mission14_rag_kaggle.ipynb)
- [Kaggle 업로드 metadata](notebooks/kernel-metadata.json)
- [Colab 실행 notebook](notebooks/mission14_rag_colab.ipynb)
- [실행 플랜·체크리스트](PLAN_CHECKLIST.md)
- [Kaggle·Google Drive 운영 가이드](KAGGLE_RUN_GUIDE.md)

Kaggle notebook은 처음에 `RUN_MODE = "inspect"`로 실행합니다. PDF 표본과 페이지 체계를 확인해 `data/metadata_rules.json`과 `data/evaluation_qa.json`을 채운 다음에만 `RUN_MODE = "build"`로 바꿉니다. Gate를 통과하기 전에는 청킹·임베딩·Chroma index를 만들지 않습니다.

## 구조

```text
PDF 무결성·SHA-256
→ 경계 기반 header/footer 정제
→ metadata + 표 보정
→ gold evidence STOP GATE
→ C1/C2/C3 token-aware chunking
→ Dense(KURE-v1) + BM25 + weighted RRF
→ BGE reranking
→ Qwen 4-bit chat template
→ 검색/답변 분리 평가
→ Kaggle output + Google Drive backup
```

## 이번 보강의 핵심

- `15`, `183` 같은 숫자 단독 행을 페이지 번호로 오인해 지우지 않습니다.
- 본문과 수동 표 문서가 같은 PDF page에 있어도 `chunk_id`가 충돌하지 않습니다.
- Chroma fingerprint에 page·section·content type 등 canonical metadata를 포함합니다.
- `any_hit@k`와 `all_facts_covered@k`를 분리하여 복합 질문을 과대평가하지 않습니다.
- Qwen Instruct tokenizer의 chat template을 명시적으로 적용합니다.
- CUDA가 없으면 reranker/generator를 조용히 CPU로 전환하지 않습니다.
- `config.json`, `run_manifest.json`, phase checkpoint, `metrics.json`, `COMPLETE.lock.json`을 남깁니다.
- 실행 중 artifact는 Kaggle working과 Drive에 이중 저장할 수 있습니다.
- Kaggle Save Version 후 기존 Drive notebook file ID를 유지한 채 bytes를 교체할 수 있습니다.

## 저장소 파일

```text
src/rag_core.py                 RAG 정제·청킹·검색·평가·생성
src/kaggle_ops.py               run contract·resume·Drive backup·seal
data/evaluation_qa.json         15개 질문과 required evidence schema
data/metadata_rules.json        육안 검증 후 채우는 metadata 규칙
data/table_corrections.json     검증된 표 보정 문서
notebooks/mission14_rag_kaggle.ipynb
notebooks/kernel-metadata.json
scripts/sync_kaggle_to_drive.py Kaggle 실행본을 기존 Drive file ID에 동기화
tests/                          네트워크/GPU 없는 핵심 테스트
```

## 로컬 정적 검증

```bash
python -m unittest discover -s tests -v
python tools/build_kaggle_notebook.py
python tools/harden_colab_notebook.py
```

## 현재 상태

코드와 notebook의 정적 검증만 완료했습니다. PDF 전체 파싱, 모델 다운로드, 임베딩, Qwen 생성, Drive OAuth upload는 실제 Kaggle GPU 실행 결과가 생긴 뒤 완료로 표시합니다. 실행되지 않은 성능 수치는 이 저장소에서 주장하지 않습니다.
