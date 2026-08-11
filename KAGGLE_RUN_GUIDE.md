# Mission 14 Kaggle + Google Drive 실행 가이드

## 1. Kaggle notebook 만들기

`notebooks/kernel-metadata.json`의 `id`가 본인 Kaggle owner/slug와 다르면 먼저 수정합니다.

```bash
pip install -U kaggle
kaggle auth login
kaggle kernels push -p notebooks
```

권장 기본 slug는 `chattybeak/mission14-korean-tax-rag`입니다. Kaggle 설정에서 GPU와 Internet을 켭니다. Internet을 끌 경우 이 저장소와 원문 PDF를 private Dataset으로 첨부해야 합니다.

첫 setup cell은 기본적으로 `main`을 가져오며, 기존 checkout이 있어도 최신 ref를 다시 fetch/checkout합니다. 이어서 `requirements-kaggle.txt`와 핵심 source/data 파일의 존재를 확인한 뒤에만 package를 설치합니다. 실행에 사용한 Git commit은 `source_revision`으로 config hash에 포함되므로, 서로 다른 코드 버전으로 같은 run을 잘못 resume할 수 없습니다. 다른 브랜치를 시험할 때만 Kaggle 환경변수 `MISSION14_REPO_REF`에 브랜치명을 지정합니다.

## 2. Google Drive OAuth를 1회 준비

Kaggle VM에서 OOB manual copy/paste 인증을 하지 않습니다.

1. Google Cloud에서 Drive API를 활성화합니다.
2. Desktop OAuth client를 만듭니다.
3. 로컬 PC 또는 Colab의 browser flow로 authorized-user token JSON을 만듭니다.
4. 가능하면 scope는 `https://www.googleapis.com/auth/drive.file`만 사용합니다.
5. Kaggle Secrets에 아래 두 값을 저장합니다.

| Secret | 내용 |
|---|---|
| `GOOGLE_OAUTH_TOKEN_JSON` | authorized-user token JSON 전체 |
| `GOOGLE_DRIVE_BACKUP_FOLDER_ID` | Mission 14 backup 전용 Drive folder ID |

값을 notebook Markdown, code, output, GitHub에 붙여 넣지 않습니다.

## 3. Inspection run

notebook 기본값을 유지합니다.

```python
RUN_MODE = "inspect"
RESUME_RUN_ID = ""
RESUME_DRIVE_FILE_ID = ""
```

Phase 0~2에서 다음을 확인합니다.

- PDF SHA-256과 page count
- PDF page 1, 33, 106, 216 추출 결과
- header/footer 제거 전제
- printed page offset
- part·section page range
- 평가 질문별 required fact와 acceptable page

Inspection gate가 발생하는 것은 정상입니다. 이 run에서는 index를 만들지 않습니다.

확인값을 다음 파일에 반영합니다.

- `data/metadata_rules.json`
- `data/evaluation_qa.json`
- 필요한 경우 `data/table_corrections.json`

`gold_status`는 사람이 PDF와 대조한 질문만 `verified`로 바꿉니다.

## 4. Build run

확인 파일을 GitHub에 반영하고 새 Kaggle Version에서 다음만 변경합니다.

```python
RUN_MODE = "build"
```

Build gate는 다음을 모두 요구합니다.

- metadata rule 1개 이상
- in-scope verified question 1개 이상
- verified question의 모든 required fact에 acceptable page 존재
- CUDA 사용 가능

본 실행은 C1/C2/C3를 비교한 뒤 `all_facts_covered@5 → MRR → any_hit@5` 순으로 best chunk를 고릅니다. Basic Dense, Hybrid RRF, Hybrid+Rerank도 동일 gold set으로 비교합니다.

## 5. 중단 후 재시작

Drive backup 성공 출력의 Run ID와 file ID를 기록합니다. 새 session에서:

```python
RESUME_RUN_ID = "이전 Run ID"
RESUME_DRIVE_FILE_ID = "Drive checkpoint ZIP file ID"
```

복구 시 다음이 다르면 resume가 중단됩니다.

- config hash
- PDF SHA-256, bytes, page count

설정을 바꿔야 하면 같은 run을 이어가지 말고 새 run으로 시작합니다.

## 6. Kaggle output

완료 run은 최소 다음을 포함합니다.

```text
config.json
run_manifest.json
environment.txt
checkpoints/cleaned_pages.jsonl.gz
indexes/chroma_*/
results/chunk_stats.json
results/dense_summary.json
results/retrieval_summary.json
results/precomputed_evidence.json
results/generation_comparison.csv
results/metrics.json
results/stage_summary.md
COMPLETE.lock.json
```

`COMPLETE.lock.json`이 없으면 완료 run으로 간주하지 않습니다.

## 7. 실행 notebook을 기존 Drive 링크에 동기화

Kaggle에서 Save Version을 완료한 뒤 로컬 또는 Colab에서 실행합니다.

```bash
python scripts/sync_kaggle_to_drive.py \
  --kernel chattybeak/mission14-korean-tax-rag \
  --drive-token-json /안전한/로컬/경로/token.json \
  --drive-notebook-file-id EXISTING_NOTEBOOK_FILE_ID \
  --drive-results-folder-id RESULTS_FOLDER_ID
```

스크립트는 다음을 검증합니다.

1. 최신 Kaggle source와 output pull
2. notebook JSON parse
3. `metrics.json` 기반 첫 summary cell 갱신
4. timestamp local backup
5. Drive `files.update`로 기존 file ID의 bytes 교체
6. 같은 file ID인지 확인
7. Drive에서 다시 다운로드
8. JSON parse와 SHA-256 readback 일치 확인
9. 선택적으로 output ZIP을 Drive folder에 업로드

Drive update가 실패하면 기존 파일을 지우거나 새 파일로 바꾸지 않습니다.

## 8. 완료 판정

- Kaggle 최신 Version이 성공
- GPU/device 로그 존재
- `metrics.json`과 notebook 출력 일치
- `COMPLETE.lock.json` 존재
- Drive checkpoint size 확인
- Drive notebook file ID 유지
- Drive readback notebook parse와 checksum 통과
- secret, model weight, Chroma index가 GitHub에 없음
