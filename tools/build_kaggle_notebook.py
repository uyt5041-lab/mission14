#!/usr/bin/env python3
"""Generate the checked-in Kaggle notebook from readable cell sources."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "mission14_rag_kaggle.ipynb"


def lines(source: str) -> list[str]:
    return dedent(source).strip("\n").splitlines(keepends=True)


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(source)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(source),
    }


cells = [
    md(
        """
        <!-- AUTO_SUMMARY_START -->
        # Mission 14 — Kaggle RAG 실행판

        ## 1. 최종 결과
        아직 실행 전입니다. 최종 수치는 `results/metrics.json`에서 이 영역으로 동기화합니다.

        ## 2. 실험 흐름 및 Notebook 구성
        PDF 무결성 → 육안 점검 → gold evidence STOP GATE → C1/C2/C3 → Dense → Hybrid → Rerank → Qwen

        ## 3. 전체 실험 결과 요약
        `any_hit@5`와 `all_facts_covered@5`를 분리하여, 관련 페이지 하나만 찾은 결과를 과대평가하지 않습니다.

        ## 4. 최종 결론
        현재 상태는 실행용 코드이며 실제 Kaggle output이 생기기 전에는 성능을 주장하지 않습니다.

        ## 5. 전체 코드 및 실행 결과
        아래 셀을 위에서 아래로 실행합니다. 첫 실행은 `RUN_MODE="inspect"`로 Phase 0~2까지만 수행합니다.
        <!-- AUTO_SUMMARY_END -->
        """
    ),
    md(
        """
        ## 실행 계약

        - Accelerator: GPU(T4 이상 권장)
        - Internet: ON 또는 `mission14` repository를 Kaggle Dataset으로 첨부
        - 첫 실행: `RUN_MODE="inspect"` — PDF와 metadata/gold page만 확인
        - 본 실행: 확인 결과를 저장소에 반영한 뒤 `RUN_MODE="build"`
        - Drive backup: Kaggle Secrets `GOOGLE_OAUTH_TOKEN_JSON`, `GOOGLE_DRIVE_BACKUP_FOLDER_ID`
        - 재시작: 이전 출력의 Run ID와 Drive file ID를 `RESUME_*`에 입력

        Secret 값은 코드·출력·GitHub에 기록하지 않습니다.
        """
    ),
    code(
        """
        from pathlib import Path
        import shutil, subprocess, sys

        REPO_URL = "https://github.com/uyt5041-lab/mission14.git"
        REPO_REF = "main"
        REPO_DIR = Path("/kaggle/working/mission14_repo")

        attached = [
            path.parent
            for path in Path("/kaggle/input").glob("**/requirements-kaggle.txt")
            if (path.parent / "src" / "rag_core.py").exists()
        ]
        if not REPO_DIR.exists():
            if attached:
                shutil.copytree(attached[0], REPO_DIR)
            else:
                subprocess.run(
                    ["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, str(REPO_DIR)],
                    check=True,
                )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_DIR / "requirements-kaggle.txt")],
            check=True,
        )
        sys.path.insert(0, str(REPO_DIR))
        print("Repository ready:", REPO_DIR)
        """
    ),
    code(
        """
        import json, os, platform, random, subprocess, time
        import numpy as np
        import pandas as pd
        import torch, transformers, langchain

        from src.kaggle_ops import (
            RagRunConfig,
            atomic_write_json,
            backup_run_to_drive,
            bind_input_pdf,
            create_run_layout,
            mark_phase_complete,
            restore_run_from_drive,
            seal_run,
            update_manifest,
        )

        RUN_MODE = "inspect"  # "inspect" -> 원문 확인, "build" -> gate 통과 후 전체 실행
        ENABLE_DRIVE_BACKUP = True
        RESUME_RUN_ID = ""
        RESUME_DRIVE_FILE_ID = ""
        WORK_ROOT = Path("/kaggle/working")

        CONFIG = RagRunConfig()
        if RESUME_RUN_ID and RESUME_DRIVE_FILE_ID:
            restore_run_from_drive(
                RESUME_DRIVE_FILE_ID,
                WORK_ROOT / CONFIG.project / RESUME_RUN_ID,
            )
        RUN = create_run_layout(WORK_ROOT, CONFIG, resume_run_id=RESUME_RUN_ID)
        RUN_DIR = RUN["run_dir"]
        RESULT_DIR = RUN["results"]
        INDEX_DIR = RUN["indexes"]

        random.seed(CONFIG.seed)
        np.random.seed(CONFIG.seed)
        torch.manual_seed(CONFIG.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(CONFIG.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        update_manifest(RUN_DIR, device={"cuda": torch.cuda.is_available(), "name": device_name})
        with (RUN_DIR / "environment.txt").open("w", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=stream, check=False)

        print("RUN_ID:", RUN["run_id"])
        print("Python:", sys.version.split()[0], "Platform:", platform.platform())
        print("torch:", torch.__version__, "transformers:", transformers.__version__)
        print("langchain:", langchain.__version__, "device:", device_name)
        """
    ),
    md(
        """
        ## 1. PDF 확보와 무결성

        `/kaggle/input`에 첨부된 PDF를 우선 사용합니다. 없으면 국세청 공식 URL을 시도합니다. 다운로드가 막히면 PDF를 private Kaggle Dataset으로 첨부하고 다시 실행합니다.
        """
    ),
    code(
        """
        import requests
        from src.rag_core import OFFICIAL_PDF_URL, validate_pdf

        pdf_candidates = sorted(Path("/kaggle/input").glob("**/*.pdf"))
        PDF_PATH = pdf_candidates[0] if pdf_candidates else RUN_DIR / "2024_year_end_tax_guide.pdf"
        if not PDF_PATH.exists():
            response = requests.get(OFFICIAL_PDF_URL, timeout=120)
            response.raise_for_status()
            if not response.content.startswith(b"%PDF"):
                raise RuntimeError(
                    "Official URL did not return a PDF. Attach the source PDF as a private Kaggle Dataset."
                )
            PDF_PATH.write_bytes(response.content)

        pdf_info = validate_pdf(PDF_PATH)
        bind_input_pdf(RUN_DIR, pdf_info)
        mark_phase_complete(RUN_DIR, "pdf_integrity", pdf_info)
        print(json.dumps(pdf_info, ensure_ascii=False, indent=2))
        """
    ),
    md("## 2. 페이지 추출, 보수적 정제, 육안 점검"),
    code(
        """
        import gzip
        from src.rag_core import apply_metadata_rules, load_and_clean_pdf

        pages, repeated_lines = load_and_clean_pdf(PDF_PATH)
        rules_path = REPO_DIR / "data" / "metadata_rules.json"
        metadata_rules = json.loads(rules_path.read_text(encoding="utf-8"))
        pages = apply_metadata_rules(pages, metadata_rules)

        clean_checkpoint = RUN["checkpoints"] / "cleaned_pages.jsonl.gz"
        with gzip.open(clean_checkpoint, "wt", encoding="utf-8") as stream:
            for doc in pages:
                stream.write(json.dumps(
                    {"page_content": doc.page_content, "metadata": doc.metadata},
                    ensure_ascii=False,
                ) + "\\n")
        atomic_write_json(
            RUN["checkpoints"] / "repeated_lines.json",
            sorted(repeated_lines),
        )
        mark_phase_complete(
            RUN_DIR,
            "page_cleaning",
            {"pages": len(pages), "repeated_boundary_lines": len(repeated_lines)},
        )
        print("pages:", len(pages), "repeated boundary lines:", len(repeated_lines))
        """
    ),
    code(
        """
        from IPython.display import display, Markdown

        SAMPLE_PDF_PAGES = [1, 33, 106, 216]
        inspection_rows = []
        for page_number in SAMPLE_PDF_PAGES:
            doc = pages[page_number - 1]
            display(Markdown(f"### PDF page {page_number}"))
            print(doc.page_content[:1800])
            print("metadata:", doc.metadata)
            inspection_rows.append({
                "pdf_page": page_number,
                "metadata": doc.metadata,
                "text_sample": doc.page_content[:1800],
            })
        atomic_write_json(RESULT_DIR / "inspection_samples.json", inspection_rows)
        print("원문 PDF와 대조하여 printed page, part, section 규칙을 확정하세요.")
        """
    ),
    md(
        """
        ## 3. Gold evidence와 실제 STOP GATE

        `gold_required_evidence`의 각 항목은 하나의 필수 사실입니다. `acceptable_pages`에는 그 사실을 입증할 수 있는 대체 가능 페이지들을 넣습니다. 여러 필수 사실 중 하나만 검색된 경우 `any_hit`는 성공하지만 `all_facts_covered`는 실패합니다.
        """
    ),
    code(
        """
        QA_PATH = REPO_DIR / "data" / "evaluation_qa.json"
        qa_rows = json.loads(QA_PATH.read_text(encoding="utf-8"))
        qa_display = []
        for row in qa_rows:
            evidence = row.get("gold_required_evidence") or []
            qa_display.append({
                "id": row["id"],
                "scope": row["scope"],
                "gold_status": row["gold_status"],
                "required_facts": len(evidence),
                "verified_page_groups": sum(bool(x.get("acceptable_pages")) for x in evidence),
                "question": row["question"],
            })
        display(pd.DataFrame(qa_display))
        print("metadata rules:", len(metadata_rules))
        """
    ),
    code(
        """
        from src.rag_core import validate_indexing_gate

        if RUN_MODE != "build":
            raise RuntimeError(
                "INSPECTION STOP: 원문 확인 결과를 metadata_rules.json과 evaluation_qa.json에 반영한 뒤 "
                "새 Kaggle Version에서 RUN_MODE='build'로 변경하세요. 인덱스는 아직 만들지 않았습니다."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("BUILD STOP: Kaggle GPU accelerator가 필요합니다. CPU fallback은 비활성화됨.")
        gate_result = validate_indexing_gate(metadata_rules, qa_rows)
        mark_phase_complete(RUN_DIR, "indexing_gate", gate_result)
        print("INDEXING GATE PASSED:", gate_result)
        """
    ),
    md("## 4. 검증된 표 보정 문서와 C1/C2/C3 청킹"),
    code(
        """
        from src.rag_core import CHUNK_CONFIGS, chunk_documents, make_table_document

        correction_rows = json.loads(
            (REPO_DIR / "data" / "table_corrections.json").read_text(encoding="utf-8")
        )
        table_documents = [make_table_document(**row) for row in correction_rows]
        source_documents = pages + table_documents
        chunks_by_config = {
            name: chunk_documents(source_documents, name)
            for name in CHUNK_CONFIGS
        }
        chunk_stats = {
            name: {
                "chunk_count": len(chunks),
                "unique_chunk_ids": len({doc.metadata["chunk_id"] for doc in chunks}),
                **CHUNK_CONFIGS[name],
            }
            for name, chunks in chunks_by_config.items()
        }
        atomic_write_json(RESULT_DIR / "chunk_stats.json", chunk_stats)
        mark_phase_complete(RUN_DIR, "chunking", chunk_stats)
        display(pd.DataFrame(chunk_stats).T)
        """
    ),
    code(
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from transformers import AutoTokenizer

        length_tokenizer = AutoTokenizer.from_pretrained(CONFIG.embedding_model)
        length_rows = []
        for config_name, chunks in chunks_by_config.items():
            for chunk in chunks:
                length_rows.append({
                    "config": config_name,
                    "tokens": len(length_tokenizer.encode(chunk.page_content, add_special_tokens=False)),
                })
        length_df = pd.DataFrame(length_rows)
        display(length_df.groupby("config")["tokens"].describe())
        figure_path = RESULT_DIR / "chunk_length_distribution.png"
        sns.histplot(
            data=length_df, x="tokens", hue="config", element="step",
            stat="density", common_norm=False,
        )
        plt.title("Chunk token-length distribution")
        plt.tight_layout()
        plt.savefig(figure_path, dpi=160)
        plt.show()
        """
    ),
    code(
        """
        def backup_checkpoint(label):
            if not ENABLE_DRIVE_BACKUP:
                return {"status": "disabled", "label": label}
            try:
                result = backup_run_to_drive(RUN_DIR)
                result.update({"status": "uploaded", "label": label})
                print("Drive backup verified:", result["file_id"], result["size"])
                return result
            except Exception as error:
                manifest = json.loads((RUN_DIR / "run_manifest.json").read_text(encoding="utf-8"))
                warnings = list(manifest.get("warnings", []))
                warnings.append({"phase": label, "error_type": type(error).__name__, "message": str(error)})
                update_manifest(RUN_DIR, warnings=warnings)
                print("Drive backup failed; Kaggle artifacts remain available:", type(error).__name__, str(error))
                return {"status": "failed", "label": label, "error_type": type(error).__name__}

        backup_checkpoint("after_chunking")
        """
    ),
    md("## 5. Dense C1/C2/C3 검색 평가"),
    code(
        """
        from src.rag_core import build_embeddings, dense_search, evaluate_retrieval, open_or_build_chroma

        embeddings = build_embeddings(model_name=CONFIG.embedding_model, device="cuda")
        vector_stores = {}
        dense_details = {}
        dense_summaries = {}
        for config_name, chunks in chunks_by_config.items():
            store = open_or_build_chroma(
                chunks,
                INDEX_DIR / f"chroma_kure_{config_name.lower()}",
                collection_name=f"mission14_kure_{config_name.lower()}",
                embeddings=embeddings,
            )
            vector_stores[config_name] = store
            started = time.perf_counter()
            detail, summary = evaluate_retrieval(
                qa_rows,
                lambda question, current=store: dense_search(current, question, k=5),
                k=5,
            )
            summary["latency_seconds_total"] = time.perf_counter() - started
            summary["latency_seconds_per_question"] = (
                summary["latency_seconds_total"] / summary["evaluated_questions"]
            )
            dense_details[config_name] = detail
            dense_summaries[config_name] = summary
            atomic_write_json(RESULT_DIR / f"dense_{config_name.lower()}_detail.json", detail)

        BEST_CONFIG = max(
            dense_summaries,
            key=lambda name: (
                dense_summaries[name][CONFIG.primary_metric],
                dense_summaries[name]["mrr"],
                dense_summaries[name]["any_hit@5"],
            ),
        )
        atomic_write_json(RESULT_DIR / "dense_summary.json", dense_summaries)
        mark_phase_complete(RUN_DIR, "dense_evaluation", {"best_config": BEST_CONFIG, **dense_summaries[BEST_CONFIG]})
        display(pd.DataFrame(dense_summaries).T.sort_values([CONFIG.primary_metric, "mrr"], ascending=False))
        print("Best chunk config:", BEST_CONFIG)
        """
    ),
    md("## 6. Basic Dense vs Hybrid vs Hybrid+Rerank"),
    code(
        """
        from sentence_transformers import CrossEncoder
        from src.rag_core import BM25Index, rerank_hits, weighted_rrf

        best_chunks = chunks_by_config[BEST_CONFIG]
        best_store = vector_stores[BEST_CONFIG]
        bm25 = BM25Index(best_chunks)
        reranker_model = CrossEncoder(CONFIG.reranker_model, device="cuda")

        def basic_search(query):
            return dense_search(best_store, query, k=5)

        def hybrid_search(query, use_reranker=True):
            dense_hits = dense_search(best_store, query, k=CONFIG.dense_k)
            bm25_hits = bm25.search(query, k=CONFIG.bm25_k)
            fused = weighted_rrf(
                {"dense": dense_hits, "bm25": bm25_hits},
                {"dense": 0.7, "bm25": 0.3},
                top_n=CONFIG.rrf_top_n,
            )
            if use_reranker:
                return rerank_hits(
                    query, fused, top_n=CONFIG.final_top_n, model=reranker_model
                )
            return fused[:CONFIG.final_top_n]

        retrievers = {
            "basic_dense": basic_search,
            "hybrid_rrf": lambda query: hybrid_search(query, use_reranker=False),
            "hybrid_rerank": lambda query: hybrid_search(query, use_reranker=True),
        }
        retrieval_details = {}
        retrieval_summaries = {}
        for name, search_fn in retrievers.items():
            started = time.perf_counter()
            detail, summary = evaluate_retrieval(qa_rows, search_fn, k=5)
            summary["latency_seconds_total"] = time.perf_counter() - started
            summary["latency_seconds_per_question"] = (
                summary["latency_seconds_total"] / summary["evaluated_questions"]
            )
            retrieval_details[name] = detail
            retrieval_summaries[name] = summary
            atomic_write_json(RESULT_DIR / f"{name}_detail.json", detail)

        BEST_RETRIEVER = max(
            retrieval_summaries,
            key=lambda name: (
                retrieval_summaries[name][CONFIG.primary_metric],
                retrieval_summaries[name]["mrr"],
            ),
        )
        atomic_write_json(RESULT_DIR / "retrieval_summary.json", retrieval_summaries)
        mark_phase_complete(
            RUN_DIR,
            "advanced_retrieval",
            {"best_retriever": BEST_RETRIEVER, **retrieval_summaries[BEST_RETRIEVER]},
        )
        display(pd.DataFrame(retrieval_summaries).T.sort_values([CONFIG.primary_metric, "mrr"], ascending=False))
        """
    ),
    code(
        """
        from src.rag_core import hits_as_rows

        def serialize_hits(hits):
            return [
                {
                    "chunk_id": hit.chunk_id,
                    "text": hit.text,
                    "metadata": hit.metadata,
                    "score": hit.score,
                    "rank": hit.rank,
                    "source": hit.source,
                }
                for hit in hits
            ]

        precomputed_hits = {}
        for row in qa_rows:
            question = row["question"]
            precomputed_hits[row["id"]] = {
                "basic": basic_search(question),
                "advanced": hybrid_search(question, use_reranker=True),
            }
        atomic_write_json(
            RESULT_DIR / "precomputed_evidence.json",
            {
                qid: {variant: serialize_hits(hits) for variant, hits in variants.items()}
                for qid, variants in precomputed_hits.items()
            },
        )
        display(pd.DataFrame(hits_as_rows(precomputed_hits[qa_rows[0]["id"]]["advanced"])))
        backup_checkpoint("after_retrieval")
        """
    ),
    md(
        """
        ## 7. Qwen 4-bit 생성 비교

        Qwen tokenizer의 chat template을 명시적으로 적용합니다. 검색 결과를 먼저 고정했으므로 embedding/reranker GPU 객체를 내린 뒤 generator를 로드합니다.
        """
    ),
    code(
        """
        import gc

        del embeddings, vector_stores, best_store, reranker_model, length_tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print("Retrieval GPU objects released.")
        """
    ),
    code(
        """
        from src.rag_core import answer_with_evidence, build_qwen_lcel_chain

        generation_chain = build_qwen_lcel_chain(model_id=CONFIG.generator_model)
        generation_rows = []
        for row in qa_rows:
            qid = row["id"]
            question = row["question"]
            variants = {
                "no_rag": [],
                "basic": precomputed_hits[qid]["basic"],
                "advanced": precomputed_hits[qid]["advanced"],
            }
            for variant, evidence in variants.items():
                result = answer_with_evidence(generation_chain, question, evidence)
                required = row.get("required_keywords") or []
                keyword_coverage = (
                    sum(keyword in result["answer"] for keyword in required) / len(required)
                    if required else 0.0
                )
                generation_rows.append({
                    "id": qid,
                    "variant": variant,
                    "scope": row["scope"],
                    "question": question,
                    "answer": result["answer"],
                    "keyword_coverage": keyword_coverage,
                    "refused_out_of_scope": (
                        "제공된 문서에서 확인되지 않습니다" in result["answer"]
                        if row["scope"] == "out_of_scope" else None
                    ),
                    "evidence_pages": [item["pdf_page"] for item in result["evidence"]],
                })
        generation_df = pd.DataFrame(generation_rows)
        generation_df.to_csv(RESULT_DIR / "generation_comparison.csv", index=False)
        display(generation_df[["id", "variant", "scope", "keyword_coverage", "refused_out_of_scope"]])
        """
    ),
    md("## 8. 결과 고정, 보고서, Drive backup, 완료 잠금"),
    code(
        """
        answer_auto_summary = (
            generation_df.groupby("variant", dropna=False)
            .agg(
                keyword_coverage=("keyword_coverage", "mean"),
                rows=("id", "count"),
            )
            .to_dict(orient="index")
        )
        out_scope = generation_df[generation_df["scope"] == "out_of_scope"]
        refusal_rates = (
            out_scope.groupby("variant")["refused_out_of_scope"].mean().to_dict()
            if len(out_scope) else {}
        )
        metrics = {
            "run_id": RUN["run_id"],
            "config_hash": RUN["config_hash"],
            "primary_metric": CONFIG.primary_metric,
            "best_chunk_config": BEST_CONFIG,
            "best_retriever": BEST_RETRIEVER,
            "dense_chunk_experiment": dense_summaries,
            "retrieval": retrieval_summaries,
            "answer_automatic_checks": answer_auto_summary,
            "out_of_scope_refusal_rate": refusal_rates,
            "manual_answer_rubric_required": True,
            "gold_set_sealed_before_index_selection": True,
            "sealed_test": False,
            "post_test_tuning_allowed": True,
        }
        metrics_path = RESULT_DIR / "metrics.json"
        atomic_write_json(metrics_path, metrics)

        summary_lines = [
            "# Mission 14 Kaggle 결과 요약",
            "",
            f"- Run ID: `{RUN['run_id']}`",
            f"- Best chunk config: `{BEST_CONFIG}`",
            f"- Best retriever: `{BEST_RETRIEVER}`",
            f"- Primary metric: `{CONFIG.primary_metric}` = "
            f"`{retrieval_summaries[BEST_RETRIEVER][CONFIG.primary_metric]:.4f}`",
            f"- MRR: `{retrieval_summaries[BEST_RETRIEVER]['mrr']:.4f}`",
            "- Official hidden test: 미사용",
            "- Manual answer rubric: 제출 전 별도 채점 필요",
        ]
        (RESULT_DIR / "stage_summary.md").write_text("\\n".join(summary_lines) + "\\n", encoding="utf-8")
        mark_phase_complete(RUN_DIR, "generation_and_reporting", {"rows": len(generation_df)})
        final_backup = backup_checkpoint("final")
        lock = seal_run(RUN_DIR, metrics_path)
        print(json.dumps(lock, ensure_ascii=False, indent=2))
        print("Final Drive backup:", final_backup)
        """
    ),
    md(
        """
        ## 제출 전 수동 평가

        `generation_comparison.csv`에 각 답변별로 정확성·근거 충실성·조건/예외·완전성(각 0~2), citation 정확성(0~1), 문서 밖 질문 거절(0~1)을 채점합니다. 자동 keyword coverage는 보조 지표이며 정답 판정이 아닙니다.

        Kaggle에서 Save Version을 완료한 뒤 `scripts/sync_kaggle_to_drive.py`로 실행 notebook을 기존 Google Drive file ID에 덮어써서 공유 링크를 유지합니다.
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "kaggle": {"accelerator": "gpu", "internet": True},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUTPUT)
