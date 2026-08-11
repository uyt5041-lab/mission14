"""Core functions for the Mission 14 tax-guide RAG notebook.

Heavy ML dependencies are imported lazily so cleaning, fusion, and metric tests can
run without a GPU or model download.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


OFFICIAL_PDF_URL = (
    "https://nts.go.kr/comm/nttFileDownload.do?"
    "fileKey=b6719de58a99ebeb6555a0d4f0e7aeda"
)

CHUNK_CONFIGS = {
    "C1": {"chunk_size": 300, "chunk_overlap": 50},
    "C2": {"chunk_size": 500, "chunk_overlap": 80},
    "C3": {"chunk_size": 800, "chunk_overlap": 120},
}

KOREAN_SEPARATORS = [
    "\n\n",
    "\n○ ",
    "\n□ ",
    "\n- ",
    "\n",
    ". ",
    " ",
    "",
]


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float
    rank: int = 0
    source: str = ""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_pdf(path: str | Path, min_pages: int = 400, max_pages: int = 450) -> dict[str, Any]:
    """Validate signature and page count, then return reproducibility metadata."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as stream:
        signature = stream.read(4)
    if signature != b"%PDF":
        raise ValueError(f"Not a PDF file: {path}")

    import fitz

    with fitz.open(path) as pdf:
        page_count = pdf.page_count
    if not min_pages <= page_count <= max_pages:
        raise ValueError(
            f"Unexpected page count {page_count}; expected {min_pages}..{max_pages}. "
            "Confirm that the Mission 14 source file is being used."
        )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "page_count": page_count,
    }


def normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def find_repeated_lines(
    page_texts: Sequence[str],
    min_page_ratio: float = 0.18,
    min_length: int = 4,
) -> set[str]:
    """Find probable headers/footers using page frequency, not raw occurrence count."""
    if not page_texts:
        return set()
    page_presence: Counter[str] = Counter()
    for text in page_texts:
        unique = {
            normalize_line(line)
            for line in text.splitlines()
            if len(normalize_line(line)) >= min_length
        }
        page_presence.update(unique)
    threshold = max(2, math.ceil(len(page_texts) * min_page_ratio))
    return {line for line, count in page_presence.items() if count >= threshold}


def clean_page_text(text: str, repeated_lines: Iterable[str] = ()) -> str:
    """Conservatively remove repeated lines while preserving tax punctuation and hierarchy."""
    repeated = {normalize_line(line) for line in repeated_lines}
    kept: list[str] = []
    for raw_line in text.replace("\u00a0", " ").splitlines():
        line = normalize_line(raw_line)
        if not line or line in repeated:
            continue
        if re.fullmatch(r"[-–—]?\s*\d{1,3}\s*[-–—]?", line):
            continue
        if re.fullmatch(r"https?://\S+", line):
            continue
        kept.append(line)
    cleaned = "\n".join(kept)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def load_and_clean_pdf(pdf_path: str | Path) -> tuple[list[Any], set[str]]:
    """Load one LangChain Document per PDF page and apply conservative cleaning."""
    from langchain_community.document_loaders import PyMuPDFLoader

    pages = PyMuPDFLoader(str(pdf_path)).load()
    repeated = find_repeated_lines([doc.page_content for doc in pages])
    cleaned: list[Any] = []
    for zero_based_page, doc in enumerate(pages):
        doc.page_content = clean_page_text(doc.page_content, repeated)
        doc.metadata.update(
            {
                "source": Path(pdf_path).name,
                "pdf_page": zero_based_page + 1,
                "printed_page": -1,
                "part": doc.metadata.get("part", "미분류"),
                "section": doc.metadata.get("section", "미분류"),
                "content_type": "body",
            }
        )
        cleaned.append(doc)
    return cleaned, repeated


def apply_metadata_rules(
    documents: Sequence[Any], rules: Sequence[Mapping[str, Any]]
) -> list[Any]:
    """Apply manually verified page ranges without guessing document structure.

    A rule contains start_pdf_page, end_pdf_page, part, section, and optionally
    printed_page_offset. The printed page is pdf_page + offset.
    """
    for doc in documents:
        pdf_page = int(doc.metadata.get("pdf_page", -1))
        for rule in rules:
            start = int(rule["start_pdf_page"])
            end = int(rule["end_pdf_page"])
            if start <= pdf_page <= end:
                doc.metadata["part"] = str(rule.get("part", "미분류"))
                doc.metadata["section"] = str(rule.get("section", "미분류"))
                offset = rule.get("printed_page_offset")
                if offset is not None:
                    doc.metadata["printed_page"] = pdf_page + int(offset)
                break
    return list(documents)


def make_table_document(
    *,
    item: str,
    previous: str,
    revised: str,
    effective_from: str,
    pdf_page: int,
    section: str,
    part: str = "2024년 귀속 연말정산 개정세법 요약",
) -> Any:
    """Create a structure-preserving LangChain Document for a manually checked table."""
    from langchain_core.documents import Document

    text = (
        f"[항목] {item}\n"
        f"[종전] {previous}\n"
        f"[개정] {revised}\n"
        f"[적용시기] {effective_from}"
    )
    return Document(
        page_content=text,
        metadata={
            "source": "2024_year_end_tax_guide.pdf",
            "pdf_page": pdf_page,
            "printed_page": -1,
            "part": part,
            "section": section,
            "content_type": "table",
        },
    )


def chunk_documents(
    documents: Sequence[Any],
    config_name: str = "C2",
    tokenizer_name: str = "nlpai-lab/KURE-v1",
) -> list[Any]:
    """Token-aware recursive splitting with a searchable section prefix."""
    if config_name not in CHUNK_CONFIGS:
        raise KeyError(f"Unknown config {config_name}; choose from {sorted(CHUNK_CONFIGS)}")

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        separators=KOREAN_SEPARATORS,
        **CHUNK_CONFIGS[config_name],
    )

    chunks: list[Any] = []
    for doc in documents:
        split_docs = splitter.split_documents([doc])
        for index, chunk in enumerate(split_docs):
            page = int(chunk.metadata.get("pdf_page", -1))
            section = str(chunk.metadata.get("section", "미분류"))
            chunk_id = f"p{page:03d}_{config_name.lower()}_{index:03d}"
            chunk.page_content = (
                f"[문서 구간]\n{section}\n\n[본문]\n{chunk.page_content.strip()}"
            )
            chunk.metadata.update(
                {
                    "chunk_id": chunk_id,
                    "chunk_config": config_name,
                    "chunk_index": index,
                }
            )
            chunks.append(chunk)
    return chunks


def build_embeddings(model_name: str = "nlpai-lab/KURE-v1", device: str = "cuda") -> Any:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
    )


def open_or_build_chroma(
    chunks: Sequence[Any],
    persist_directory: str | Path,
    collection_name: str,
    embeddings: Any,
) -> Any:
    from langchain_chroma import Chroma

    directory = Path(persist_directory)
    fingerprint = hashlib.sha256()
    for chunk in chunks:
        fingerprint.update(str(chunk.metadata.get("chunk_id", "")).encode("utf-8"))
        fingerprint.update(b"\0")
        fingerprint.update(chunk.page_content.encode("utf-8"))
        fingerprint.update(b"\0")
    expected_manifest = {
        "collection_name": collection_name,
        "embedding_model": str(getattr(embeddings, "model_name", "unknown")),
        "chunk_count": len(chunks),
        "chunk_fingerprint": fingerprint.hexdigest(),
    }
    manifest_path = directory / "index_manifest.json"
    if (directory / "chroma.sqlite3").exists():
        if not manifest_path.exists():
            raise RuntimeError(
                f"Existing index has no manifest: {directory}. "
                "Move it aside and rebuild to avoid stale results."
            )
        actual_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if actual_manifest != expected_manifest:
            raise RuntimeError(
                f"Index manifest mismatch: {directory}. "
                "The PDF/chunks/model changed; use a new index directory."
            )
        return Chroma(
            collection_name=collection_name,
            persist_directory=str(directory),
            embedding_function=embeddings,
        )
    directory.mkdir(parents=True, exist_ok=True)
    store = Chroma.from_documents(
        documents=list(chunks),
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(directory),
        collection_metadata={"hnsw:space": "cosine"},
    )
    manifest_path.write_text(
        json.dumps(expected_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return store


def dense_search(vector_store: Any, query: str, k: int = 5) -> list[SearchHit]:
    pairs = vector_store.similarity_search_with_relevance_scores(query, k=k)
    hits: list[SearchHit] = []
    for rank, (doc, score) in enumerate(pairs, start=1):
        hits.append(
            SearchHit(
                chunk_id=str(doc.metadata["chunk_id"]),
                text=doc.page_content,
                metadata=dict(doc.metadata),
                score=float(score),
                rank=rank,
                source="dense",
            )
        )
    return hits


def tokenize_korean(text: str) -> list[str]:
    """A deterministic BM25 tokenizer preserving Korean, numbers, %, and section marks."""
    return re.findall(r"[가-힣A-Za-z]+|\d+(?:[,.]\d+)*|[%§]", text.lower())


class BM25Index:
    def __init__(self, chunks: Sequence[Any]):
        from rank_bm25 import BM25Okapi

        self.chunks = list(chunks)
        self.corpus_tokens = [tokenize_korean(doc.page_content) for doc in self.chunks]
        self.model = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, k: int = 15) -> list[SearchHit]:
        scores = self.model.get_scores(tokenize_korean(query))
        order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:k]
        return [
            SearchHit(
                chunk_id=str(self.chunks[i].metadata["chunk_id"]),
                text=self.chunks[i].page_content,
                metadata=dict(self.chunks[i].metadata),
                score=float(scores[i]),
                rank=rank,
                source="bm25",
            )
            for rank, i in enumerate(order, start=1)
        ]


def weighted_rrf(
    ranked_lists: Mapping[str, Sequence[SearchHit]],
    weights: Mapping[str, float],
    rrf_k: int = 60,
    top_n: int = 12,
) -> list[SearchHit]:
    """Fuse incomparable dense/BM25 scores by weighted rank position."""
    totals: Counter[str] = Counter()
    exemplar: dict[str, SearchHit] = {}
    for source, hits in ranked_lists.items():
        weight = float(weights.get(source, 0.0))
        for fallback_rank, hit in enumerate(hits, start=1):
            rank = hit.rank or fallback_rank
            totals[hit.chunk_id] += weight / (rrf_k + rank)
            exemplar.setdefault(hit.chunk_id, hit)

    ordered_ids = sorted(totals, key=totals.get, reverse=True)[:top_n]
    return [
        SearchHit(
            chunk_id=chunk_id,
            text=exemplar[chunk_id].text,
            metadata=exemplar[chunk_id].metadata,
            score=float(totals[chunk_id]),
            rank=rank,
            source="rrf",
        )
        for rank, chunk_id in enumerate(ordered_ids, start=1)
    ]


def rerank_hits(
    query: str,
    hits: Sequence[SearchHit],
    model_name: str = "BAAI/bge-reranker-v2-m3",
    top_n: int = 5,
    device: str = "cuda",
    model: Any | None = None,
) -> list[SearchHit]:
    if model is None:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(model_name, device=device)
    scores = model.predict([(query, hit.text) for hit in hits])
    ordered = sorted(zip(hits, scores), key=lambda pair: float(pair[1]), reverse=True)[:top_n]
    return [
        SearchHit(
            chunk_id=hit.chunk_id,
            text=hit.text,
            metadata=hit.metadata,
            score=float(score),
            rank=rank,
            source="reranker",
        )
        for rank, (hit, score) in enumerate(ordered, start=1)
    ]


def page_hit_at_k(hits: Sequence[SearchHit], gold_pages: Sequence[int], k: int) -> int:
    gold = {int(page) for page in gold_pages}
    return int(any(int(hit.metadata.get("pdf_page", -1)) in gold for hit in hits[:k]))


def reciprocal_rank(hits: Sequence[SearchHit], gold_pages: Sequence[int]) -> float:
    gold = {int(page) for page in gold_pages}
    for rank, hit in enumerate(hits, start=1):
        if int(hit.metadata.get("pdf_page", -1)) in gold:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    qa_rows: Sequence[Mapping[str, Any]],
    retrieve_fn: Any,
    k: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Evaluate only rows whose gold pages have already been manually verified."""
    results: list[dict[str, Any]] = []
    for row in qa_rows:
        gold_pages = row.get("gold_pdf_pages") or []
        if not gold_pages or row.get("gold_status") != "verified":
            continue
        hits = retrieve_fn(str(row["question"]))
        results.append(
            {
                "id": row["id"],
                "hit@1": page_hit_at_k(hits, gold_pages, 1),
                f"hit@{k}": page_hit_at_k(hits, gold_pages, k),
                "mrr": reciprocal_rank(hits, gold_pages),
                "retrieved_pages": [hit.metadata.get("pdf_page") for hit in hits[:k]],
            }
        )
    if not results:
        raise ValueError("No verified gold pages. Verify the evaluation set before scoring.")
    summary = {
        key: sum(float(row[key]) for row in results) / len(results)
        for key in ("hit@1", f"hit@{k}", "mrr")
    }
    return results, summary


def hits_as_rows(hits: Sequence[SearchHit], snippet_chars: int = 180) -> list[dict[str, Any]]:
    return [
        {
            "rank": hit.rank,
            "pdf_page": hit.metadata.get("pdf_page"),
            "section": hit.metadata.get("section"),
            "score": round(hit.score, 6),
            "source": hit.source,
            "chunk_id": hit.chunk_id,
            "snippet": re.sub(r"\s+", " ", hit.text)[:snippet_chars],
        }
        for hit in hits
    ]


def format_context(hits: Sequence[SearchHit]) -> str:
    blocks = []
    for hit in hits:
        blocks.append(
            "\n".join(
                [
                    f"[근거 {hit.rank}]",
                    f"PDF page: {hit.metadata.get('pdf_page')}",
                    f"Section: {hit.metadata.get('section', '미분류')}",
                    hit.text,
                ]
            )
        )
    return "\n\n".join(blocks)


RAG_PROMPT = """당신은 국세청의 「2024년 귀속 연말정산 신고안내」만을 근거로 답변하는 도우미입니다.

규칙:
1. 아래 문서 근거만 사용하세요.
2. 문서에 없는 내용은 추측하지 말고 "제공된 문서에서 확인되지 않습니다."라고 답하세요.
3. 금액, 비율, 소득 기준, 적용 시기, 예외를 구분하세요.
4. 종전 내용과 개정 내용을 혼동하지 마세요.
5. 마지막에 사용한 PDF page를 적으세요.
6. 근거가 충돌하면 충돌 사실을 설명하고 적용 시기가 명확한 근거를 우선하세요.

Context:
{context}

Question:
{question}

답변:
"""


def build_qwen_lcel_chain(
    model_id: str = "Qwen/Qwen3-4B-Instruct-2507",
    max_new_tokens: int = 300,
) -> Any:
    """Load a deterministic 4-bit Qwen pipeline and wrap it in an LCEL chain."""
    import torch
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_huggingface import HuggingFacePipeline
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        pipeline,
    )

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        return_full_text=False,
    )
    llm = HuggingFacePipeline(pipeline=generator)
    return PromptTemplate.from_template(RAG_PROMPT) | llm | StrOutputParser()


def answer_with_evidence(chain: Any, question: str, hits: Sequence[SearchHit]) -> dict[str, Any]:
    answer = chain.invoke({"context": format_context(hits), "question": question})
    return {"question": question, "evidence": hits_as_rows(hits), "answer": answer}
