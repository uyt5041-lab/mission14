import copy
import unittest

from src.rag_core import (
    SearchHit,
    all_facts_covered_at_k,
    any_page_hit_at_k,
    chunk_fingerprint,
    clean_page_text,
    evidence_coverage_at_k,
    evaluate_retrieval,
    find_repeated_lines,
    make_chunk_id,
    page_hit_at_k,
    reciprocal_rank,
    tokenize_korean,
    validate_indexing_gate,
    weighted_rrf,
)


class DummyDocument:
    def __init__(self, text, metadata):
        self.page_content = text
        self.metadata = metadata


def hit(chunk_id: str, page: int, rank: int, source: str = "dense") -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        text=f"text {chunk_id}",
        metadata={"pdf_page": page, "section": "test"},
        score=1.0,
        rank=rank,
        source=source,
    )


class RagCoreTests(unittest.TestCase):
    def test_cleaning_removes_boundary_header_and_decorated_page_number(self):
        pages = [
            "국세청 연말정산\n총급여 8천만원 이하\n- 1 -",
            "국세청 연말정산\n공제율 15%\n- 2 -",
            "국세청 연말정산\n적용시기 2024.1.1.\n- 3 -",
        ]
        repeated = find_repeated_lines(pages, min_page_ratio=0.6, boundary_lines=1)
        cleaned = clean_page_text(pages[0], repeated, boundary_lines=1)
        self.assertNotIn("국세청 연말정산", cleaned)
        self.assertNotIn("- 1 -", cleaned)
        self.assertIn("총급여 8천만원 이하", cleaned)

    def test_cleaning_preserves_bare_tax_numbers(self):
        text = "공제율\n15\n거주자 기준\n183\n연간 한도\n1000"
        cleaned = clean_page_text(text, boundary_lines=2)
        self.assertEqual(cleaned.splitlines(), text.splitlines())

    def test_repeated_content_line_is_not_removed_from_page_middle(self):
        text = "머리말\n실제 조건\n공제율 15%\n실제 조건\n꼬리말"
        cleaned = clean_page_text(text, repeated_lines={"실제 조건"}, boundary_lines=1)
        self.assertEqual(cleaned.count("실제 조건"), 2)

    def test_tokenizer_preserves_korean_numbers_and_symbols(self):
        self.assertEqual(
            tokenize_korean("총급여 8,000만원·공제율 15% (§122)"),
            ["총급여", "8,000", "만원", "공제율", "15", "%", "§", "122"],
        )

    def test_chunk_id_distinguishes_body_and_table_on_same_page(self):
        body = {
            "pdf_page": 33,
            "content_type": "body",
            "document_id": "body-p0033-aaa",
        }
        table = {
            "pdf_page": 33,
            "content_type": "table",
            "document_id": "table-p0033-bbb",
        }
        self.assertNotEqual(make_chunk_id(body, "C2", 0), make_chunk_id(table, "C2", 0))

    def test_chunk_fingerprint_changes_when_citation_metadata_changes(self):
        original = DummyDocument(
            "same text",
            {"chunk_id": "x", "pdf_page": 10, "printed_page": 5, "section": "A"},
        )
        changed = DummyDocument("same text", copy.deepcopy(original.metadata))
        changed.metadata["printed_page"] = 6
        self.assertNotEqual(chunk_fingerprint([original]), chunk_fingerprint([changed]))

    def test_weighted_rrf_rewards_agreement(self):
        dense = [hit("a", 10, 1), hit("b", 20, 2)]
        bm25 = [hit("b", 20, 1, "bm25"), hit("c", 30, 2, "bm25")]
        fused = weighted_rrf(
            {"dense": dense, "bm25": bm25},
            {"dense": 0.7, "bm25": 0.3},
            top_n=3,
        )
        self.assertEqual(fused[0].chunk_id, "b")
        self.assertEqual([row.rank for row in fused], [1, 2, 3])

    def test_any_hit_and_all_fact_coverage_are_separate(self):
        hits = [hit("a", 10, 1), hit("b", 30, 2)]
        groups = [{10, 11}, {20}]
        self.assertEqual(any_page_hit_at_k(hits, groups, 2), 1)
        self.assertEqual(all_facts_covered_at_k(hits, groups, 2), 0)
        self.assertEqual(evidence_coverage_at_k(hits, groups, 2), 0.5)

    def test_evaluate_retrieval_reports_multi_fact_metrics(self):
        qa = [
            {
                "id": "QX",
                "scope": "in_scope",
                "gold_status": "verified",
                "question": "test",
                "gold_required_evidence": [
                    {"fact": "A", "acceptable_pages": [10]},
                    {"fact": "B", "acceptable_pages": [20]},
                ],
            }
        ]
        detail, summary = evaluate_retrieval(
            qa, lambda _: [hit("a", 10, 1), hit("b", 20, 2)], k=2
        )
        self.assertEqual(detail[0]["all_facts_covered@2"], 1)
        self.assertEqual(summary["evidence_coverage@2"], 1.0)

    def test_indexing_gate_fails_before_heavy_work(self):
        with self.assertRaisesRegex(RuntimeError, "metadata_rules.json is empty"):
            validate_indexing_gate([], [])
        gate = validate_indexing_gate(
            [{"start_pdf_page": 1, "end_pdf_page": 426}],
            [
                {
                    "id": "Q1",
                    "scope": "in_scope",
                    "gold_status": "verified",
                    "gold_required_evidence": [
                        {"fact": "answer", "acceptable_pages": [10]}
                    ],
                }
            ],
        )
        self.assertEqual(gate["verified_questions"], 1)

    def test_legacy_page_metrics_remain_available(self):
        hits = [hit("a", 3, 1), hit("b", 8, 2)]
        self.assertEqual(page_hit_at_k(hits, [8], 1), 0)
        self.assertEqual(page_hit_at_k(hits, [8], 2), 1)
        self.assertEqual(reciprocal_rank(hits, [8]), 0.5)


if __name__ == "__main__":
    unittest.main()
