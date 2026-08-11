import unittest

from src.rag_core import (
    SearchHit,
    clean_page_text,
    find_repeated_lines,
    page_hit_at_k,
    reciprocal_rank,
    tokenize_korean,
    weighted_rrf,
)


def hit(chunk_id: str, page: int, rank: int, source: str) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        text=f"text {chunk_id}",
        metadata={"pdf_page": page, "section": "test"},
        score=1.0,
        rank=rank,
        source=source,
    )


class RagCoreTests(unittest.TestCase):
    def test_repeated_line_cleaning_preserves_tax_symbols(self):
        pages = [
            "국세청 연말정산\n총급여 8천만원 이하\n- 1 -",
            "국세청 연말정산\n공제율 15%\n- 2 -",
            "국세청 연말정산\n적용시기 2024.1.1.\n- 3 -",
        ]
        repeated = find_repeated_lines(pages, min_page_ratio=0.6)
        cleaned = clean_page_text(pages[0], repeated)
        self.assertNotIn("국세청 연말정산", cleaned)
        self.assertIn("총급여 8천만원 이하", cleaned)

    def test_tokenizer_preserves_korean_numbers_and_symbols(self):
        self.assertEqual(
            tokenize_korean("총급여 8,000만원·공제율 15% (§122)"),
            ["총급여", "8,000", "만원", "공제율", "15", "%", "§", "122"],
        )

    def test_weighted_rrf_rewards_agreement(self):
        dense = [hit("a", 10, 1, "dense"), hit("b", 20, 2, "dense")]
        bm25 = [hit("b", 20, 1, "bm25"), hit("c", 30, 2, "bm25")]
        fused = weighted_rrf(
            {"dense": dense, "bm25": bm25},
            {"dense": 0.7, "bm25": 0.3},
            top_n=3,
        )
        self.assertEqual(fused[0].chunk_id, "b")
        self.assertEqual([row.rank for row in fused], [1, 2, 3])

    def test_page_metrics(self):
        hits = [hit("a", 3, 1, "dense"), hit("b", 8, 2, "dense")]
        self.assertEqual(page_hit_at_k(hits, [8], 1), 0)
        self.assertEqual(page_hit_at_k(hits, [8], 2), 1)
        self.assertEqual(reciprocal_rank(hits, [8]), 0.5)


if __name__ == "__main__":
    unittest.main()
