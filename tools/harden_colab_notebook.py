#!/usr/bin/env python3
"""Apply the shared hard gate and current metric names to the legacy Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks" / "mission14_rag_colab.ipynb"
MARKER = "INDEXING STOP GATE (shared with Kaggle)"


def source(cell: dict) -> str:
    return "".join(cell.get("source", []))


notebook = json.loads(PATH.read_text(encoding="utf-8"))

for cell in notebook["cells"]:
    if cell.get("cell_type") != "code":
        continue
    text = source(cell)
    text = text.replace(
        "dense_summaries[name]['hit@5'], dense_summaries[name]['mrr']",
        "dense_summaries[name]['all_facts_covered@5'], dense_summaries[name]['mrr']",
    )
    cell["source"] = text.splitlines(keepends=True)

if not any(MARKER in source(cell) for cell in notebook["cells"]):
    insert_at = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if cell.get("cell_type") == "markdown"
        and "## 6. Token-aware chunking experiments" in source(cell)
    )
    gate_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": (
            "# INDEXING STOP GATE (shared with Kaggle)\n"
            "from src.rag_core import validate_indexing_gate\n"
            "\n"
            "gate_result = validate_indexing_gate(metadata_rules, qa_rows)\n"
            "print('INDEXING GATE PASSED:', gate_result)\n"
        ).splitlines(keepends=True),
    }
    notebook["cells"].insert(insert_at, gate_cell)

PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(PATH)
