"""The committed docs/week05 files are what the generator makes from the committed live results."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from week05.docs import LIVE, generate

DOCS = Path(__file__).resolve().parents[1] / "docs" / "week05"


@pytest.mark.skipif(not all((DOCS / name).is_file() for name in LIVE), reason="no live results")
def test_generated_docs_are_up_to_date(tmp_path):
    out = tmp_path / "week05"
    shutil.copytree(DOCS, out)
    generate(out)
    for name in ("wire.json", "summary.json", "RESULTS.md"):
        assert (out / name).read_text() == (DOCS / name).read_text(), f"{name} is stale: run python -m week05 docs"


def test_no_dashes_in_docs():
    for path in DOCS.glob("*.md"):
        text = path.read_text()
        assert chr(0x2013) not in text and chr(0x2014) not in text, path.name
