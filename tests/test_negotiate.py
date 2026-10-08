"""Accept negotiation helpers used by the Cloudflare Pages _worker.js."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "site" / "_worker.js"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

HARNESS = r"""
const workerUrl = process.argv[1];
const cases = JSON.parse(process.argv[2]);
const { preferredType, markdownPath, normalizePath } = await import(workerUrl);
const results = cases.map((c) => {
  if (c.fn === "preferredType") return preferredType(c.accept, c.produces);
  if (c.fn === "markdownPath") return markdownPath(c.path);
  if (c.fn === "normalizePath") return normalizePath(c.path);
  throw new Error(`unknown fn ${c.fn}`);
});
process.stdout.write(JSON.stringify(results));
"""


def _run(cases: list[dict]) -> list:
    # Pages requires the file to be named _worker.js; Node needs .mjs to parse ESM exports.
    with tempfile.TemporaryDirectory() as tmp:
        module = Path(tmp) / "worker.mjs"
        module.write_text(WORKER.read_text(encoding="utf-8"), encoding="utf-8")
        proc = subprocess.run(
            [NODE, "--input-type=module", "-e", HARNESS, module.resolve().as_uri(), json.dumps(cases)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            raise AssertionError(proc.stderr or proc.stdout)
        return json.loads(proc.stdout)


def test_preferred_type_vectors() -> None:
    produces = ["text/html", "text/markdown"]
    cases = [
        {"fn": "preferredType", "accept": "text/markdown", "produces": produces},
        {"fn": "preferredType", "accept": "text/markdown, text/html;q=0.8", "produces": produces},
        {"fn": "preferredType", "accept": "text/html", "produces": produces},
        {"fn": "preferredType", "accept": "text/markdown;q=0, text/html", "produces": produces},
        {"fn": "preferredType", "accept": "text/markdown;q=0, text/html;q=0", "produces": produces},
        {"fn": "preferredType", "accept": None, "produces": produces},
        {"fn": "preferredType", "accept": "*/*", "produces": produces},
        {
            "fn": "preferredType",
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "produces": produces,
        },
    ]
    got = _run(cases)
    assert got == [
        "text/markdown",
        "text/markdown",
        "text/html",
        "text/html",
        None,
        "text/html",
        "text/html",
        "text/html",
    ]


def test_markdown_paths() -> None:
    cases = [
        {"fn": "markdownPath", "path": "/"},
        {"fn": "markdownPath", "path": "/week3"},
        {"fn": "markdownPath", "path": "/week3/"},
        {"fn": "markdownPath", "path": "/about"},
        {"fn": "normalizePath", "path": "/privacy/"},
    ]
    assert _run(cases) == ["/index.md", "/week3.md", "/week3.md", "/about.md", "/privacy"]


def test_worker_ships_markdown_404_body() -> None:
    text = WORKER.read_text(encoding="utf-8")
    assert "Page not found" in text
    assert "llms.txt" in text and "sitemap.xml" in text
    assert len(text) > 20
