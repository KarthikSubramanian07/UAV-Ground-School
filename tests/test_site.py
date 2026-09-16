"""The showcase site builds, links resolve, and the browser demo matches Python."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "week02"
SCRIPT = ROOT / "scripts" / "build_site.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_site", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["build_site"] = module
    spec.loader.exec_module(module)
    return module


build_site = _load_builder()

pytestmark = pytest.mark.skipif(
    not (DOCS / "benchmark.json").is_file(), reason="docs/week02 has no benchmark output yet (run python -m week02 benchmark docs/week02)"
)


@pytest.fixture(scope="module")
def site(tmp_path_factory) -> Path:
    return build_site.build(tmp_path_factory.mktemp("site") / "site")


@pytest.fixture(scope="module")
def index_html(site: Path) -> str:
    return (site / "index.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return json.loads((DOCS / "benchmark.json").read_text())


class _Head(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.meta: dict[str, str] = {}
        self.links: dict[str, str] = {}
        self.json_ld: list[str] = []
        self._in_ld = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "meta" and "content" in a:
            self.meta[a.get("property") or a.get("name") or ""] = a["content"]
        elif tag == "link" and "rel" in a:
            self.links[a["rel"]] = a.get("href", "")
        elif tag == "script" and a.get("type") == "application/ld+json":
            self._in_ld = True

    def handle_endtag(self, tag):
        self._in_ld = False if tag == "script" else self._in_ld

    def handle_data(self, data):
        if self._in_ld:
            self.json_ld.append(data)


def test_required_files_exist(site: Path) -> None:
    required = [
        "index.html",
        "404.html",
        "og.jpg",
        "favicon.svg",
        "robots.txt",
        "sitemap.xml",
        "css/site.css",
        "js/hsv-segment.js",
        "js/demo.js",
        "js/site.js",
        *(f"assets/demo/{name}.jpg" for name in build_site.DEMO_IMAGES),
        *(f"assets/img/{name}-{min(1600, cv2.imread(str(DOCS / f'{name}.jpg')).shape[1])}.webp" for name in build_site.STITCH_IMAGES),
    ]
    missing = [path for path in required if not (site / path).is_file()]
    assert not missing


def test_images_are_resized_and_og_card_is_1200_by_630(site: Path) -> None:
    for path in (site / "assets" / "img").glob("*.webp"):
        image = cv2.imread(str(path))
        assert image is not None and image.shape[1] <= 1600, path
    og = cv2.imread(str(site / "og.jpg"))
    assert og.shape == (630, 1200, 3)


def test_demo_images_are_byte_identical_to_docs(site: Path) -> None:
    for name in build_site.DEMO_IMAGES:
        assert (site / "assets" / "demo" / f"{name}.jpg").read_bytes() == (DOCS / "colors" / f"{name}.jpg").read_bytes()


def test_no_em_or_en_dashes(site: Path) -> None:
    offenders = [
        str(path.relative_to(site))
        for path in site.rglob("*")
        if path.suffix in {".html", ".js", ".css", ".svg", ".txt", ".xml"} and re.search("[\u2013\u2014]", path.read_text(encoding="utf-8"))
    ]
    assert not offenders


def test_local_references_resolve_and_are_relative(site: Path) -> None:
    checked = 0
    for path in site.rglob("*"):
        if path.suffix not in {".html", ".css", ".js"}:
            continue
        base = site if path.suffix == ".js" else path.parent
        for ref in build_site.local_references(path):
            assert not ref.startswith("/"), f"{path.name}: {ref} is root absolute"
            clean = ref.split("#")[0].split("?")[0]
            target = site / "index.html" if clean in ("", "./") else base / clean
            assert target.exists(), f"{path.name}: {ref} does not resolve"
            checked += 1
    assert checked > 20


def test_no_template_tokens_left(site: Path) -> None:
    for path in site.glob("*.html"):
        assert not build_site.TOKEN.search(path.read_text(encoding="utf-8")), path.name


def test_seo_metadata(site: Path, index_html: str) -> None:
    head = _Head()
    head.feed(index_html)
    canonical = build_site.CANONICAL
    assert head.links["canonical"] == canonical
    for key in ("description", "og:title", "og:description", "og:type", "twitter:card", "twitter:title", "twitter:image"):
        assert head.meta.get(key), key
    assert head.meta["og:url"] == canonical
    assert head.meta["og:image"] == canonical + "og.jpg"
    assert head.meta["twitter:card"] == "summary_large_image"
    ld = json.loads("".join(head.json_ld))
    assert ld["@type"] == "SoftwareSourceCode" and ld["codeRepository"].startswith("https://github.com/")
    assert f"Sitemap: {canonical}sitemap.xml" in (site / "robots.txt").read_text()
    assert f"<loc>{canonical}</loc>" in (site / "sitemap.xml").read_text()
    assert canonical in (site / "404.html").read_text()


def test_benchmark_numbers_come_from_json(index_html: str, rows: list[dict]) -> None:
    for row in rows:
        assert row["variant"] in index_html
        assert f'<td class="num">{row["seconds"]:.1f}</td>' in index_html
        if "rmse_px" in row:
            assert f'<td class="num">{row["rmse_px"]:.2f}</td>' in index_html
            assert f'<td class="num">{row["max_px"]:.2f}</td>' in index_html
            if row.get("zncc") is not None:
                assert f'<td class="num">{row["zncc"]:.3f}</td>' in index_html
    rough = [r for r in rows if r["scenario"] == "rough" and "rmse_px" in r]
    assert f"{min(r['rmse_px'] for r in rough):.2f} px pose RMSE" in index_html


def test_reference_reports_are_embedded(index_html: str) -> None:
    match = re.search(r'<script type="application/json" id="reference-reports">(.*?)</script>', index_html, re.S)
    assert match
    reports = json.loads(match.group(1))
    assert set(reports) == set(build_site.DEMO_IMAGES)
    assert any(obj["shape"] == "octagon" for layer in reports["targets"]["layers"] for obj in layer["objects"])


def test_missing_inputs_fail_loudly(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    shutil.copytree(DOCS, docs, ignore=shutil.ignore_patterns("calm_*", "*.png"))
    (docs / "benchmark.json").unlink()
    with pytest.raises(build_site.BuildError) as error:
        build_site.build(tmp_path / "out", docs)
    message = str(error.value)
    assert "benchmark.json" in message and "calm_panorama.jpg" in message and "calm_cv2_stitcher.jpg" in message


def test_refuses_to_overwrite_the_repository() -> None:
    with pytest.raises(build_site.BuildError):
        build_site.build(ROOT)


def test_report_parser_round_trips_format_report() -> None:
    from week02 import colors

    image = colors.load_image(DOCS / "colors" / "stop_sign.jpg")
    layers = colors.split_colors(image)
    parsed = build_site.parse_report(colors.format_report(layers))
    assert [layer["name"] for layer in parsed["layers"]] == [layer.name for layer in layers]
    for got, want in zip(parsed["layers"], layers):
        assert len(got["objects"]) == len(want.objects)
        for obj, ref in zip(got["objects"], want.objects):
            assert obj["center"] == pytest.approx(ref.center, abs=0.051)
            assert obj["shape"] == ref.shape


# ------------------------------------------------ JavaScript port parity ----

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")

SPLIT_HARNESS = """
const fs = require("fs");
const S = require(process.argv[2]);
const [file, w, h] = process.argv.slice(3);
const res = S.splitColors(fs.readFileSync(file), +w, +h);
console.log(JSON.stringify(res.layers.map((l) => ({
  name: l.name, pixels: l.pixels, center: l.center,
  objects: l.objects.map((o) => ({ center: o.center, area: o.area, bbox: o.bbox, touches: o.touchesBorder })),
}))));
"""

HSV_HARNESS = """
const fs = require("fs");
const S = require(process.argv[2]);
const ref = fs.readFileSync(process.argv[3]);
let bad = 0;
for (let r = 0, i = 0; r < 256; r++) for (let g = 0; g < 256; g++) for (let b = 0; b < 256; b++, i += 3) {
  const [h, s, v] = S.rgbToHsv(r, g, b);
  if (h !== ref[i] || s !== ref[i + 1] || v !== ref[i + 2]) bad++;
}
console.log(bad);
"""


def _node(tmp_path: Path, source: str, *args: str) -> str:
    script = tmp_path / "harness.js"
    script.write_text(source)
    js = str(ROOT / "site" / "js" / "hsv-segment.js")
    return subprocess.run([NODE, str(script), js, *args], check=True, capture_output=True, text=True, timeout=120).stdout


@needs_node
def test_js_hsv_matches_opencv_on_every_rgb_color(tmp_path: Path) -> None:
    v = np.arange(256, dtype=np.uint8)
    r, g, b = np.meshgrid(v, v, v, indexing="ij")
    bgr = np.stack([b, g, r], axis=-1).reshape(4096, 4096, 3)
    hsv = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2HSV)
    path = tmp_path / "hsv.bin"
    hsv.tofile(path)
    assert _node(tmp_path, HSV_HARNESS, str(path)).strip() == "0"


@needs_node
@pytest.mark.parametrize("name", ["targets", "stop_sign", "apple"])
def test_js_split_colors_matches_python_exactly(tmp_path: Path, name: str) -> None:
    from week02 import colors

    image = colors.load_image(DOCS / "colors" / f"{name}.jpg")
    rgba = tmp_path / "image.rgba"
    cv2.cvtColor(image, cv2.COLOR_BGR2RGBA).tofile(rgba)
    got = json.loads(_node(tmp_path, SPLIT_HARNESS, str(rgba), str(image.shape[1]), str(image.shape[0])))
    want = colors.split_colors(image, classify_shapes=False)

    assert [layer["name"] for layer in got] == [layer.name for layer in want]
    for js, py in zip(got, want):
        assert js["pixels"] == py.pixels
        assert js["center"] == pytest.approx(py.center, abs=1e-6)
        assert len(js["objects"]) == len(py.objects)
        for a, b in zip(js["objects"], py.objects):
            assert a["area"] == b.area
            assert tuple(a["bbox"]) == b.bbox
            assert a["touches"] == b.touches_border
            assert a["center"] == pytest.approx(b.center, abs=1e-6)
