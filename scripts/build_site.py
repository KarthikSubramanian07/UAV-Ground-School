"""Build the static showcase site into a directory ready for Cloudflare Pages.

    PYTHONPATH=. python scripts/build_site.py --out build/site

Steps: copy ``site/``, re-encode the committed images from ``docs/week02`` as
WebP (800 and 1600 px wide), copy the Color Me Impressed test images for the
in-browser demo, render the benchmark table from ``benchmark.json``, write
``og.jpg``, ``robots.txt``, ``sitemap.xml`` and the Cloudflare ``_headers`` file, then verify the result (no
unreplaced tokens, no em or en dashes, every local reference resolves).

Every input is checked up front and the build fails with a list of whatever is
missing, so a stale or partial ``docs/week02`` never ships silently.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
DOCS = ROOT / "docs" / "week02"
CANONICAL = "https://uav-ground-school.pages.dev/"
REPO = "https://github.com/KarthikSubramanian07/UAV-Ground-School"
DESCRIPTION = (
    "UAVs@Berkeley Ground School Week 2, solved: exact HSV color segmentation with object centers, SUAS shape "
    "classification, and a from scratch drone video mosaicking pipeline with loop closure and bundle adjustment, "
    "in Python and C++."
)

STITCH_IMAGES = ("rough_world", "rough_frame", "rough_sequential", "rough_panorama", "rough_path", "calm_panorama", "calm_cv2_stitcher")
DEMO_IMAGES = ("targets", "stop_sign", "apple")
IMAGE_WIDTHS = (800, 1600)
WEBP_QUALITY = 82
DASHES = re.compile("[\u2013\u2014]")
TOKEN = re.compile(r"\{\{([A-Z_]+)(?::([^}]*))?\}\}")
TEXT_SUFFIXES = {".html", ".css", ".js", ".svg", ".txt", ".xml", ".json"}

SCENARIO_TITLES = {"calm": "Calm flight", "rough": "Rough flight"}
SCENARIO_FALLBACK = {
    "calm": "3 passes, light noise, no lens distortion",
    "rough": "5 passes, barrel distortion k1=-0.06, motion blur, heavy noise, +/-18% exposure, +/-5% altitude",
}


class BuildError(RuntimeError):
    """Raised for anything that should stop a deploy."""


# ------------------------------------------------------------------ inputs ----


def check_inputs(docs: Path) -> dict:
    """Fail with one message listing every missing or malformed input."""
    missing = [str(p) for p in required_inputs(docs) if not p.is_file()]
    if not SITE.is_dir():
        missing.append(str(SITE))
    if missing:
        raise BuildError("missing build inputs:\n  " + "\n  ".join(missing))
    try:
        rows = json.loads((docs / "benchmark.json").read_text())
    except json.JSONDecodeError as error:
        raise BuildError(f"{docs / 'benchmark.json'} is not valid JSON: {error}") from error
    return {"rows": validate_rows(rows)}


def required_inputs(docs: Path) -> list[Path]:
    paths = [docs / "benchmark.json"]
    paths += [docs / f"{name}.jpg" for name in STITCH_IMAGES]
    for name in DEMO_IMAGES:
        paths += [docs / "colors" / f"{name}.jpg", docs / "colors" / f"{name}_report.txt"]
    return paths


def validate_rows(rows: object) -> list[dict]:
    if not isinstance(rows, list) or not rows:
        raise BuildError("benchmark.json must be a non empty list of rows")
    for scenario in ("calm", "rough"):
        scored = [r for r in rows if r.get("scenario") == scenario and "rmse_px" in r]
        if not scored:
            raise BuildError(f"benchmark.json has no scored rows for the {scenario} flight")
        if not any(r["variant"].startswith("sequential") for r in scored):
            raise BuildError(f"benchmark.json has no sequential chaining row for the {scenario} flight")
        for row in scored:
            for key in ("variant", "keyframes", "loop_closures", "rmse_px", "max_px", "seconds"):
                if key not in row:
                    raise BuildError(f"benchmark.json row {row!r} is missing {key!r}")
    if stitcher_row(rows) is None:
        raise BuildError("benchmark.json has no cv2.Stitcher row for the calm flight")
    return rows


# --------------------------------------------------------------- benchmark ----


def scored(rows: list[dict], scenario: str) -> list[dict]:
    return [r for r in rows if r.get("scenario") == scenario and "rmse_px" in r]


def sequential_row(rows: list[dict], scenario: str) -> dict:
    return next(r for r in scored(rows, scenario) if r["variant"].startswith("sequential"))


def best_row(rows: list[dict], scenario: str) -> dict:
    return min(scored(rows, scenario), key=lambda r: (r["rmse_px"], -(r.get("zncc") or 0.0)))


def final_row(rows: list[dict], scenario: str) -> dict:
    """The run whose panorama was saved: the last variant of the scenario."""
    return scored(rows, scenario)[-1]


def stitcher_row(rows: list[dict]) -> dict | None:
    return next((r for r in rows if r.get("scenario") == "calm" and "status" in r), None)


def stitcher_frames(row: dict) -> str:
    match = re.search(r"\((\d+) frames\)", row["variant"])
    return match.group(1) if match else "the sampled"


def fmt_px(value: float) -> str:
    return f"{value:.2f}"


def fmt_zncc(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def fmt_seconds(value: float) -> str:
    return f"{value:.1f}"


def scenario_descriptions() -> dict[str, str]:
    try:
        from week02 import benchmark

        return {s.name: s.description for s in benchmark.scenarios(seed=42)}
    except Exception:  # the site must still build without the package importable
        return dict(SCENARIO_FALLBACK)


def benchmark_rows(rows: list[dict]) -> str:
    descriptions = scenario_descriptions()
    order = list(dict.fromkeys(r["scenario"] for r in rows))
    out = []
    for scenario in order:
        best = best_row(rows, scenario) if scored(rows, scenario) else None
        title = SCENARIO_TITLES.get(scenario, scenario.title())
        desc = descriptions.get(scenario, SCENARIO_FALLBACK.get(scenario, ""))
        out.append("            <tbody>")
        out.append(f'              <tr><th scope="rowgroup" colspan="7">{html.escape(title)}<small>{html.escape(desc)}</small></th></tr>')
        for row in (r for r in rows if r["scenario"] == scenario):
            variant = html.escape(row["variant"])
            if "rmse_px" in row:
                css = ' class="best"' if row is best else ""
                cells = [
                    f'<td class="variant">{variant}</td>',
                    f'<td class="num">{row["keyframes"]}</td>',
                    f'<td class="num">{row["loop_closures"]}</td>',
                    f'<td class="num">{fmt_px(row["rmse_px"])}</td>',
                    f'<td class="num">{fmt_px(row["max_px"])}</td>',
                    f'<td class="num">{fmt_zncc(row.get("zncc"))}</td>',
                    f'<td class="num">{fmt_seconds(row["seconds"])}</td>',
                ]
            else:
                css = ' class="baseline"'
                status = html.escape(str(row.get("status", "")))
                cells = [
                    f'<td class="variant"><code>{variant}</code></td>',
                    f'<td colspan="3">{status}, no poses to score</td>',
                    '<td class="num"></td>',
                    '<td class="num"></td>',
                    f'<td class="num">{fmt_seconds(row["seconds"])}</td>',
                ]
            out.append(f"              <tr{css}>" + "".join(cells) + "</tr>")
        out.append("            </tbody>")
    return "\n".join(out)


def stats(rows: list[dict]) -> dict[str, str]:
    seq, best, final = sequential_row(rows, "rough"), best_row(rows, "rough"), final_row(rows, "rough")
    calm_final = final_row(rows, "calm")
    return {
        "rough_seq_rmse": fmt_px(seq["rmse_px"]),
        "rough_seq_max": fmt_px(seq["max_px"]),
        "rough_best_rmse": fmt_px(best["rmse_px"]),
        "rough_keyframes": str(final["keyframes"]),
        "rough_loops": str(final["loop_closures"]),
        "calm_rmse": fmt_px(calm_final["rmse_px"]),
        "calm_seconds": fmt_seconds(calm_final["seconds"]),
    }


def stitcher_copy(rows: list[dict], images: dict[str, dict]) -> tuple[str, str]:
    row = stitcher_row(rows)
    assert row is not None
    ours = final_row(rows, "calm")
    frames, seconds = stitcher_frames(row), fmt_seconds(row["seconds"])
    finished = row.get("status") == "finished"
    ours_text = f"{fmt_seconds(ours['seconds'])} s, {fmt_px(ours['rmse_px'])} px RMSE"
    if finished:
        sentence = (
            f"OpenCV's built in stitcher in SCANS mode was handed {frames} frames of the calm flight and took {seconds} s "
            f"to return the mosaic labeled cv2.Stitcher. This pipeline stitched the same flight in {fmt_seconds(ours['seconds'])} s "
            f"with {fmt_px(ours['rmse_px'])} px pose RMSE and a ZNCC of {fmt_zncc(ours.get('zncc'))} against the true world."
        )
    else:
        sentence = (
            f"OpenCV's built in stitcher in SCANS mode was handed {frames} frames of the calm flight and gave up after "
            f"{seconds} s ({html.escape(str(row.get('status')))}). This pipeline stitched the same flight in "
            f"{fmt_seconds(ours['seconds'])} s with {fmt_px(ours['rmse_px'])} px pose RMSE."
        )
    figures = []
    if finished:
        figures.append(
            '<figure>\n              <p class="label"><b>cv2.Stitcher</b>'
            f'<span class="bad">{frames} frames, {seconds} s</span></p>\n'
            f'              <img {img_attrs(images["calm_cv2_stitcher"])} sizes="(min-width: 900px) 33vw, 100vw" '
            'loading="lazy" decoding="async" alt="cv2.Stitcher output for the calm flight, built from the sampled frames">'
            "\n            </figure>"
        )
    figures.append(
        '<figure>\n              <p class="label"><b>This pipeline</b>'
        f"<span>{ours_text}</span></p>\n"
        f'              <img {img_attrs(images["calm_panorama"])} sizes="(min-width: 900px) 62vw, 100vw" '
        'loading="lazy" decoding="async" alt="Full pipeline mosaic of the calm flight: a clean rectangular map with '
        'lakes, forest and snow">\n            </figure>'
    )
    return sentence, "\n            ".join(figures)


# ----------------------------------------------------------------- reports ----

LAYER_LINE = re.compile(r"^(?P<name>[a-z-]+)\s+(?P<cov>[\d.]+)% of pixels(?:, overall center \(x=(?P<x>[\d.]+), y=(?P<y>[\d.]+)\))?")
OBJECT_LINE = re.compile(
    r"object (?P<i>\d+): center \(x=(?P<x>[\d.]+), y=(?P<y>[\d.]+)\), area (?P<area>\d+) px(?:, (?P<shape>[a-z ]+) \((?P<iou>[\d.]+) IoU\))?"
)


def parse_report(text: str) -> dict:
    """Parse ``colors.format_report`` output back into data."""
    layers: list[dict] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if not raw.startswith(" "):
            match = LAYER_LINE.match(line)
            if not match:
                raise BuildError(f"unrecognized report line: {raw!r}")
            center = [float(match["x"]), float(match["y"])] if match["x"] else None
            layers.append({"name": match["name"], "coverage": float(match["cov"]) / 100, "center": center, "objects": []})
            continue
        match = OBJECT_LINE.match(line)
        if match and layers:
            layers[-1]["objects"].append(
                {
                    "center": [float(match["x"]), float(match["y"])],
                    "area": int(match["area"]),
                    "shape": match["shape"],
                    "iou": float(match["iou"]) if match["iou"] else None,
                }
            )
    if not layers:
        raise BuildError("empty color report")
    return {"layers": layers}


# ------------------------------------------------------------------ images ----


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise BuildError(f"OpenCV could not decode {path}")
    return image


def write_webp(path: Path, image: np.ndarray, quality: int = WEBP_QUALITY) -> None:
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_WEBP_QUALITY, quality]):
        raise BuildError(f"could not write {path} (is WebP support missing from OpenCV?)")


def export_image(source: Path, out_dir: Path) -> dict:
    image = read_image(source)
    h, w = image.shape[:2]
    variants = []
    for target in sorted({min(width, w) for width in IMAGE_WIDTHS}):
        scaled = image if target == w else cv2.resize(image, (target, round(h * target / w)), interpolation=cv2.INTER_AREA)
        name = f"{source.stem}-{target}.webp"
        write_webp(out_dir / name, scaled)
        variants.append({"src": f"assets/img/{name}", "width": target, "height": scaled.shape[0]})
    return {"variants": variants, "ratio": f"{w} / {h}"}


def img_attrs(info: dict) -> str:
    largest = info["variants"][-1]
    srcset = ", ".join(f"{v['src']} {v['width']}w" for v in info["variants"])
    return f'src="{largest["src"]}" srcset="{srcset}" width="{largest["width"]}" height="{largest["height"]}"'


def preload(info: dict) -> str:
    largest = info["variants"][-1]
    srcset = ", ".join(f"{v['src']} {v['width']}w" for v in info["variants"])
    return f'<link rel="preload" as="image" href="{largest["src"]}" imagesrcset="{srcset}" imagesizes="100vw">'


def export_demo(docs: Path, out_dir: Path) -> None:
    """Originals are copied byte for byte so browser centers match the Python report."""
    for name in DEMO_IMAGES:
        source = docs / "colors" / f"{name}.jpg"
        shutil.copyfile(source, out_dir / f"{name}.jpg")
        image = read_image(source)
        h, w = image.shape[:2]
        side = min(h, w)
        crop = image[(h - side) // 2 : (h - side) // 2 + side, (w - side) // 2 : (w - side) // 2 + side]
        write_webp(out_dir / f"{name}-thumb.webp", cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA), 80)


def cover(image: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max(width / w, height / h)
    resized = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    y0 = (resized.shape[0] - height) // 2
    x0 = (resized.shape[1] - width) // 2
    return resized[y0 : y0 + height, x0 : x0 + width]


def make_og_image(panorama: np.ndarray, path: Path, rmse: str) -> None:
    """1200x630 social card: the mosaic under a left to right night gradient."""
    width, height = 1200, 630
    base = cover(panorama, width, height).astype(np.float32)
    night = np.array([30, 25, 21], np.float32)  # BGR of the site's night color
    ramp = np.clip(1.05 - np.linspace(0, 1, width) * 0.95, 0.28, 0.93)[None, :, None]
    card = base * (1 - ramp) + night * ramp
    card = np.clip(card, 0, 255).astype(np.uint8)

    orange = (60, 138, 240)
    white = (242, 240, 236)
    muted = (200, 196, 190)
    aa = cv2.LINE_AA
    cv2.rectangle(card, (72, 150), (152, 156), orange, -1)
    cv2.putText(card, "UAV Ground School", (66, 250), cv2.FONT_HERSHEY_TRIPLEX, 2.35, white, 3, aa)
    lines = ("Color segmentation, SUAS shapes and", "drone video mosaicking, in Python and C++")
    for i, line in enumerate(lines):
        cv2.putText(card, line, (72, 322 + i * 50), cv2.FONT_HERSHEY_DUPLEX, 1.1, muted, 2, aa)
    cv2.putText(card, f"{rmse} px pose RMSE on a rough simulated survey", (72, 468), cv2.FONT_HERSHEY_DUPLEX, 0.85, orange, 2, aa)
    cv2.putText(card, "UAVs@Berkeley Software Ground School 2026, Week 2", (72, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.72, muted, 1, aa)
    if not cv2.imwrite(str(path), card, [cv2.IMWRITE_JPEG_QUALITY, 88]):
        raise BuildError(f"could not write {path}")


# ---------------------------------------------------------------- template ----


@dataclass
class Context:
    tokens: dict[str, str]
    stats: dict[str, str]
    images: dict[str, dict]


def render(text: str, ctx: Context, source: Path) -> str:
    def replace(match: re.Match) -> str:
        key, arg = match.group(1), match.group(2)
        if key == "STAT":
            if arg not in ctx.stats:
                raise BuildError(f"{source.name}: unknown stat {arg!r}")
            return ctx.stats[arg]
        if key in ("IMG", "RATIO", "PRELOAD"):
            if arg not in ctx.images:
                raise BuildError(f"{source.name}: image {arg!r} is not exported")
            info = ctx.images[arg]
            return {"IMG": img_attrs, "RATIO": lambda i: i["ratio"], "PRELOAD": preload}[key](info)
        if arg is None and key in ctx.tokens:
            return ctx.tokens[key]
        raise BuildError(f"{source.name}: unknown token {match.group(0)}")

    return TOKEN.sub(replace, text)


def json_ld() -> str:
    data = {
        "@context": "https://schema.org",
        "@type": "SoftwareSourceCode",
        "name": "UAV Ground School",
        "description": DESCRIPTION,
        "url": CANONICAL,
        "image": CANONICAL + "og.jpg",
        "codeRepository": REPO,
        "programmingLanguage": ["Python", "C++"],
        "runtimePlatform": "OpenCV",
        "license": "https://opensource.org/licenses/MIT",
        "keywords": "computer vision, OpenCV, HSV segmentation, drone mapping, image stitching, orthomosaic, bundle adjustment, SUAS",
        "author": {"@type": "Person", "name": "Karthik Subramanian"},
    }
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


# ------------------------------------------------------------------ verify ----

REF_ATTR = re.compile(r"""\b(?:src|href)=["']([^"']+)["']""")
SRCSET_ATTR = re.compile(r"""\b(?:srcset|imagesrcset)=["']([^"']+)["']""")
CSS_URL = re.compile(r"""url\(\s*["']?([^"')]+)["']?\s*\)""")
JS_ASSET = re.compile(r"""["'`]((?:assets|css|js)/[^"'`$]+)["'`]""")


def local_references(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    refs: list[str] = []
    if path.suffix == ".html":
        refs += REF_ATTR.findall(text)
        for srcset in SRCSET_ATTR.findall(text):
            refs += [part.strip().split()[0] for part in srcset.split(",") if part.strip()]
    elif path.suffix == ".css":
        refs += CSS_URL.findall(text)
    elif path.suffix == ".js":
        refs += JS_ASSET.findall(text)
    local = []
    for ref in refs:
        ref = html.unescape(ref)
        if re.match(r"^[a-z][a-z0-9+.-]*:", ref, re.I) or ref.startswith(("//", "#", "data:")):
            continue
        local.append(ref)
    return local


def verify(out: Path) -> None:
    problems = []
    for path in sorted(out.rglob("*")):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(out)
        if DASHES.search(text):
            problems.append(f"{rel}: contains an em or en dash")
        if TOKEN.search(text):
            problems.append(f"{rel}: unreplaced token {TOKEN.search(text).group(0)}")
        for ref in local_references(path):
            if ref.startswith("/"):
                problems.append(f"{rel}: root absolute reference {ref!r} breaks on a project site")
                continue
            # Scripts fetch relative to the page (site root); HTML and CSS relative to themselves.
            base = out if path.suffix == ".js" else path.parent
            target = (base / ref.split("#")[0].split("?")[0]).resolve()
            if ref.split("#")[0] in ("", "./"):
                target = out / "index.html"
            if not target.exists():
                problems.append(f"{rel}: broken reference {ref!r}")
    if problems:
        raise BuildError("site verification failed:\n  " + "\n  ".join(problems))


# ------------------------------------------------------------------- build ----


def build(out: Path, docs: Path = DOCS, today: dt.date | None = None) -> Path:
    out = out.resolve()
    for protected in (ROOT, SITE, docs.resolve(), Path.home()):
        if out == protected or out in protected.parents:
            raise BuildError(f"refusing to overwrite {out}")
    inputs = check_inputs(docs)
    rows = inputs["rows"]
    today = today or dt.date.today()

    for source in SITE.rglob("*"):
        if source.is_file() and source.suffix in TEXT_SUFFIXES and DASHES.search(source.read_text(encoding="utf-8")):
            raise BuildError(f"{source} contains an em or en dash")

    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(SITE, out, ignore=shutil.ignore_patterns(".DS_Store"))
    (out / "assets" / "img").mkdir(parents=True, exist_ok=True)
    (out / "assets" / "demo").mkdir(parents=True, exist_ok=True)

    images = {name: export_image(docs / f"{name}.jpg", out / "assets" / "img") for name in STITCH_IMAGES}
    export_demo(docs, out / "assets" / "demo")
    reports = {name: parse_report((docs / "colors" / f"{name}_report.txt").read_text()) for name in DEMO_IMAGES}

    numbers = stats(rows)
    sentence, figures = stitcher_copy(rows, images)
    ctx = Context(
        tokens={
            "CANONICAL": CANONICAL,
            "DESCRIPTION": html.escape(DESCRIPTION),
            "JSON_LD": json_ld(),
            "BENCHMARK_ROWS": benchmark_rows(rows),
            "STITCHER_SENTENCE": sentence,
            "STITCHER_FIGURES": figures,
            "REFERENCE_REPORTS": json.dumps(reports, separators=(",", ":")).replace("</", "<\\/"),
            "BUILD_DATE": today.isoformat(),
        },
        stats=numbers,
        images=images,
    )
    for page in out.glob("*.html"):
        page.write_text(render(page.read_text(encoding="utf-8"), ctx, page), encoding="utf-8")

    make_og_image(read_image(docs / "rough_panorama.jpg"), out / "og.jpg", numbers["rough_best_rmse"])
    (out / "robots.txt").write_text(f"User-agent: *\nAllow: /\n\nSitemap: {CANONICAL}sitemap.xml\n")
    (out / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"  <url>\n    <loc>{CANONICAL}</loc>\n    <lastmod>{today.isoformat()}</lastmod>\n  </url>\n"
        "</urlset>\n"
    )
    (out / "_headers").write_text(HEADERS)
    verify(out)
    return out


# Cloudflare Pages response headers. Images keep their names across builds but
# change rarely, so they get a day of caching; HTML, CSS and JS revalidate.
HEADERS = """/*
  X-Content-Type-Options: nosniff
  Referrer-Policy: strict-origin-when-cross-origin
  X-Frame-Options: DENY
  Permissions-Policy: camera=(), microphone=(), geolocation=()

/assets/*
  Cache-Control: public, max-age=86400, stale-while-revalidate=604800

/og.jpg
  Cache-Control: public, max-age=86400

/css/*
  Cache-Control: public, max-age=0, must-revalidate

/js/*
  Cache-Control: public, max-age=0, must-revalidate
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(ROOT / "build" / "site"), help="output directory (replaced)")
    parser.add_argument("--docs", default=str(DOCS), help="directory with benchmark.json and preview images")
    args = parser.parse_args(argv)
    try:
        out = build(Path(args.out), Path(args.docs))
    except BuildError as error:
        print(f"build_site: error: {error}", file=sys.stderr)
        return 1
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"built {out} ({size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
