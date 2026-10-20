"""Build the static showcase site into a directory ready for Cloudflare Pages.

    PYTHONPATH=. python scripts/build_site.py --out build/site

Steps: copy ``site/``, re-encode the committed images from ``docs/week02`` and
``docs/week03`` as WebP (800 and 1600 px wide), copy the Color Me Impressed test
images for the in-browser demo and the week 3 detections for the explorer,
render the benchmark tables from their JSON, write ``og.jpg`` and one card per
week page, ``robots.txt``, ``sitemap.xml`` and the Cloudflare ``_headers`` file, then verify
the result (no unreplaced tokens, no em or en dashes, every local reference
resolves).

Every input is checked up front and the build fails with a list of whatever is
missing, so a stale or partial ``docs/`` never ships silently.
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
DOCS3 = ROOT / "docs" / "week03"
DOCS4 = ROOT / "docs" / "week04"
DOCS5 = ROOT / "docs" / "week05"
DOCS6 = ROOT / "docs" / "week06"
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
TOKEN = re.compile(r"\{\{([A-Z0-9_]+)(?::([^}]*))?\}\}")
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


W3_DESCRIPTION = (
    "UAVs@Berkeley Ground School Week 3, solved: SimpleBlobDetector, LoG, DoG and DoH blob detection in CIELAB scored "
    "against hand checked ground truth, a classical cone, cube and ring detector checked against SAM masks, and a "
    "YOLOv8 study with sliced inference that finds the dataset's split leaks."
)
W3_PHOTOS = ("polka_dots_1", "polka_dots_2", "polka_dots_3")
W3_IMAGES = {
    "w3_hero": "hero.jpg",
    "w3_scale_space": "scale_space.jpg",
    "w3_synthetic": "synthetic.jpg",
    "w3_objects": "objects.jpg",
    "w3_objects_truth": "objects_truth.jpg",
    "w3_robustness": "objects_robustness.jpg",
    "w3_targets": "targets.jpg",
    "w3_beyond": "beyond.jpg",
    "w3_yolo_whole": "yolo/whole.jpg",
    "w3_yolo_sliced": "yolo/sliced.jpg",
}
W3_DATA = ("dots_benchmark.json", "objects.json", "beyond.json", "yolo/results.json", "yolo/dataset.json")
METHOD_LABELS = {
    "gray": "SimpleBlobDetector, grayscale",
    "simple": "SimpleBlobDetector, per palette color",
    "contrast": "SimpleBlobDetector, background contrast",
    "contour": "Contours on color edges",
    "log": "Laplacian of Gaussian",
    "dog": "Difference of Gaussians",
    "doh": "Determinant of Hessian",
}
PHOTO_TITLES = {"polka_dots_1.png": "Flat print", "polka_dots_2.jpg": "Fabric", "polka_dots_3.jpg": "Cards"}
YOLO_CLASSES = ("ball", "goalkeeper", "player", "referee")


def required_week3(docs3: Path) -> list[Path]:
    paths = [docs3 / name for name in W3_IMAGES.values()]
    paths += [docs3 / name for name in W3_DATA]
    for photo in W3_PHOTOS:
        paths.append(docs3 / "dots" / f"{photo}.json")
    paths += [docs3 / "photos" / name for name in ("polka_dots_1.png", "polka_dots_2.jpg", "polka_dots_3.jpg")]
    return paths


def load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise BuildError(f"{path} is not valid JSON: {error}") from error


def fmt3(value: float) -> str:
    return f"{value:.3f}"


def week3_bench(report: dict) -> tuple[str, str, dict[str, str]]:
    """Header cells, body rows and headline numbers for the dot benchmark table."""
    methods = report["methods"]
    real = report.get("real") or {}
    if not real:
        raise BuildError("dots_benchmark.json has no real photo results (run the benchmark with the photos)")
    names = list(real)
    head = "\n                ".join(
        f'<th scope="col" class="num">{html.escape(PHOTO_TITLES.get(n, n))}<small>{real[n]["required_dots"]} dots</small></th>'
        for n in names
    )
    best_real = {n: max(real[n][m]["f1"] for m in methods) for n in names}
    syn = report["synthetic"]
    overall = report["synthetic_overall"]
    best_syn = max(overall[m]["f1"] for m in methods)
    rows = []
    for m in methods:
        cells = [f'<td class="variant">{html.escape(METHOD_LABELS.get(m, m))}</td>']
        for n in names:
            f1 = real[n][m]["f1"]
            cls = "num top" if f1 == best_real[n] else "num"
            cells.append(f'<td class="{cls}">{fmt3(f1)}</td>')
        centers = [syn[p][m]["center_error_px"] for p in syn if syn[p][m].get("center_error_px") is not None]
        radii = [syn[p][m]["radius_error"] for p in syn if syn[p][m].get("radius_error") is not None]
        cls = "num top" if overall[m]["f1"] == best_syn else "num"
        cells.append(f'<td class="{cls}">{fmt3(overall[m]["f1"])}</td>')
        cells.append(f'<td class="num">{np.median(centers):.2f} px</td>' if centers else '<td class="num"></td>')
        cells.append(f'<td class="num">{100 * np.median(radii):.1f}%</td>' if radii else '<td class="num"></td>')
        rows.append("              <tr>" + "".join(cells) + "</tr>")
    real_overall = report["real_overall"]
    best_method = max(methods, key=lambda m: (real_overall[m]["f1"], overall[m]["f1"]))
    center_best = [syn[p][best_method]["center_error_px"] for p in syn if syn[p][best_method].get("center_error_px") is not None]
    numbers = {
        "w3_best_f1": fmt3(real_overall[best_method]["f1"]),
        "w3_best_method": METHOD_LABELS[best_method],
        "w3_required": str(sum(real[n]["required_dots"] for n in names)),
        "w3_center_err": f"{np.median(center_best):.2f}",
        "w3_contrast_f1": fmt3(real_overall["contrast"]["f1"]) if "contrast" in real_overall else "",
        "w3_scenes": str(len(syn) * report["seeds"]),
        "w3_log_found": str(real_overall["log"]["tp"]),
        "w3_log_fp": str(real_overall["log"]["fp"]),
    }
    return head, "\n".join(rows), numbers


def week3_pieces(result: dict) -> tuple[str, dict[str, str]]:
    colors = {"cone": "#f5c814", "cube": "#7a3cbe", "ring": "#eb3c3c"}
    ious = {}
    for m in result["score"]["matches"]:
        ious.setdefault(m["label"], []).append(m["mask_iou"])
    rows = []
    for piece in result["pieces"]:
        f = piece["features"]
        label = piece["label"]
        iou = ious.get(label, [None]).pop(0) if ious.get(label) else None
        rows.append(
            "              <tr>"
            f'<td><span class="dot" style="background: {colors.get(label, "#999")}"></span>{html.escape(label)}</td>'
            f'<td class="num">{f["solidity"]:.2f}</td><td class="num">{f["hole_ratio"]:.2f}</td><td class="num">{f["triangularity"]:.2f}</td>'
            f'<td class="num">{"" if iou is None else f"{iou:.3f}"}</td></tr>'
        )
    score = result["score"]
    counts = result["counts"]
    summary = ", ".join(f"{counts.get(c, 0)} {c}s" for c in ("cone", "cube", "ring"))
    robust = result.get("robustness", {})
    all_same = all(v == counts for v in robust.values())
    numbers = {
        "w3_pieces": f"{score['correct_class']} of {score['truth_objects']} pieces found and classified",
        "w3_mask_iou": f"{score['mean_mask_iou']:.3f}",
        "w3_pieces_sentence": (
            f"All {score['truth_objects']} pieces found and labeled with no false positives: mean mask IoU {score['mean_mask_iou']:.3f} "
            f"and box IoU {score['mean_box_iou']:.3f} against masks from SAM 2, prompted with hand drawn boxes and checked by eye. "
            f"Refining the outlines with GrabCut was tried and scored {result['grabcut_score']['mean_mask_iou']:.3f}, so it stays off."
        ),
        "w3_robust": (summary if all_same else "mostly the same") + (f" in all {len(robust)} versions" if all_same else ""),
    }
    return "\n".join(rows), numbers


def week3_beyond(result: dict) -> dict[str, str]:
    """Challenge 1 on the objects and shapes photos, as sentences for the page."""
    obj, shp = result["objects"], result["shapes"]
    log, contours = obj["log"], obj["contours"]
    raw_best = max(shp[m]["tp"] for m in METHOD_LABELS)
    return {
        "w3_beyond_objects": (
            f"LoG's {log['strongest_matched']} strongest peaks are the {log['strongest_matched']} solid pieces, and it misses both rings: "
            f"a ring is not a blob. Contours of saturation find {contours['matched']} of 6 and merge the touching cubes."
        ),
        "w3_beyond_shapes": (
            f"On the noisy shapes file every dot detector finds {raw_best} of 10 dots. After non-local means denoising LoG finds "
            f"{shp['log+nlm']['tp']}, and contours of the Delta E segmentation find all {shp['contours']['tp']} with no false positives."
        ),
    }


def week3_yolo(results: dict, dataset: dict) -> tuple[str, str, dict[str, str]]:
    rows = []
    for name, r in results.items():
        exp = r["experiment"]
        for mode in ("full_frame", "sliced"):
            if mode not in r:
                continue
            m = r[mode]
            c = m["classes"]
            split = "Roboflow (leaky)" if exp["split"] == "original" else "held out clips"
            inference = f"sliced, {exp['imgsz']} px" if mode == "sliced" else f"whole frame, {exp['imgsz']} px"
            cells = [
                f'<td class="variant"><code>{html.escape(name)}</code></td>',
                f"<td>{split}</td>",
                f"<td>{inference}</td>",
                f'<td class="num">{fmt3(m["mAP50"])}</td>',
                f'<td class="num">{fmt3(m["mAP50-95"])}</td>',
            ]
            cells += [f'<td class="num">{fmt3(c[k]["AP50"])}</td>' for k in YOLO_CLASSES]
            rows.append("              <tr>" + "".join(cells) + "</tr>")
    notes = [f"<p>{html.escape(line)}</p>" for line in dataset.get("notes", [])]
    boxes = dataset["boxes"]
    total = sum(sum(v.values()) for v in boxes.values())
    balls = sum(v.get("ball", 0) for v in boxes.values())
    frames = sum(dataset["frames"].values())
    numbers = {
        "w3_frames": str(frames),
        "w3_leak": f"{len(dataset['test_clips_seen_in_train'])} of {len(dataset['clips']['test'])}",
        "w3_ball_px": f"{dataset['median_box_size_px']['ball']:.0f}",
        "w3_ball_share": f"{100 * balls / total:.0f}%",
        "w3_yolo_headline": dataset.get("headline", ""),
    }
    return "\n".join(rows), "\n".join(notes), numbers


def export_explorer(docs3: Path, out_dir: Path) -> None:
    """Photos (WebP at full size, byte faithful geometry) and detections for the explorer."""
    for photo in W3_PHOTOS:
        source = next((docs3 / "photos").glob(f"{photo}.*"))
        image = read_image(source)
        write_webp(out_dir / f"{photo}.webp", image, 86)
        h, w = image.shape[:2]
        side = min(h, w)
        crop = image[(h - side) // 2 : (h - side) // 2 + side, (w - side) // 2 : (w - side) // 2 + side]
        write_webp(out_dir / f"{photo}-thumb.webp", cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA), 80)
        shutil.copyfile(docs3 / "dots" / f"{photo}.json", out_dir / f"{photo}.json")


def make_og_week3(hero: np.ndarray, path: Path, f1: str) -> None:
    width, height = 1200, 630
    base = cover(hero, width, height).astype(np.float32)
    night = np.array([30, 25, 21], np.float32)
    ramp = np.clip(1.08 - np.linspace(0, 1, width) * 0.7, 0.5, 0.94)[None, :, None]
    card = np.clip(base * (1 - ramp) + night * ramp, 0, 255).astype(np.uint8)
    orange, white, muted = (60, 138, 240), (242, 240, 236), (200, 196, 190)
    aa = cv2.LINE_AA
    cv2.rectangle(card, (72, 150), (152, 156), orange, -1)
    cv2.putText(card, "Finding what", (66, 238), cv2.FONT_HERSHEY_TRIPLEX, 2.2, white, 3, aa)
    cv2.putText(card, "stands out.", (66, 318), cv2.FONT_HERSHEY_TRIPLEX, 2.2, orange, 3, aa)
    for i, line in enumerate(("Blob detection, cones, cubes and rings,", "and YOLOv8 with sliced inference")):
        cv2.putText(card, line, (72, 392 + i * 46), cv2.FONT_HERSHEY_DUPLEX, 1.0, muted, 2, aa)
    cv2.putText(card, f"{f1} F1 on hand checked polka dots", (72, 512), cv2.FONT_HERSHEY_DUPLEX, 0.85, orange, 2, aa)
    cv2.putText(card, "UAVs@Berkeley Software Ground School 2026, Week 3", (72, 575), cv2.FONT_HERSHEY_SIMPLEX, 0.72, muted, 1, aa)
    if not cv2.imwrite(str(path), card, [cv2.IMWRITE_JPEG_QUALITY, 88]):
        raise BuildError(f"could not write {path}")


W4_DESCRIPTION = (
    "UAVs@Berkeley Ground School Week 4, solved: a drone designed from real parts around the Cube Orange+, wired pin by pin "
    "and checked by a design rule checker, with MAVLink, DroneCAN, CAN, RTCM, CRSF, SBUS and DShot implemented from their "
    "specifications and the generated parameters verified on ArduCopter 4.7.1 in simulation."
)
W4_DATA = (
    "summary.json",
    "site.json",
    "check.json",
    "scope.json",
    "journey.json",
    "pid.json",
    "bom.json",
    "sitl.json",
    "performance.json",
    "wiring.svg",
)


def required_week4(docs4: Path) -> list[Path]:
    return [docs4 / name for name in W4_DATA]


def cpp_case_count() -> str:
    readme = (ROOT / "week04" / "cpp" / "README.md").read_text()
    match = re.search(r"All (\d+) cases pass", readme)
    if not match:
        raise BuildError("week04/cpp/README.md has no total parity case count")
    return f"{int(match.group(1)):,}"


def week4(docs4: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Stats and HTML fragments for week4.html."""
    summary = load_json(docs4 / "summary.json")
    bom = load_json(docs4 / "bom.json")
    sitl = load_json(docs4 / "sitl.json")
    perf = load_json(docs4 / "performance.json")
    pid = load_json(docs4 / "pid.json")
    report = load_json(docs4 / "check.json")
    rows = []
    for r in bom["rows"]:
        if r.get("included_in"):
            price = f"in the {r['included_in']} kit"
        elif r["unit_usd"] is None:
            price = "not found"
        else:
            price = f"${r['unit_usd']:,.2f}"
        mass = "" if r["mass_g"] is None else f"{r['mass_g']:g} g" + (" est." if "mass_g" in r["estimated"] else "")
        name = html.escape(r["name"])
        link = f'<a href="{html.escape(r["url"])}" rel="noopener">{name}</a>' if r.get("url") else name
        rows.append(
            f'<tr><td>{link}</td><td>{html.escape(r["role"].replace("_", " "))}</td><td class="num">{r["count"]}</td><td class="num">{price}</td><td class="num">{mass}</td></tr>'
        )
    rows.append(
        f'<tr class="total"><td><strong>Total</strong></td><td></td><td></td><td class="num"><strong>${bom["total_usd"]:,.2f}</strong></td><td class="num"><strong>{summary["mass_kg"]} kg</strong></td></tr>'
    )
    budget_rows = []
    for r in sitl["link_budget"]["rows"]:
        budget_rows.append(
            f'<tr><td>{html.escape(r["stream"])}</td><td class="mono">{html.escape(r["message"])}</td><td class="num">{r["predicted_hz"]:g}</td>'
            f'<td class="num">{r["measured_hz"]:.2f}</td><td class="num">{r["measured_bytes_per_s"]:.0f}</td></tr>'
        )
    air = pid["airframe"]
    rules = len({f["rule"] for f in report})
    stats = {
        "w4_errors": str(summary["errors"]),
        "w4_warnings": str(summary["warnings"]),
        "w4_passed": str(summary["passed"]),
        "w4_links": str(summary["links"]),
        "w4_protocols": str(len(summary["protocols"])),
        "w4_rules": str(rules),
        "w4_mavlink": str(summary["mavlink_messages"]),
        "w4_mass": f"{summary['mass_kg']:.2f}",
        "w4_tw": f"{summary['thrust_to_weight']:.2f}",
        "w4_minutes": f"{summary['hover_minutes']:.1f}",
        "w4_hover": f"{100 * perf['hover_throttle']:.0f}",
        "w4_hover_a": f"{perf['hover_current_a'] + perf['avionics_w'] / 14.8:.1f}",
        "w4_ref_minutes": f"{perf['calibration']['reference_minutes']:.0f}",
        "w4_sitl_params": f"{sitl['params']['accepted']} of {sitl['params']['sent']}",
        "w4_link_pred": f"{sitl['link_budget']['predicted_bytes_per_s']:,.0f}",
        "w4_link_meas": f"{sitl['link_budget']['measured_bytes_per_s']:,.0f}",
        "w4_window": f"{sitl['link_budget']['window_s']:.0f}",
        "w4_sigs": str(sitl["signing"]["usb_signatures_verified"]),
        "w4_mav2_hz": f"{sitl['channel_mapping']['serial1_attitude_hz']:.1f}",
        "w4_sim_rise": f"{air['sim_metrics']['rise_time']:.2f}",
        "w4_sitl_rise": f"{air['sitl_metrics']['rise_time']:.2f}",
        "w4_cpp_cases": cpp_case_count(),
    }
    svg = (docs4 / "wiring.svg").read_text()
    tokens = {
        "W4_DESCRIPTION": html.escape(W4_DESCRIPTION),
        "W4_JSON_LD": week4_json_ld(),
        "W4_BOM_ROWS": "".join(rows),
        "W4_BUDGET_ROWS": "".join(budget_rows),
        "W4_WIRING": svg,
    }
    tokens["W4_FIRMWARE_ROWS"], fw_stats = week4_firmware(docs4)
    stats.update(fw_stats)
    return stats, tokens


def week4_firmware(docs4: Path) -> tuple[str, dict[str, str]]:
    """The firmware matrix rows (from week04/data/firmware.json) and one verdict per firmware."""
    from week04 import firmware

    report = load_json(docs4 / "firmware.json")
    data = firmware.load()
    labels = {"yes": "yes", "partial": "partly", "no": "no", "unconfirmed": "unconfirmed"}
    rows = []
    for f in data["features"]:
        if f["id"] not in report["needed"]:
            continue
        cells = []
        for fw in firmware.FIRMWARES:
            c = f[fw]
            cells.append(
                f'<td><span class="fw fw-{c["support"]}">{labels[c["support"]]}</span> {html.escape(c["how"])} '
                f'<a class="src" href="{html.escape(c["source"])}" rel="noopener">source</a></td>'
            )
        rows.append(f'<tr><th scope="row">{html.escape(f["feature"])}</th>{"".join(cells)}</tr>')
    total = len(report["needed"])
    stats = {}
    for fw, v in report["verdicts"].items():
        stats[f"w4_fw_{fw}"] = f"{len(v['supported'])} of {total}"
        stats[f"w4_fw_{fw}_changes"] = str(len(v["partial"]))
        stats[f"w4_fw_{fw}_lost"] = str(len(v["lost"]))
    return "".join(rows), stats


def export_week4(docs4: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("site.json", "check.json", "scope.json", "journey.json", "pid.json"):
        shutil.copyfile(docs4 / name, out_dir / name)


def make_og_week4(svg_png: np.ndarray | None, path: Path, stats: dict[str, str]) -> None:
    width, height = 1200, 630
    card = np.zeros((height, width, 3), np.uint8)
    card[:] = (29, 25, 18)
    for x in range(0, width, 40):
        cv2.line(card, (x, 0), (x, height), (42, 37, 27), 1)
    for y in range(0, height, 40):
        cv2.line(card, (0, y), (width, y), (42, 37, 27), 1)
    orange, white, muted = (60, 138, 240), (242, 240, 236), (200, 196, 190)
    colors = [(60, 138, 240), (255, 140, 181), (138, 211, 106), (216, 201, 76), (76, 201, 242), (255, 140, 181)]
    aa = cv2.LINE_AA
    # a stylised flight controller with its wires, echoing the diagram
    cx, cy = 930, 315
    for i, color in enumerate(colors):
        y = 150 + i * 66
        cv2.line(card, (cx - 110, cy - 90 + i * 36), (cx - 170, cy - 90 + i * 36), color, 4, aa)
        cv2.line(card, (cx - 170, cy - 90 + i * 36), (cx - 170, y), color, 4, aa)
        cv2.line(card, (cx - 170, y), (cx - 250, y), color, 4, aa)
        cv2.circle(card, (cx - 250, y), 7, color, -1, aa)
    cv2.rectangle(card, (cx - 110, cy - 115), (cx + 110, cy + 115), (42, 111, 232), -1)
    cv2.putText(card, "Cube", (cx - 58, cy + 12), cv2.FONT_HERSHEY_DUPLEX, 1.5, (9, 17, 27), 2, aa)
    cv2.rectangle(card, (72, 150), (152, 156), orange, -1)
    cv2.putText(card, "Protocol", (66, 238), cv2.FONT_HERSHEY_TRIPLEX, 2.2, white, 3, aa)
    cv2.putText(card, "Pro.", (66, 318), cv2.FONT_HERSHEY_TRIPLEX, 2.2, orange, 3, aa)
    for i, line in enumerate(("A drone wired pin by pin, every", "protocol implemented and checked")):
        cv2.putText(card, line, (72, 392 + i * 46), cv2.FONT_HERSHEY_DUPLEX, 1.0, muted, 2, aa)
    cv2.putText(
        card,
        f"{stats['w4_errors']} errors, {stats['w4_passed']} checks passed, ArduCopter 4.7.1 verified",
        (72, 512),
        cv2.FONT_HERSHEY_DUPLEX,
        0.72,
        orange,
        2,
        aa,
    )
    cv2.putText(card, "UAVs@Berkeley Software Ground School 2026, Week 4", (72, 575), cv2.FONT_HERSHEY_SIMPLEX, 0.72, muted, 1, aa)
    if not cv2.imwrite(str(path), card, [cv2.IMWRITE_JPEG_QUALITY, 88]):
        raise BuildError(f"could not write {path}")


def week4_json_ld() -> str:
    data = {
        "@context": "https://schema.org",
        "@type": "SoftwareSourceCode",
        "name": "UAV Ground School Week 4: flight controllers and protocols",
        "description": W4_DESCRIPTION,
        "url": CANONICAL + "week4",
        "image": CANONICAL + "og-week4.jpg",
        "codeRepository": REPO,
        "programmingLanguage": ["Python", "C++", "JavaScript"],
        "runtimePlatform": "ArduPilot",
        "license": "https://opensource.org/licenses/MIT",
        "keywords": "flight controller, Cube Orange, ArduPilot, MAVLink, DroneCAN, CAN bus, RTCM, RTK, CRSF, ExpressLRS, SBUS, DShot, PID, wiring diagram",
        "author": {"@type": "Person", "name": "Karthik Subramanian"},
    }
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


W5_DESCRIPTION = (
    "UAVs@Berkeley Ground School Week 5, solved: the ROS 2 Jazzy tutorials, a simulated drone wired with topics, services "
    "and an action, measured QoS and executor experiments, and a pure Python RTPS and DDS participant with its own CDR codec, "
    "REP 2011 type hashes and QoS rules, checked against real ROS 2 nodes and Cyclone DDS."
)
W5_DATA = ("summary.json", "wire.json", "mission.json", "lab.json", "interop.json", "transcripts.json", "checks.json")
W5_EXPORT = ("mission.json", "wire.json", "lab.json")


def required_week5(docs5: Path) -> list[Path]:
    return [docs5 / name for name in W5_DATA]


def _cell(value: object, cls: str = "") -> str:
    attr = f' class="{cls}"' if cls else ""
    return f"<td{attr}>{html.escape(str(value))}</td>"


def _policy(name: str) -> str:
    return f"<code>{html.escape(name.replace('_', ' '))}</code>"


def week5_tables(lab: dict) -> dict[str, str]:
    """The measured QoS and executor tables, rendered at build time so they work without JavaScript."""
    rel = []
    for r in lab["reliability"]:
        event = ", ".join(sorted(set(r["publisher_event"]) | set(r["subscription_event"]))) or "none"
        verdict = r["rclpy_check"].split(":")[0].lower()
        rel.append(
            f'<tr class="{"bad" if not r["matched"] else ""}"><td>{_policy(r["publisher"])}</td><td>{_policy(r["subscription"])}</td>'
            f'<td>{"yes" if r["matched"] else "no"}</td><td class="num">{r["received"]} of {r["sent"]}</td>'
            f'<td>{html.escape(event)}</td><td><span class="pill {verdict}">{verdict}</span></td></tr>'
        )
    dur = []
    for r in lab["durability"]:
        dur.append(
            f'<tr><td>{_policy(r["publisher"])}</td><td>{_policy(r["late_subscription"])}</td><td class="num">{r["received"]}</td>'
            f'<td class="mono small">{html.escape(r["rclpy_check"])}</td></tr>'
        )
    top = max(r["received"] for r in lab["depth"])
    depth = []
    for r in lab["depth"]:
        share = r["received"] / r["sent"]
        depth.append(
            f"<tr><td><code>{html.escape(r['history'])}</code></td><td>{_policy(r['reliability'])}</td>"
            f'<td class="num">{r["received"]} of {r["sent"]}</td>'
            f'<td class="bar-cell"><span class="bar" style="--w: {100 * r["received"] / top:.1f}%"></span><span class="mono">{100 * share:.0f}%</span></td>'
            f"<td>{'yes' if r['in_order'] else 'no'}</td></tr>"
        )
    block = []
    for r in lab["blocking"]:
        block.append(
            f'<tr><td>{html.escape(r["executor"])}</td><td class="num">{r["ticks"]}</td><td class="num">{r["mean_gap_ms"]:.1f}</td>'
            f'<td class="num">{r["max_gap_ms"]:.1f}</td></tr>'
        )
    dead = []
    for r in lab["deadlock"]:
        ok = r["outcome"] == "returned"
        ms = "" if r["ms"] is None else f"{r['ms']:g}"
        dead.append(
            f'<tr class="{"" if ok else "bad"}"><td>{html.escape(r["setup"])}</td>'
            f'<td><span class="pill {"ok" if ok else "error"}">{html.escape(r["outcome"])}</span></td>'
            f'<td class="num">{ms}</td></tr>'
        )
    return {
        "W5_RELIABILITY_ROWS": "".join(rel),
        "W5_DURABILITY_ROWS": "".join(dur),
        "W5_DEPTH_ROWS": "".join(depth),
        "W5_BLOCKING_ROWS": "".join(block),
        "W5_DEADLOCK_ROWS": "".join(dead),
    }


def week5_transcripts(transcripts: list[dict]) -> str:
    """The tutorial runs as terminal windows, one pair per tutorial."""
    out = []
    for item in transcripts:
        terms = []
        for term in item["terminals"]:
            lines = "\n".join(html.escape(line) for line in term["output"])
            words = term["command"].split()
            package, executable = (words[2], words[3]) if len(words) > 3 else ("ros2", term["command"])
            terms.append(
                '<figure class="terminal">'
                f"<header><span>{html.escape(executable)}</span><span>{html.escape(package)}</span></header>"
                f'<pre><code><span class="p">$ </span>{html.escape(term["command"])}\n{lines}</code></pre></figure>'
            )
        out.append(
            f'<div class="transcript"><h3>{html.escape(item["title"])}</h3><div class="transcript-pair">{"".join(terms)}</div></div>'
        )
    return "".join(out)


def week5_interop(interop: dict) -> str:
    """The five interop runs as a list of what crossed the wire."""
    a = interop["ros_talker_to_python_listener"]
    b = interop["python_talker_to_ros_listener"]
    c = interop["python_client_to_ros_service"]
    d = interop["ros_clients_to_python_service"]
    e = interop["cyclone_talker_to_python_listener"]
    heard = next((line for line in b["listener_output"] if "I heard" in line), b["listener_output"][0])
    heard = heard.split("]: ", 1)[-1]
    clients = d["client_output"]
    rows = [
        ("ROS talker", "Python listener", f"received {', '.join(repr(m) for m in a['received'][:3])}", "Fast DDS"),
        ("Python talker", "ROS listener", heard, "Fast DDS"),
        ("Python client", "ROS service", f"{c['request']} gave {c['reply']} in {c['round_trip_ms']:.2f} ms", "Fast DDS"),
        (
            "ROS clients, Python and C++",
            "Python service",
            "; ".join(f"{k}: {v[0].split(']: ', 1)[-1]}" for k, v in clients.items()),
            "Fast DDS",
        ),
        ("Cyclone DDS talker", "Python listener", f"received {len(e['received'])} messages", ", ".join(e["vendors"])),
    ]
    items = []
    for i, (src, dst, what, vendor) in enumerate(rows, 1):
        items.append(
            f'<li><span class="n">{i}</span><div><p class="dir"><b>{html.escape(src)}</b> to <b>{html.escape(dst)}</b>'
            f'<span class="vendor">{html.escape(vendor)}</span></p><p class="mono small">{html.escape(what)}</p></div></li>'
        )
    return "".join(items)


def week5(docs5: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Stats and HTML fragments for week5.html."""
    summary = load_json(docs5 / "summary.json")
    checks = load_json(docs5 / "checks.json")
    lab = load_json(docs5 / "lab.json")
    mission = load_json(docs5 / "mission.json")
    wire = load_json(docs5 / "wire.json")
    for key in ("type_hashes", "qos_pairs", "cdr", "interop_directions", "mission_outcome", "target_confirmed"):
        if key not in summary:
            raise BuildError(f"{docs5 / 'summary.json'} is missing {key!r}")
    if not isinstance(wire, list) or not all({"hex", "spans", "title"} <= set(p) for p in wire):
        raise BuildError(f"{docs5 / 'wire.json'} must be a list of packets with hex, spans and title")
    for key in ("events", "odometry", "feedback", "status", "detections"):
        if not mission.get(key):
            raise BuildError(f"{docs5 / 'mission.json'} has no {key!r}")
    target = re.search(r"\(([-\d.]+), ([-\d.]+)\)", summary["target_confirmed"])
    landed = re.search(r"\(([-\d.]+), ([-\d.]+), ([-\d.]+)\)", summary["mission_outcome"])
    if not target or not landed:
        raise BuildError("summary.json target_confirmed and mission_outcome must carry coordinates")
    single = next(r for r in lab["blocking"] if r["executor"] == "SingleThreadedExecutor")
    groups = next(r for r in lab["blocking"] if "separate" in r["executor"])
    keep_all = next(r for r in lab["depth"] if r["history"] == "keep_all")
    keep1 = next(r for r in lab["depth"] if r["history"] == "keep_last 1")
    stats = {
        "w5_hashes": f"{summary['type_hashes'][0]:,} of {summary['type_hashes'][1]:,}",
        "w5_qos": f"{summary['qos_pairs'][0]:,}",
        "w5_qos_total": f"{summary['qos_pairs'][1]:,}",
        "w5_cdr": f"{summary['cdr'][0]} of {summary['cdr'][1]}",
        "w5_cdr_reads": f"{summary['cdr_rclpy_reads_ours']} of {summary['cdr'][1]}",
        "w5_cdr_dirty": str(summary["cdr_rclpy_dirty_padding"]),
        "w5_overshoot": f"{summary['cdr_rclpy_overshoot']:.1f}",
        "w5_directions": str(summary["interop_directions"]),
        "w5_cyclone": ", ".join(summary["cyclone_vendors"]),
        "w5_rtt": f"{summary['client_round_trip_ms']:.2f}",
        "w5_mission_s": f"{summary['mission_seconds']:.1f}",
        "w5_target": f"({target.group(1)}, {target.group(2)})",
        "w5_landed": f"({landed.group(1)}, {landed.group(2)})",
        "w5_frames": str(summary["recorded_frames"]),
        "w5_megabytes": f"{summary['recorded_megabytes']:.1f}",
        "w5_topics": str(summary["recorded_topics"]),
        "w5_pubsub_packets": str(summary["packets"]["pubsub"]),
        "w5_srvcli_packets": str(summary["packets"]["srvcli"]),
        "w5_block_single": f"{single['max_gap_ms']:.1f}",
        "w5_block_groups": f"{groups['max_gap_ms']:.1f}",
        "w5_ticks_single": str(single["ticks"]),
        "w5_ticks_groups": str(groups["ticks"]),
        "w5_keep_all": f"{keep_all['received']} of {keep_all['sent']}",
        "w5_keep1": f"{keep1['received']} of {keep1['sent']}",
        "w5_packets_tour": str(len(wire)),
        "w5_checks_hashes": f"{checks['type_hashes_identical']:,}",
    }
    tokens = {
        "W5_DESCRIPTION": html.escape(W5_DESCRIPTION),
        "W5_JSON_LD": week5_json_ld(),
        "W5_TRANSCRIPTS": week5_transcripts(load_json(docs5 / "transcripts.json")),
        "W5_INTEROP": week5_interop(load_json(docs5 / "interop.json")),
        **week5_tables(lab),
    }
    return stats, tokens


def export_week5(docs5: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in W5_EXPORT:
        shutil.copyfile(docs5 / name, out_dir / name)


def make_og_week5(path: Path, stats: dict[str, str]) -> None:
    width, height = 1200, 630
    card = np.zeros((height, width, 3), np.uint8)
    card[:] = (29, 25, 18)
    for x in range(0, width, 40):
        cv2.line(card, (x, 0), (x, height), (42, 37, 27), 1)
    for y in range(0, height, 40):
        cv2.line(card, (0, y), (width, y), (42, 37, 27), 1)
    orange, white, muted = (60, 138, 240), (242, 240, 236), (200, 196, 190)
    steel, green = (216, 201, 76), (138, 211, 106)
    aa = cv2.LINE_AA
    # the four drone nodes and the edges between them, echoing the page's graph
    boxes = {"camera": (770, 150), "px4": (1030, 150), "detector": (770, 440), "planner": (1030, 440)}
    edges = [("px4", "camera", steel), ("camera", "detector", steel), ("detector", "planner", steel), ("px4", "planner", green)]
    for a, b, color in edges:
        cv2.line(card, boxes[a], boxes[b], color, 4, aa)
    cv2.line(card, (boxes["planner"][0] + 40, 440), (boxes["px4"][0] + 40, 150), orange, 4, aa)
    for (x1, y1), (x2, y2) in ((boxes["px4"], boxes["camera"]), (boxes["camera"], boxes["detector"])):
        cv2.circle(card, ((x1 + x2) // 2, (y1 + y2) // 2), 9, orange, -1, aa)
    for name, (x, y) in boxes.items():
        cv2.rectangle(card, (x - 95, y - 34), (x + 95, y + 34), (42, 37, 27), -1)
        cv2.rectangle(card, (x - 95, y - 34), (x + 95, y + 34), (110, 100, 88), 2, aa)
        size = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)[0]
        cv2.putText(card, name, (x - size[0] // 2, y + size[1] // 2), cv2.FONT_HERSHEY_DUPLEX, 0.85, white, 2, aa)
    cv2.rectangle(card, (72, 150), (152, 156), orange, -1)
    cv2.putText(card, "ROS 2, from", (66, 238), cv2.FONT_HERSHEY_TRIPLEX, 2.0, white, 3, aa)
    cv2.putText(card, "the wire up.", (66, 318), cv2.FONT_HERSHEY_TRIPLEX, 2.0, orange, 3, aa)
    for i, line in enumerate(("A drone in four nodes, and DDS", "rebuilt from scratch in Python")):
        cv2.putText(card, line, (72, 392 + i * 46), cv2.FONT_HERSHEY_DUPLEX, 1.0, muted, 2, aa)
    cv2.putText(
        card,
        f"{stats['w5_hashes']} type hashes, {stats['w5_qos']} QoS pairs match ROS 2",
        (72, 512),
        cv2.FONT_HERSHEY_DUPLEX,
        0.72,
        orange,
        2,
        aa,
    )
    cv2.putText(card, "UAVs@Berkeley Software Ground School 2026, Week 5", (72, 575), cv2.FONT_HERSHEY_SIMPLEX, 0.72, muted, 1, aa)
    if not cv2.imwrite(str(path), card, [cv2.IMWRITE_JPEG_QUALITY, 88]):
        raise BuildError(f"could not write {path}")


def week5_json_ld() -> str:
    data = {
        "@context": "https://schema.org",
        "@type": "SoftwareSourceCode",
        "name": "UAV Ground School Week 5: ROS 2 nodes and communication",
        "description": W5_DESCRIPTION,
        "url": CANONICAL + "week5",
        "image": CANONICAL + "og-week5.jpg",
        "codeRepository": REPO,
        "programmingLanguage": ["Python", "C++", "JavaScript"],
        "runtimePlatform": "ROS 2 Jazzy",
        "license": "https://opensource.org/licenses/MIT",
        "keywords": "ROS 2, Jazzy, DDS, RTPS, CDR, QoS, topics, services, actions, executors, callback groups, Fast DDS, Cyclone DDS, REP 2011",
        "author": {"@type": "Person", "name": "Karthik Subramanian"},
    }
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


# ------------------------------------------------------------------ week 6 ----

W6_DESCRIPTION = (
    "UAVs@Berkeley Ground School Week 6, solved: a ROS 2 action server and client that take a PX4 v1.17 quadcopter off to "
    "5 m in Gazebo through the Micro XRCE-DDS Agent, measured against Gazebo's ground truth, with DDS-XRCE rebuilt in pure Python."
)
W6_DATA = ("summary.json", "replay.json", "xrce.json", "NOTES.md")
W6_EXPORT = ("replay.json", "xrce.json")
W6_MAIN = "offboard_5m"


def required_week6(docs6: Path) -> list[Path]:
    return [docs6 / name for name in W6_DATA]


def _num(value, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def week6_flight_rows(summary: dict) -> str:
    rows = []
    for name, r in summary["flights"].items():
        title = summary["titles"].get(name, name)
        outcome = "succeeded" if r["succeeded"] else ("canceled" if r["result"] and "canceled" in r["result"] else "aborted")
        cells = [
            _cell(title, "name"),
            _cell(outcome, "ok" if r["succeeded"] else ""),
            _cell(f"{_num(r.get('peak_altitude'))} m", "num"),
            _cell(f"{_num(r.get('rise_time'))} s", "num"),
            _cell(f"{_num(r.get('max_climb_rate'))} m/s", "num"),
            _cell(f"{_num(r.get('overshoot'))} m", "num"),
            _cell(f"{_num(r.get('settling_time'), 1)} s", "num"),
            _cell(f"{_num(r.get('hover_rms_error'), 3)} m", "num"),
            _cell(f"{_num(r.get('max_drift'))} m", "num"),
            _cell("yes" if r.get("landed") else "no"),
        ]
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(rows)


def week6_topic_rows(session: dict) -> str:
    rows = []
    busy = [t for t in session["topics"] if t["samples"]]
    for t in sorted(busy, key=lambda t: (t["direction"] != "out", -t["samples"])):
        q = t["qos"]
        qos = f"{'reliable' if q.get('reliable') else 'best effort'}, {q.get('durability', '?')}"
        rate = f"{t['rate_hz']:.1f} Hz" if t["rate_hz"] else "n/a"
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(t['topic'])}</code></td><td><code>{html.escape(t['type'].split('/')[-1])}</code></td>"
            + _cell("publishes" if t["direction"] == "out" else "subscribes")
            + _cell(qos)
            + _cell(f"{t['samples']:,}", "num")
            + _cell(rate, "num")
            + _cell(t["sample_bytes"] if t["sample_bytes"] is not None else "n/a", "num")
            + "</tr>"
        )
    idle = len(session["topics"]) - len(busy)
    if idle:
        rows.append(f'<tr><td colspan="7" class="muted">and {idle} more topics PX4 created that carried nothing in this flight</td></tr>')
    return "\n".join(rows)


def week6_problems(notes: str) -> str:
    """The "### N. Title" entries of NOTES.md's problems section, with their symptom and fix."""
    part = notes.split("## Problems", 1)
    if len(part) < 2:
        raise BuildError("docs/week06/NOTES.md has no '## Problems' section")
    body = part[1].split("\n## ", 1)[0]
    items = []
    for block in re.split(r"\n### ", "\n" + body)[1:]:
        title, _, text = block.partition("\n")
        symptom = re.search(r"\*\*Symptom:\*\*\s*(.+)", text)
        fix = re.search(r"\*\*Fix:\*\*\s*(.+)", text)
        if not symptom or not fix:
            raise BuildError(f"NOTES.md problem {title!r} needs a **Symptom:** and a **Fix:** line")

        def inline(s: str) -> str:
            s = html.escape(s.strip())
            return re.sub(r"`([^`]+)`", r"<code>\1</code>", s)

        items.append(
            f"<li><b>{inline(re.sub(r'^[0-9]+[.] ', '', title))}</b>"
            f'<span class="symptom">{inline(symptom.group(1))}</span><span class="fix">{inline(fix.group(1))}</span></li>'
        )
    if not items:
        raise BuildError("docs/week06/NOTES.md lists no problems")
    return "\n".join(items)


def week6(docs6: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Stats and HTML fragments for week6.html."""
    summary = load_json(docs6 / "summary.json")
    xr = load_json(docs6 / "xrce.json")
    replay = load_json(docs6 / "replay.json")
    flights = summary.get("flights", {})
    for name in (W6_MAIN, "xrce_5m"):
        if name not in flights:
            raise BuildError(f"{docs6 / 'summary.json'} has no {name!r} flight")
    if not isinstance(replay, list) or not all({"id", "rows", "events", "metrics"} <= set(f) for f in replay):
        raise BuildError(f"{docs6 / 'replay.json'} must be a list of flights with rows, events and metrics")
    if not xr.get("packets") or not all({"hex", "spans", "title"} <= set(p) for p in xr["packets"]):
        raise BuildError(f"{docs6 / 'xrce.json'} must carry packets with hex, spans and title")
    main = flights[W6_MAIN]
    pure = flights["xrce_5m"]
    session = summary["xrce"]
    xlog = (docs6 / "flights" / "xrce_5m.log").read_text() if (docs6 / "flights" / "xrce_5m.log").is_file() else ""
    sent = re.search(r"(\d+) XRCE messages sent, (\d+) received", xlog)
    created = re.search(r"created (\d+) topics", xlog)
    from week06.lab import SCENARIOS

    stats = {
        "w6_px4": summary["versions"]["px4"],
        "w6_agent": summary["versions"]["agent"],
        "w6_hold": f"{main['target'] + main['hover_mean_error']:.2f}",
        "w6_hover_rms": f"{main['hover_rms_error']:.3f}",
        "w6_rise": f"{main['rise_time']:.2f}",
        "w6_overshoot": f"{main['overshoot']:.2f}",
        "w6_settle": f"{main['settling_time']:.1f}",
        "w6_truth": _num(main.get("estimate_vs_truth_rms"), 2),
        "w6_true_hold": _num(main.get("hover_truth_mean"), 2),
        "w6_topics": str(len(session["topics"])),
        "w6_created": str(session["created"]),
        "w6_kbps_out": f"{session['kbps']['px4>agent']:.0f}",
        "w6_kbps_in": f"{session['kbps']['agent>px4']:.0f}",
        "w6_lab_scenarios": str(len(SCENARIOS)),
        "w6_xrce_hold": f"{pure['target'] + pure['hover_mean_error']:.2f}"
        if pure.get("hover_mean_error") is not None
        else _num(pure.get("peak_altitude")),
        "w6_xrce_msgs": f"{int(sent.group(1)) + int(sent.group(2)):,}" if sent else "n/a",
        "w6_xrce_topics": created.group(1) if created else "7",
        "w6_px4_msgs": str(len(list((ROOT / "week06" / "interfaces" / "px4_msgs" / "msg").glob("*.msg")))),
    }
    tokens = {
        "W6_DESCRIPTION": html.escape(W6_DESCRIPTION),
        "W6_JSON_LD": week6_json_ld(),
        "W6_FLIGHT_ROWS": week6_flight_rows(summary),
        "W6_TOPIC_ROWS": week6_topic_rows(session),
        "W6_PROBLEMS": week6_problems((docs6 / "NOTES.md").read_text(encoding="utf-8")),
    }
    return stats, tokens


def export_week6(docs6: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in W6_EXPORT:
        shutil.copyfile(docs6 / name, out_dir / name)


def make_og_week6(path: Path, replay: list[dict], stats: dict[str, str]) -> None:
    """The 5 m climb itself: altitude against time, over the setpoint profile."""
    width, height = 1200, 630
    card = np.zeros((height, width, 3), np.uint8)
    card[:] = (29, 25, 18)
    for x in range(0, width, 40):
        cv2.line(card, (x, 0), (x, height), (42, 37, 27), 1)
    for y in range(0, height, 40):
        cv2.line(card, (0, y), (width, y), (42, 37, 27), 1)
    orange, white, muted, green = (60, 138, 240), (242, 240, 236), (200, 196, 190), (138, 211, 106)
    aa = cv2.LINE_AA
    flight = next(f for f in replay if f["id"] == W6_MAIN)
    rows = [r for r in flight["rows"] if r[4] in ("ARMING", "CLIMB", "SETTLE")] or flight["rows"]
    t0 = rows[0][0] - 2.0
    t1 = rows[-1][0] + 6.0
    x0, x1, y0, y1 = 640, 1140, 540, 120
    top = 6.0
    pts = lambda vals: np.array(  # noqa: E731
        [
            (int(x0 + (x1 - x0) * (r[0] - t0) / (t1 - t0)), int(y0 - (y0 - y1) * v / top))
            for r, v in vals
            if v is not None and t0 <= r[0] <= t1
        ],
        np.int32,
    )
    window = [r for r in flight["rows"] if t0 <= r[0] <= t1]
    cv2.line(card, (x0, y0), (x1, y0), (110, 100, 88), 2, aa)
    goal_y = int(y0 - (y0 - y1) * flight["target"] / top)
    for x in range(x0, x1, 14):
        cv2.line(card, (x, goal_y), (x + 6, goal_y), green, 2, aa)
    sp = pts([(r, r[2]) for r in window])
    for a, b in zip(sp[:-1:2], sp[1::2]):
        cv2.line(card, tuple(a), tuple(b), orange, 4, aa)
    cv2.polylines(card, [pts([(r, r[1]) for r in window])], False, white, 4, aa)
    cv2.putText(card, f"{flight['target']:.0f} m", (x1 - 70, goal_y - 14), cv2.FONT_HERSHEY_DUPLEX, 0.8, green, 2, aa)
    cv2.rectangle(card, (72, 150), (152, 156), orange, -1)
    cv2.putText(card, "Take off,", (66, 238), cv2.FONT_HERSHEY_TRIPLEX, 2.0, white, 3, aa)
    cv2.putText(card, "for real.", (66, 318), cv2.FONT_HERSHEY_TRIPLEX, 2.0, orange, 3, aa)
    for i, line in enumerate(("A ROS 2 action flying PX4 SITL", "in Gazebo, over uXRCE-DDS")):
        cv2.putText(card, line, (72, 392 + i * 46), cv2.FONT_HERSHEY_DUPLEX, 1.0, muted, 2, aa)
    cv2.putText(
        card,
        f"{stats['w6_hold']} m held, {stats['w6_rise']} s rise, {stats['w6_overshoot']} m overshoot",
        (72, 512),
        cv2.FONT_HERSHEY_DUPLEX,
        0.72,
        orange,
        2,
        aa,
    )
    cv2.putText(card, "UAVs@Berkeley Software Ground School 2026, Week 6", (72, 575), cv2.FONT_HERSHEY_SIMPLEX, 0.72, muted, 1, aa)
    if not cv2.imwrite(str(path), card, [cv2.IMWRITE_JPEG_QUALITY, 88]):
        raise BuildError(f"could not write {path}")


def week6_json_ld() -> str:
    data = {
        "@context": "https://schema.org",
        "@type": "SoftwareSourceCode",
        "name": "UAV Ground School Week 6: a ROS 2 takeoff action for PX4 SITL over uXRCE-DDS",
        "description": W6_DESCRIPTION,
        "url": CANONICAL + "week6",
        "image": CANONICAL + "og-week6.jpg",
        "codeRepository": REPO,
        "programmingLanguage": ["Python", "JavaScript"],
        "runtimePlatform": "ROS 2 Jazzy, PX4 v1.17, Gazebo Harmonic",
        "license": "https://opensource.org/licenses/MIT",
        "keywords": "PX4, SITL, Gazebo, uXRCE-DDS, Micro XRCE-DDS Agent, DDS-XRCE, ROS 2 actions, offboard control, px4_msgs, takeoff",
        "author": {"@type": "Person", "name": "Karthik Subramanian"},
    }
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


def week3_json_ld() -> str:
    data = {
        "@context": "https://schema.org",
        "@type": "SoftwareSourceCode",
        "name": "UAV Ground School Week 3: object detection",
        "description": W3_DESCRIPTION,
        "url": CANONICAL + "week3",
        "image": CANONICAL + "og-week3.jpg",
        "codeRepository": REPO,
        "programmingLanguage": ["Python", "C++"],
        "runtimePlatform": "OpenCV",
        "license": "https://opensource.org/licenses/MIT",
        "keywords": "blob detection, SimpleBlobDetector, Laplacian of Gaussian, Difference of Gaussians, Determinant of Hessian, YOLOv8, object detection, SAHI",
        "author": {"@type": "Person", "name": "Karthik Subramanian"},
    }
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


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
            elif not target.exists() and not target.suffix and target.with_suffix(".html").exists():
                target = target.with_suffix(".html")  # Cloudflare Pages serves page.html at /page
            if not target.exists():
                problems.append(f"{rel}: broken reference {ref!r}")
    if problems:
        raise BuildError("site verification failed:\n  " + "\n  ".join(problems))


# ------------------------------------------------------------------- build ----


def build(
    out: Path,
    docs: Path = DOCS,
    today: dt.date | None = None,
    docs3: Path = DOCS3,
    docs4: Path = DOCS4,
    docs5: Path = DOCS5,
    docs6: Path = DOCS6,
) -> Path:
    out = out.resolve()
    for protected in (ROOT, SITE, docs.resolve(), docs3.resolve(), docs4.resolve(), docs5.resolve(), docs6.resolve(), Path.home()):
        if out == protected or out in protected.parents:
            raise BuildError(f"refusing to overwrite {out}")
    missing3 = [str(p) for p in required_week3(docs3) if not p.is_file()]
    if missing3:
        raise BuildError("missing week 3 build inputs:\n  " + "\n  ".join(missing3))
    missing4 = [str(p) for p in required_week4(docs4) if not p.is_file()]
    if missing4:
        raise BuildError("missing week 4 build inputs:\n  " + "\n  ".join(missing4))
    missing5 = [str(p) for p in required_week5(docs5) if not p.is_file()]
    if missing5:
        raise BuildError("missing week 5 build inputs:\n  " + "\n  ".join(missing5))
    missing6 = [str(p) for p in required_week6(docs6) if not p.is_file()]
    if missing6:
        raise BuildError("missing week 6 build inputs:\n  " + "\n  ".join(missing6))
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
    (out / "assets" / "week3").mkdir(parents=True, exist_ok=True)
    for key, name in W3_IMAGES.items():
        info = export_image(docs3 / name, out / "assets" / "img")
        images[key] = info
    export_explorer(docs3, out / "assets" / "week3")
    head, bench_rows, w3_numbers = week3_bench(load_json(docs3 / "dots_benchmark.json"))
    piece_rows, piece_numbers = week3_pieces(load_json(docs3 / "objects.json"))
    piece_numbers.update(week3_beyond(load_json(docs3 / "beyond.json")))
    yolo_rows, yolo_notes, yolo_numbers = week3_yolo(load_json(docs3 / "yolo" / "results.json"), load_json(docs3 / "yolo" / "dataset.json"))
    reports = {name: parse_report((docs / "colors" / f"{name}_report.txt").read_text()) for name in DEMO_IMAGES}

    w4_numbers, w4_tokens = week4(docs4)
    export_week4(docs4, out / "assets" / "week4")
    w5_numbers, w5_tokens = week5(docs5)
    export_week5(docs5, out / "assets" / "week5")
    w6_numbers, w6_tokens = week6(docs6)
    export_week6(docs6, out / "assets" / "week6")
    numbers = {**stats(rows), **w3_numbers, **piece_numbers, **yolo_numbers, **w4_numbers, **w5_numbers, **w6_numbers}
    sentence, figures = stitcher_copy(rows, images)
    ctx = Context(
        tokens={
            "CANONICAL": CANONICAL,
            "REPO": REPO,
            "DESCRIPTION": html.escape(DESCRIPTION),
            "JSON_LD": json_ld(),
            "BENCHMARK_ROWS": benchmark_rows(rows),
            "STITCHER_SENTENCE": sentence,
            "STITCHER_FIGURES": figures,
            "REFERENCE_REPORTS": json.dumps(reports, separators=(",", ":")).replace("</", "<\\/"),
            "BUILD_DATE": today.isoformat(),
            "W3_DESCRIPTION": html.escape(W3_DESCRIPTION),
            "W3_JSON_LD": week3_json_ld(),
            "W3_REAL_HEAD": head,
            "W3_BENCH_ROWS": bench_rows,
            "W3_PIECE_ROWS": piece_rows,
            "W3_YOLO_ROWS": yolo_rows,
            "W3_YOLO_NOTES": yolo_notes,
            **w4_tokens,
            **w5_tokens,
            **w6_tokens,
        },
        stats=numbers,
        images=images,
    )
    for page in out.glob("*.html"):
        page.write_text(render(page.read_text(encoding="utf-8"), ctx, page), encoding="utf-8")

    make_og_image(read_image(docs / "rough_panorama.jpg"), out / "og.jpg", numbers["rough_best_rmse"])
    make_og_week3(read_image(docs3 / "hero.jpg"), out / "og-week3.jpg", numbers["w3_best_f1"])
    make_og_week4(None, out / "og-week4.jpg", w4_numbers)
    make_og_week5(out / "og-week5.jpg", w5_numbers)
    make_og_week6(out / "og-week6.jpg", load_json(docs6 / "replay.json"), w6_numbers)
    (out / "robots.txt").write_text(f"User-agent: *\nAllow: /\n\nSitemap: {CANONICAL}sitemap.xml\n")
    (out / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"  <url>\n    <loc>{CANONICAL}</loc>\n    <lastmod>{today.isoformat()}</lastmod>\n  </url>\n"
        f"  <url>\n    <loc>{CANONICAL}week3</loc>\n    <lastmod>{today.isoformat()}</lastmod>\n  </url>\n"
        f"  <url>\n    <loc>{CANONICAL}week4</loc>\n    <lastmod>{today.isoformat()}</lastmod>\n  </url>\n"
        f"  <url>\n    <loc>{CANONICAL}week5</loc>\n    <lastmod>{today.isoformat()}</lastmod>\n  </url>\n"
        f"  <url>\n    <loc>{CANONICAL}week6</loc>\n    <lastmod>{today.isoformat()}</lastmod>\n  </url>\n"
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

/og-week3.jpg
  Cache-Control: public, max-age=86400

/og-week4.jpg
  Cache-Control: public, max-age=86400

/og-week5.jpg
  Cache-Control: public, max-age=86400

/og-week6.jpg
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
    parser.add_argument("--docs3", default=str(DOCS3), help="week 3 docs directory")
    parser.add_argument("--docs4", default=str(DOCS4), help="week 4 docs directory")
    parser.add_argument("--docs5", default=str(DOCS5), help="week 5 docs directory")
    parser.add_argument("--docs6", default=str(DOCS6), help="week 6 docs directory")
    args = parser.parse_args(argv)
    try:
        out = build(
            Path(args.out), Path(args.docs), docs3=Path(args.docs3), docs4=Path(args.docs4), docs5=Path(args.docs5), docs6=Path(args.docs6)
        )
    except BuildError as error:
        print(f"build_site: error: {error}", file=sys.stderr)
        return 1
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"built {out} ({size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
