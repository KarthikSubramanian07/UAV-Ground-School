"""Option 2: train YOLOv8 on the football dataset and try to improve it.

Every experiment is a named recipe (dataset variant plus training settings).
``run`` prepares the data, trains with Ultralytics, then scores the best
checkpoint on the held out test clips with :mod:`week03.detmetrics`, both on
whole frames and with sliced inference, so all numbers are comparable.

Compute budget: the notebook trains YOLOv8s at 800 px on a Colab T4. This
runs on an 8 GB Apple M1, where YOLOv8s at 800 px does not fit in memory, so
the experiments use YOLOv8n and compare recipes at equal budgets.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import augment, detmetrics, football, slicing


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    split: str = "clip"  # clip (leakage free) or original (Roboflow's random split)
    model: str = "yolov8n.pt"
    imgsz: int = 640
    epochs: int = 30
    batch: int = 8
    tiles: bool = False  # train on 2x2 tiles instead of whole frames
    augment: str | None = None  # offline Roboflow style recipe
    copies: int = 2
    sliced_eval: bool = False  # also score sliced inference
    weights_from: str | None = None  # score another experiment's weights instead of training
    extra: dict = field(default_factory=dict)  # passed to Ultralytics train()

    @property
    def variant(self) -> str:
        parts = [self.split]
        if self.tiles:
            parts.append("tiles")
        if self.augment:
            parts.append(f"{self.augment}x{self.copies}")
        return "-".join(parts)


EXPERIMENTS: dict[str, Experiment] = {
    e.name: e
    for e in [
        Experiment("notebook-split", "Notebook recipe on Roboflow's random split (clips leak into test)", split="original"),
        Experiment("baseline", "Notebook recipe on the leakage free clip split"),
        Experiment(
            "offline-aug",
            "Plus Roboflow style offline augmentation (2 extra copies), equal step budget",
            augment="broadcast",
            copies=2,
            epochs=10,
        ),
        # Training at 1280 px does not fit this machine (over 25 minutes per epoch, swapping), so only inference is scaled up.
        Experiment("hires-inference", "Baseline weights, inference at 1280 px", imgsz=1280, weights_from="baseline"),
        Experiment(
            "tiles", "Train on 960x540 tiles at 640 px, equal step budget, sliced inference", tiles=True, epochs=8, sliced_eval=True
        ),
        Experiment("tiles-long", "Tiles for 4x the steps, sliced inference", tiles=True, epochs=30, sliced_eval=True),
    ]
}


def device() -> str:
    import torch

    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def prepare(exp: Experiment, raw_data: str | Path, root: str | Path) -> Path:
    """Build (or reuse) the dataset directory for an experiment; returns its data.yaml."""
    root = Path(root)
    base = root / "datasets" / exp.split
    if not (base / "data.yaml").is_file():
        (football.original_split if exp.split == "original" else football.clip_split)(raw_data, base)
    current = base
    if exp.tiles:
        tiled = root / "datasets" / f"{exp.split}-tiles"
        if not (tiled / "data.yaml").is_file():
            football.tile_dataset(current, tiled, tile=(960, 540), overlap=0.0)
        current = tiled
    if exp.augment:
        augmented = root / "datasets" / exp.variant
        if not (augmented / "data.yaml").is_file():
            augment.augment_dataset(current, augmented, augment.Recipe.preset(exp.augment), copies=exp.copies)
        current = augmented
    return current / "data.yaml"


def train(exp: Experiment, data_yaml: Path, root: str | Path) -> Path:
    from ultralytics import YOLO

    root = Path(root)
    weights = root / "runs" / (exp.weights_from or exp.name) / "weights" / "best.pt"
    if weights.is_file():
        return weights
    if exp.weights_from:
        raise FileNotFoundError(f"{exp.name} scores the weights of {exp.weights_from}; train that first")
    model = YOLO(exp.model)
    model.train(
        data=str(data_yaml),
        imgsz=exp.imgsz,
        epochs=exp.epochs,
        batch=exp.batch,
        device=device(),
        workers=2,
        seed=0,
        deterministic=True,
        project=str((root / "runs").resolve()),
        name=exp.name,
        exist_ok=True,
        plots=True,
        amp=False,
        verbose=False,
        **exp.extra,
    )
    return weights


def ultralytics_predictor(weights: Path, imgsz: int, conf: float = 0.001):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    dev = device()

    def predict(image: np.ndarray) -> detmetrics.Detections:
        result = model.predict(image, imgsz=imgsz, conf=conf, iou=0.6, device=dev, verbose=False, max_det=300)[0]
        boxes = result.boxes
        return detmetrics.Detections(
            boxes.xyxy.cpu().numpy().astype(np.float64), boxes.conf.cpu().numpy().astype(np.float64), boxes.cls.cpu().numpy().astype(int)
        )

    return predict


def load_truth(split_dir: Path, split: str = "test") -> tuple[list[Path], list[detmetrics.Detections]]:
    images = sorted((split_dir / split / "images").glob("*.jpg"))
    truths = []
    for image_path in images:
        h, w = cv2.imread(str(image_path)).shape[:2]
        labels = football.read_labels(split_dir / split / "labels" / f"{image_path.stem}.txt")
        truths.append(
            detmetrics.Detections(
                football.to_xyxy(labels, w, h), np.ones(len(labels)), labels[:, 0].astype(int) if len(labels) else np.zeros(0, int)
            )
        )
    return images, truths


def evaluate(exp: Experiment, weights: Path, root: str | Path) -> dict:
    """Score a trained model on the test split of its (untiled, unaugmented) base dataset."""
    root = Path(root)
    base = root / "datasets" / exp.split
    images, truths = load_truth(base)
    predict = ultralytics_predictor(weights, exp.imgsz)
    result: dict = {"experiment": asdict(exp), "test_frames": len(images)}

    start = time.perf_counter()
    preds = [predict(cv2.imread(str(p))) for p in images]
    result["full_frame"] = {
        **detmetrics.evaluate(preds, truths, football.NAMES).summary(),
        "ms_per_frame": round(1000 * (time.perf_counter() - start) / max(len(images), 1), 1),
    }

    if exp.sliced_eval:
        start = time.perf_counter()
        preds = [slicing.sliced_predict(predict, cv2.imread(str(p)), tile=(960, 540), overlap=0.2) for p in images]
        result["sliced"] = {
            **detmetrics.evaluate(preds, truths, football.NAMES).summary(),
            "ms_per_frame": round(1000 * (time.perf_counter() - start) / max(len(images), 1), 1),
        }
    return result


def run(name: str, raw_data: str | Path, root: str | Path = "out/yolo") -> dict:
    exp = EXPERIMENTS[name]
    root = Path(root)
    data_yaml = prepare(exp, raw_data, root)
    start = time.perf_counter()
    weights = train(exp, data_yaml, root)
    train_seconds = time.perf_counter() - start
    result = evaluate(exp, weights, root)
    result["train_seconds"] = round(train_seconds, 1)
    results_path = root / "results.json"
    results = json.loads(results_path.read_text()) if results_path.is_file() else {}
    previous = results.get(name, {})
    if previous.get("train_seconds") and train_seconds < 60:
        result["train_seconds"] = previous["train_seconds"]  # weights were reused; keep the real training time
    results[name] = result
    results_path.write_text(json.dumps(results, indent=2))
    return result


def to_markdown(results: dict) -> str:
    lines = [
        "# Week 3 YOLOv8 experiments",
        "",
        "All models are YOLOv8n trained on an 8 GB Apple M1 and scored by `week03/detmetrics.py` on the test split (COCO style AP).",
        "",
        "| Experiment | Split | Inference | mAP50 | mAP50-95 | Ball AP50 | Goalkeeper AP50 | Player AP50 | Referee AP50 | ms / frame | Train time |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, r in results.items():
        exp = r["experiment"]
        for mode in ("full_frame", "sliced"):
            if mode not in r:
                continue
            m = r[mode]
            c = m["classes"]
            lines.append(
                f"| `{name}` | {exp['split']} | {'sliced' if mode == 'sliced' else 'whole frame'} {exp['imgsz']} px | {m['mAP50']:.3f} | {m['mAP50-95']:.3f} | "
                f"{c['ball']['AP50']:.3f} | {c['goalkeeper']['AP50']:.3f} | {c['player']['AP50']:.3f} | {c['referee']['AP50']:.3f} | {m['ms_per_frame']:.0f} | {r.get('train_seconds', 0) / 60:.0f} min |"
            )
    lines += ["", "Experiments:", ""]
    for name, r in results.items():
        lines.append(f"* `{name}`: {r['experiment']['description']}.")
    return "\n".join(lines) + "\n"


def main_run(names: list[str], raw_data: str, root: str) -> None:
    os.environ.setdefault("YOLO_VERBOSE", "False")
    for name in names:
        print(f"== {name}: {EXPERIMENTS[name].description}", flush=True)
        result = run(name, raw_data, root)
        print(json.dumps({k: v for k, v in result.items() if k != "experiment"}, indent=1), flush=True)


# ------------------------------------------------------------ leak check ----


def leak_check(root: str | Path) -> dict | None:
    """Score the leaky and the clean model on the same frames.

    The frames are Roboflow test frames from the clips the clean split holds
    out. Neither model trained on these exact frames, but the leaky model
    trained on other frames of the same clips, often a second apart; the clean
    model never saw those clips. Same frames, same scorer, so the difference is
    the leak.
    """
    root = Path(root)
    leaky = root / "runs" / "notebook-split" / "weights" / "best.pt"
    clean = root / "runs" / "baseline" / "weights" / "best.pt"
    if not (leaky.is_file() and clean.is_file()):
        return None
    original = root / "datasets" / "original"
    images, truths = load_truth(original, "test")
    shared = [(p, t) for p, t in zip(images, truths) if football.clip_id(p.name) in football.TEST_CLIPS]
    if not shared:
        return None
    out = {"frames": len(shared), "clips": list(football.TEST_CLIPS)}
    for name, weights in (("leaky", leaky), ("clean", clean)):
        predict = ultralytics_predictor(weights, 640)
        preds = [predict(cv2.imread(str(p))) for p, _ in shared]
        out[name] = detmetrics.evaluate(preds, [t for _, t in shared], football.NAMES).summary()
    return out


# ---------------------------------------------------------------- figures ----

COLORS = {0: (0, 215, 255), 1: (255, 140, 40), 2: (90, 220, 90), 3: (80, 80, 245)}


def draw(image: np.ndarray, det: detmetrics.Detections, conf: float = 0.35) -> np.ndarray:
    out = image.copy()
    scale = max(image.shape[:2]) / 1920
    for box, score, c in zip(det.boxes, det.scores, det.classes):
        if score < conf:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        color = COLORS[int(c) % 4]
        if int(c) == 0:  # the ball is tiny: ring it so it can be seen
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(out, (cx, cy), round(22 * scale), (255, 255, 255), round(5 * scale), cv2.LINE_AA)
            cv2.circle(out, (cx, cy), round(22 * scale), color, round(3 * scale), cv2.LINE_AA)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, max(2, round(2 * scale)), cv2.LINE_AA)
    return out


def _ball_recall(det: detmetrics.Detections, truth: detmetrics.Detections, conf: float = 0.35) -> float:
    balls = truth.boxes[truth.classes == 0]
    if not len(balls):
        return -1.0
    keep = (det.classes == 0) & (det.scores >= conf)
    iou = detmetrics.box_iou_matrix(det.boxes[keep], balls)
    return float((iou.max(axis=0) >= 0.3).mean()) if iou.size else 0.0


def figures(root: str | Path, out: str | Path, raw_data: str | Path) -> dict:
    """Write docs/week03/yolo: results, dataset stats and notes, and a whole versus sliced example."""
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    results = json.loads((root / "results.json").read_text())
    order = [name for name in EXPERIMENTS if name in results]
    results = {name: results[name] for name in order}
    (out / "results.json").write_text(json.dumps(results, indent=2))
    (out / "results.md").write_text(to_markdown(results))

    dataset = football.stats(raw_data)
    leak = leak_check(root)
    if leak:
        dataset["leak_check"] = leak
    notes, headline = _notes(results, leak)
    dataset["notes"] = notes
    dataset["headline"] = headline
    (out / "dataset.json").write_text(json.dumps(dataset, indent=2))

    sliced_runs = [n for n in order if "sliced" in results[n]]
    if sliced_runs:
        best = max(sliced_runs, key=lambda n: results[n]["sliced"]["mAP50"])
        exp = EXPERIMENTS[best]
        predict = ultralytics_predictor(root / "runs" / best / "weights" / "best.pt", exp.imgsz)
        images, truths = load_truth(root / "datasets" / exp.split)
        # The frame where slicing helps the ball most (ties: most balls, then first).
        scored = []
        for path, truth in zip(images, truths):
            image = cv2.imread(str(path))
            whole = predict(image)
            sliced = slicing.sliced_predict(predict, image)
            gain = _ball_recall(sliced, truth) - _ball_recall(whole, truth)
            scored.append((gain, int((truth.classes == 0).sum()), path, whole, sliced))
        gain, _, path, whole, sliced = max(scored, key=lambda s: (s[0], s[1]))
        image = cv2.imread(str(path))
        cv2.imwrite(str(out / "whole.jpg"), draw(image, whole), [cv2.IMWRITE_JPEG_QUALITY, 88])
        cv2.imwrite(str(out / "sliced.jpg"), draw(image, sliced), [cv2.IMWRITE_JPEG_QUALITY, 88])
        dataset["example"] = {"experiment": best, "frame": path.name, "ball_recall_gain": gain}
        (out / "dataset.json").write_text(json.dumps(dataset, indent=2))
    for name in order:
        curves = root / "runs" / name / "results.png"
        if curves.is_file():
            shutil.copyfile(curves, out / f"{name}_curves.png")
    return dataset


def _notes(results: dict, leak: dict | None = None) -> tuple[list[str], str]:
    """Plain sentences comparing the experiments that exist."""

    def m(name: str, mode: str = "full_frame", key: str = "mAP50") -> float | None:
        r = results.get(name, {}).get(mode)
        return None if r is None else r[key]

    def ball(name: str, mode: str = "full_frame") -> float | None:
        r = results.get(name, {}).get(mode)
        return None if r is None else r["classes"]["ball"]["AP50"]

    notes = []
    base = m("baseline")
    if base is not None and m("notebook-split") is not None:
        notes.append(
            f"On their own test sets the same recipe scores {m('notebook-split'):.3f} mAP50 on Roboflow's split and {base:.3f} on held out clips. "
            "Those are different frames, so the gap mixes the leak with how hard each test set is."
        )
    if leak:
        lk, cl = leak["leaky"], leak["clean"]
        notes.append(
            f"A controlled check scores both models on the same {leak['frames']} frames: Roboflow test frames from the two held out clips, "
            f"whose neighboring frames the leaky model trained on and the clean model never saw. The leaky model scores {lk['mAP50']:.3f} mAP50 and the clean one {cl['mAP50']:.3f} "
            f"({lk['mAP50'] - cl['mAP50']:+.3f}); on referees, the class that depends most on context, {lk['classes']['referee']['AP50']:.3f} against {cl['classes']['referee']['AP50']:.3f}. "
            f"{leak['frames']} frames is a small sample, and the leaky model also had more training frames (298 against 259), but both comparisons point the same way. "
            "Every row below the first uses held out clips."
        )
    if base is not None:
        notes.append(
            f"The baseline finds players well but the ball not at all: ball AP50 {ball('baseline'):.3f}, because a 12 px ball is 4 px after resizing to 640."
        )
    if "hires" in results and base is not None:
        notes.append(
            f"Training and testing at 1280 px instead of 640 moves mAP50 to {m('hires'):.3f} and ball AP50 to {ball('hires'):.3f}, at four times the compute per image."
        )
    if "offline-aug" in results and base is not None:
        notes.append(
            f"Roboflow style offline augmentation (two extra copies, the same number of training steps) gives {m('offline-aug'):.3f} mAP50. "
            "Ultralytics already augments online with mosaic, HSV jitter, flips and scaling, so the extra copies mostly repeat what it does."
        )
    for name in ("tiles", "tiles-long"):
        if name in results and "sliced" in results[name]:
            notes.append(
                f"{name}: trained on 960x540 tiles, then run on overlapping tiles plus the whole frame and merged. "
                f"mAP50 {m(name, 'sliced'):.3f} and ball AP50 {ball(name, 'sliced'):.3f}, against {m(name):.3f} and {ball(name):.3f} for the same weights on whole frames."
            )
    best = max(
        (
            (n, mode)
            for n in results
            for mode in ("full_frame", "sliced")
            if mode in results[n] and results[n]["experiment"]["split"] == "clip"
        ),
        key=lambda nm: results[nm[0]][nm[1]]["mAP50"],
        default=None,
    )
    headline = ""
    if best and base is not None:
        n, mode = best
        headline = f"mAP50 {base:.3f} to {m(n, mode):.3f} on held out clips, ball AP50 {ball('baseline'):.2f} to {ball(n, mode):.2f}"
    return notes, headline
