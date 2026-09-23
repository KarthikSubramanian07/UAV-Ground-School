import cv2
import numpy as np
import pytest

from week03 import augment, detector, detmetrics, football, slicing
from week03.detmetrics import Detections

# ------------------------------------------------------------ dataset tools ----


def make_dataset(root, clips=("aaaaaa", "bbbbbb", "cccccc"), frames=3, size=(320, 180)):
    """A tiny YOLO dataset with one colored box per frame, split like Roboflow does (leaky)."""
    rng = np.random.default_rng(0)
    splits = ["train", "valid", "test"]
    for c, clip in enumerate(clips):
        for f in range(frames):
            split = splits[(c + f) % 3]
            (root / split / "images").mkdir(parents=True, exist_ok=True)
            (root / split / "labels").mkdir(parents=True, exist_ok=True)
            image = np.full((size[1], size[0], 3), 90, np.uint8)
            x, y = int(rng.integers(20, size[0] - 60)), int(rng.integers(20, size[1] - 60))
            cv2.rectangle(image, (x, y), (x + 40, y + 30), (255, 255, 255), -1)
            name = f"{clip}_{f}_1_png.rf.{c}{f}"
            cv2.imwrite(str(root / split / "images" / f"{name}.jpg"), image)
            football.write_labels(
                root / split / "labels" / f"{name}.txt", football.from_xyxy(np.array([2]), np.array([[x, y, x + 40, y + 30]], float), *size)
            )
    return root


def test_clip_id():
    assert football.clip_id("08fd33_3_6_png.rf.fc5f6b621352712ee4e8fcdb56074c77.jpg") == "08fd33"


def test_label_round_trip(tmp_path):
    labels = np.array([[0, 0.5, 0.5, 0.1, 0.2], [3, 0.25, 0.75, 0.05, 0.05]])
    football.write_labels(tmp_path / "a.txt", labels)
    assert np.allclose(football.read_labels(tmp_path / "a.txt"), labels, atol=1e-6)
    boxes = football.to_xyxy(labels, 1920, 1080)
    assert np.allclose(football.from_xyxy(labels[:, 0], boxes, 1920, 1080), labels)
    assert football.read_labels(tmp_path / "missing.txt").shape == (0, 5)


def test_clip_split_removes_leakage(tmp_path):
    raw = make_dataset(tmp_path / "raw")
    assert football.stats(raw)["test_clips_seen_in_train"]  # the random split leaks
    football.clip_split(raw, tmp_path / "clean", test_clips=("aaaaaa",), valid_clips=("bbbbbb",))
    stats = football.stats(tmp_path / "clean")
    assert stats["test_clips_seen_in_train"] == []
    assert stats["clips"]["test"] == ["aaaaaa"] and stats["frames"]["train"] == 3
    assert (tmp_path / "clean" / "data.yaml").read_text().count("names") == 1


def test_tile_grid_covers_the_image():
    corners = football.tile_grid(1920, 1080, (960, 540), overlap=0.2)
    covered = np.zeros((1080, 1920), bool)
    for x, y in corners:
        covered[y : y + 540, x : x + 960] = True
    assert covered.all()
    assert football.tile_grid(1920, 1080, (960, 540), overlap=0.0) == [(0, 0), (960, 0), (0, 540), (960, 540)]


def test_tile_boxes_clip_and_drop():
    labels = football.from_xyxy(np.array([0, 1]), np.array([[900, 100, 1000, 200], [950, 300, 1100, 400]], float), 1920, 1080)
    left = football.tile_boxes(labels, 1920, 1080, 0, 0, 960, 540, min_visible=0.4)
    # Box 0 keeps 60% in the left tile; box 1 keeps only 7% and is dropped.
    assert len(left) == 1 and left[0, 0] == 0
    assert np.allclose(football.to_xyxy(left, 960, 540)[0], [900, 100, 960, 200])


def test_tile_dataset(tmp_path):
    raw = make_dataset(tmp_path / "raw", size=(640, 360))
    football.tile_dataset(raw, tmp_path / "tiles", tile=(320, 180), overlap=0.0)
    train = list((tmp_path / "tiles" / "train" / "images").glob("*.jpg"))
    assert len(train) == 4 * len(list((raw / "train" / "images").glob("*.jpg")))
    assert all(cv2.imread(str(p)).shape[:2] == (180, 320) for p in train)
    assert len(list((tmp_path / "tiles" / "test" / "images").glob("*.jpg"))) == len(list((raw / "test" / "images").glob("*.jpg")))


# ------------------------------------------------------------- augmentation ----


def box_pixels(image):
    ys, xs = np.nonzero(image[..., 0] > 200)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float)


@pytest.mark.parametrize(
    "recipe",
    [
        augment.Recipe(flip_horizontal=1.0),
        augment.Recipe(flip_horizontal=0.0, flip_vertical=1.0),
        augment.Recipe(flip_horizontal=0.0, rotate90=1.0),
        augment.Recipe(flip_horizontal=0.0, rotation=20, shear=10, crop=0.2),
    ],
)
def test_boxes_follow_the_pixels(recipe):
    rng = np.random.default_rng(3)
    image = np.full((180, 320, 3), 60, np.uint8)
    cv2.rectangle(image, (120, 60), (179, 109), (255, 255, 255), -1)
    labels = football.from_xyxy(np.array([1]), np.array([[120, 60, 180, 110]], float), 320, 180)
    out, out_labels = augment.apply(image, labels, recipe, rng)
    h, w = out.shape[:2]
    assert len(out_labels) == 1 and out_labels[0, 0] == 1
    predicted = football.to_xyxy(out_labels, w, h)[0]
    actual = box_pixels(out)
    # Rotation makes the label the bounding box of the rotated box, which contains the pixels.
    assert np.all(predicted[:2] <= actual[:2] + 2) and np.all(predicted[2:] >= actual[2:] - 2)
    if not recipe.rotation:
        assert np.abs(predicted - actual).max() <= 1.5


def test_boxes_pushed_out_of_frame_are_dropped():
    boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], float)
    shift = np.array([[1, 0, -8], [0, 1, 0]], float)
    clipped, visible = augment.warp_boxes(boxes, shift, 100, 100)
    assert visible[0] == pytest.approx(0.2) and visible[1] == pytest.approx(1.0)
    assert np.allclose(clipped[1], [42, 50, 52, 60])


def test_photometric_augmentations_keep_labels():
    rng = np.random.default_rng(0)
    image = np.full((90, 160, 3), 120, np.uint8)
    labels = np.array([[2, 0.5, 0.5, 0.2, 0.3]])
    out, out_labels = augment.apply(
        image,
        labels,
        augment.Recipe(
            flip_horizontal=0, hue=20, saturation=0.3, brightness=0.3, exposure=0.2, blur=2, noise=0.05, grayscale=1.0, cutout=2
        ),
        rng,
    )
    assert out.shape == image.shape and np.allclose(out_labels, labels)
    assert not np.array_equal(out, image)


def test_augment_dataset(tmp_path):
    raw = make_dataset(tmp_path / "raw")
    augment.augment_dataset(raw, tmp_path / "aug", augment.Recipe.preset("broadcast"), copies=2)
    n = len(list((raw / "train" / "images").glob("*.jpg")))
    assert len(list((tmp_path / "aug" / "train" / "images").glob("*.jpg"))) == 3 * n
    assert len(list((tmp_path / "aug" / "train" / "labels").glob("*.txt"))) == 3 * n
    with pytest.raises(ValueError):
        augment.Recipe.preset("nope")


# ------------------------------------------------------------------ metrics ----


def det(boxes, scores=None, classes=None):
    boxes = np.array(boxes, float).reshape(-1, 4)
    return Detections(
        boxes,
        np.array(scores if scores is not None else [1.0] * len(boxes), float),
        np.array(classes if classes is not None else [0] * len(boxes), int),
    )


def test_perfect_and_empty_predictions():
    truth = [det([[0, 0, 10, 10], [20, 20, 40, 40]])]
    assert detmetrics.evaluate(truth, truth, ("a",)).map == pytest.approx(1.0)
    empty = detmetrics.evaluate([Detections.empty()], truth, ("a",))
    assert empty.map50 == 0.0 and empty.map == 0.0


def test_a_false_positive_ranked_first_costs_precision():
    truth = [det([[0, 0, 10, 10]])]
    pred = [det([[50, 50, 60, 60], [0, 0, 10, 10]], scores=[0.9, 0.8])]
    report = detmetrics.evaluate(pred, truth, ("a",))
    # Recall reaches 1 at precision 0.5, so every interpolated point is 0.5.
    assert report.map50 == pytest.approx(0.5)


def test_iou_thresholds_are_averaged():
    truth = [det([[0, 0, 100, 100]])]
    pred = [det([[0, 0, 100, 80]])]  # IoU 0.8
    report = detmetrics.evaluate(pred, truth, ("a",))
    assert report.map50 == pytest.approx(1.0)
    assert report.map == pytest.approx(0.7)  # hits at 0.50 ... 0.80, misses 0.85 ... 0.95


def test_classes_are_kept_apart():
    truth = [det([[0, 0, 10, 10], [20, 20, 30, 30]], classes=[0, 1])]
    pred = [det([[0, 0, 10, 10], [20, 20, 30, 30]], classes=[1, 0])]
    assert detmetrics.evaluate(pred, truth, ("a", "b")).map50 == 0.0


def test_agrees_with_ultralytics():
    metrics = pytest.importorskip("ultralytics.utils.metrics")
    rng = np.random.default_rng(0)
    preds, truths = [], []
    for _ in range(20):
        n = int(rng.integers(3, 12))
        xy = rng.uniform(0, 500, (n, 2))
        wh = rng.uniform(10, 80, (n, 2))
        boxes = np.concatenate([xy, xy + wh], axis=1)
        classes = rng.integers(0, 3, n)
        truths.append(Detections(boxes, np.ones(n), classes))
        noisy = boxes + rng.normal(0, 4, boxes.shape)
        keep = rng.random(n) > 0.2
        extra = rng.uniform(0, 500, (3, 2))
        fp = np.concatenate([extra, extra + 40], axis=1)
        preds.append(
            Detections(
                np.concatenate([noisy[keep], fp]), rng.random(keep.sum() + 3), np.concatenate([classes[keep], rng.integers(0, 3, 3)])
            )
        )
    ours = detmetrics.evaluate(preds, truths, ("a", "b", "c"))
    # Ultralytics: per prediction true positive flags at each IoU threshold.
    tp, conf, pred_cls, target_cls = [], [], [], []
    for p, t in zip(preds, truths):
        iou = detmetrics.box_iou_matrix(p.boxes, t.boxes)
        correct = np.zeros((len(p), 10), bool)
        for k, thr in enumerate(detmetrics.IOU_THRESHOLDS):
            taken = np.zeros(len(t), bool)
            for i in np.argsort(-p.scores, kind="stable"):
                cand = np.where(~taken & (t.classes == p.classes[i]) & (iou[i] >= thr), iou[i], -1)
                j = int(np.argmax(cand)) if len(cand) else 0
                if len(cand) and cand[j] >= thr:
                    taken[j] = True
                    correct[i, k] = True
        tp.append(correct)
        conf.append(p.scores)
        pred_cls.append(p.classes)
        target_cls.append(t.classes)
    result = metrics.ap_per_class(np.concatenate(tp), np.concatenate(conf), np.concatenate(pred_cls), np.concatenate(target_cls))
    ap = result[5]  # (classes, 10)
    assert ours.map50 == pytest.approx(float(ap[:, 0].mean()), abs=0.02)
    assert ours.map == pytest.approx(float(ap.mean()), abs=0.02)


# ------------------------------------------------------------------ slicing ----


def test_nms_is_class_aware():
    boxes = det([[0, 0, 10, 10], [1, 1, 11, 11], [0, 0, 10, 10]], scores=[0.9, 0.8, 0.7], classes=[0, 0, 1])
    kept = slicing.nms(boxes, 0.5)
    assert len(kept) == 2 and sorted(kept.classes.tolist()) == [0, 1]
    assert slicing.overlap_matrix(np.array([[0, 0, 10, 10]], float), np.array([[0, 0, 5, 5]], float), "ios")[0, 0] == pytest.approx(1.0)


def test_sliced_predict_finds_small_objects_and_merges_duplicates():
    image = np.zeros((1080, 1920, 3), np.uint8)
    ball = (1000, 500, 1012, 512)  # sits in several overlapping tiles

    def predict(img):
        # A fake detector that only sees the ball when it is at least 6 px wide in its input.
        h, w = img.shape[:2]
        found = []
        if w <= 960:  # a tile at native resolution
            # Figure out where this crop is from its content marker.
            ys, xs = np.nonzero(img[..., 1] == 7)
            if len(xs):
                found.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        return det(found, scores=[0.9] * len(found), classes=[0] * len(found))

    image[ball[1] : ball[3], ball[0] : ball[2], 1] = 7
    whole = predict(image)
    assert len(whole) == 0
    merged = slicing.sliced_predict(predict, image, tile=(960, 540), overlap=0.2)
    assert len(merged) == 1
    assert np.allclose(merged.boxes[0], ball)


def test_boxes_cut_by_an_inner_tile_border_are_dropped():
    image = np.zeros((200, 400, 3), np.uint8)

    def predict(img):
        h, w = img.shape[:2]
        if w == 400:
            return det([[180, 50, 240, 150]], [0.8])  # the full frame sees the whole object
        return det([[w - 20, 50, w, 150]], [0.95])  # every tile reports a truncated box at its right edge

    merged = slicing.sliced_predict(predict, image, tile=(200, 200), overlap=0.0)
    assert len(merged) == 2  # the rightmost tile's box touches the image border, which is real
    assert any(np.allclose(b, [180, 50, 240, 150]) for b in merged.boxes)


def test_experiment_catalogue():
    assert "baseline" in detector.EXPERIMENTS and detector.EXPERIMENTS["tiles"].variant == "clip-tiles"
    assert detector.EXPERIMENTS["offline-aug"].variant == "clip-broadcastx2"
    assert detector.EXPERIMENTS["notebook-split"].split == "original"
