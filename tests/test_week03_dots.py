import math

import cv2
import numpy as np
import pytest

from week03 import blobs, dots, evaluate, synth


def score(scene, method, **kw):
    report = dots.find_dots(scene.image, method=method, **kw)
    truth = [blobs.Blob(d.x, d.y, d.radius) for d in scene.dots]
    h, w = scene.image.shape[:2]
    difficult = [d.x < 0 or d.y < 0 or d.x > w - 1 or d.y > h - 1 for d in scene.dots]
    return evaluate.match_dots(report.dots, truth, difficult), report, truth


@pytest.fixture(scope="module")
def flat():
    return synth.polka_scene(synth.SceneConfig(width=480, height=360, dots=25, seed=3))


@pytest.mark.parametrize("method", ["log", "dog", "doh", "contrast", "simple"])
def test_every_method_finds_dots_on_a_clean_print(flat, method):
    s, _, _ = score(flat, method)
    assert s.f1 >= 0.95


def test_centers_and_radii_are_sub_pixel_accurate(flat):
    s, report, truth = score(flat, "log")
    errors = [math.hypot(report.dots[i].x - truth[j].x, report.dots[i].y - truth[j].y) for i, j in s.matches]
    radius = [abs(report.dots[i].radius - truth[j].radius) / truth[j].radius for i, j in s.matches]
    assert np.median(errors) < 0.08
    assert np.median(radius) < 0.02


def test_the_edge_fit_is_what_makes_it_accurate(flat):
    fitted, report, truth = score(flat, "log")
    raw, raw_report, _ = score(flat, "log", refine=False)
    err = lambda s, r: np.median([math.hypot(r.dots[i].x - truth[j].x, r.dots[i].y - truth[j].y) for i, j in s.matches])  # noqa: E731
    assert err(fitted, report) < err(raw, raw_report)


def test_squares_stars_and_thin_ellipses_are_rejected():
    scene = synth.polka_scene(synth.SceneConfig(width=480, height=360, dots=15, distractors=20, seed=5))
    s, _, _ = score(scene, "log")
    assert s.fp == 0
    assert s.recall >= 0.9


def test_color_and_size_filters():
    image = np.full((200, 400, 3), 245, np.uint8)
    cv2.circle(image, (60, 100), 30, (40, 40, 220), -1, cv2.LINE_AA)  # red, large
    cv2.circle(image, (180, 100), 12, (40, 40, 220), -1, cv2.LINE_AA)  # red, small
    cv2.circle(image, (300, 100), 30, (60, 190, 60), -1, cv2.LINE_AA)  # green, large
    everything = dots.find_dots(image, method="log")
    assert sorted(d.color for d in everything.dots) == ["green", "red", "red"]
    reds = dots.find_dots(image, method="log", colors=("red",))
    assert len(reds.dots) == 2
    big = dots.find_dots(image, method="log", min_radius=20)
    assert len(big.dots) == 2 and all(d.radius > 25 for d in big.dots)


def test_grayscale_baseline_misses_isoluminant_dots():
    # An orange with exactly the gray value of the background: invisible in grayscale.
    background, dot = (120, 120, 120), (40, 95, 200)
    to_gray = lambda c: int(cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_BGR2GRAY)[0, 0])  # noqa: E731
    assert to_gray(background) == to_gray(dot)
    image = np.full((200, 300, 3), background, np.uint8)
    cv2.circle(image, (80, 100), 30, dot, -1)
    cv2.circle(image, (210, 100), 30, dot, -1)
    assert len(dots.find_dots(image, method="gray").dots) == 0
    assert len(dots.find_dots(image, method="log").dots) == 2


def test_gap_between_dots_is_not_a_dot():
    # White space enclosed by four colored dots looks like a white blob to LoG.
    image = np.full((200, 200, 3), 250, np.uint8)
    for (x, y), c in zip([(60, 60), (140, 60), (60, 140), (140, 140)], [(40, 40, 220), (60, 190, 60), (220, 170, 20), (200, 60, 150)]):
        cv2.circle(image, (x, y), 38, c, -1, cv2.LINE_AA)
    found = dots.find_dots(image, method="log")
    assert len(found.dots) == 4
    assert all(d.color != "white" for d in found.dots)


def test_dots_cut_by_the_border_are_found():
    image = np.full((200, 300, 3), 245, np.uint8)
    cv2.circle(image, (8, 100), 30, (40, 40, 220), -1, cv2.LINE_AA)
    cv2.circle(image, (150, 100), 30, (40, 40, 220), -1, cv2.LINE_AA)
    found = dots.find_dots(image, method="log")
    assert len(found.dots) == 2
    edge = min(found.dots, key=lambda d: d.x)
    assert edge.x == pytest.approx(8, abs=1.0) and edge.radius == pytest.approx(30, rel=0.06)


def test_color_names():
    assert dots.color_name(dots.lab_image(np.uint8([[[40, 40, 220]]]))[0, 0]) == "red"
    assert dots.color_name(dots.lab_image(np.uint8([[[240, 225, 205]]]))[0, 0]) == "light blue"
    assert dots.color_name(dots.lab_image(np.uint8([[[210, 190, 245]]]))[0, 0]) == "pink"
    assert dots.color_name(dots.lab_image(np.uint8([[[245, 245, 245]]]))[0, 0]) == "white"


def test_palette_finds_every_color():
    image = np.full((200, 300, 3), 245, np.uint8)
    for i, c in enumerate([(40, 40, 220), (60, 190, 60), (200, 60, 20)]):
        cv2.circle(image, (60 + 90 * i, 100), 35, c, -1)
    palette = dots.learn_palette(dots.lab_image(image), k=4)
    assert {"red", "green", "blue", "white"} <= set(palette.names)


def test_report_summary_and_annotation(flat):
    report = dots.find_dots(flat.image, method="doh")
    summary = report.summary()
    assert summary["count"] == len(report.dots) and summary["method"] == "doh"
    assert sum(report.counts().values()) == len(report.dots)
    annotated = dots.annotate(flat.image, report.dots, label=True)
    assert annotated.shape == flat.image.shape and not np.array_equal(annotated, flat.image)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        dots.find_dots(np.zeros((10, 10), np.uint8))
    with pytest.raises(ValueError):
        dots.find_dots(np.zeros((50, 50, 3), np.uint8), method="nope")
