import cv2
import numpy as np
import pytest

from week02 import colors, synth


def test_named_bands_partition_every_possible_color():
    """Every HSV value belongs to exactly one named color."""
    h, s, v = np.meshgrid(np.arange(180), np.arange(0, 256, 3), np.arange(0, 256, 3), indexing="ij")
    hsv = np.stack([h, s, v], axis=-1).reshape(-1, 1, 3).astype(np.uint8)
    total = np.zeros(hsv.shape[:2], np.int32)
    for band in colors.COLOR_BANDS:
        total += colors.color_mask(hsv, band) // 255
    assert total.min() == 1 and total.max() == 1


@pytest.mark.parametrize(
    "bgr, name",
    [
        ((0, 0, 255), "red"),
        ((20, 20, 180), "red"),
        ((0, 140, 255), "orange"),
        ((0, 230, 255), "yellow"),
        ((40, 90, 130), "brown"),
        ((0, 200, 0), "green"),
        ((230, 220, 0), "cyan"),
        ((255, 60, 0), "blue"),
        ((200, 40, 140), "purple"),
        ((200, 100, 255), "pink"),
        ((10, 10, 10), "black"),
        ((128, 128, 128), "gray"),
        ((250, 250, 250), "white"),
    ],
)
def test_name_color(bgr, name):
    assert colors.name_color(bgr) == name


def test_mask_center_matches_image_moments():
    rng = np.random.default_rng(0)
    mask = (rng.random((120, 200)) > 0.7).astype(np.uint8) * 255
    moments = cv2.moments(mask, binaryImage=True)
    cx, cy = colors.mask_center(mask)
    assert cx == pytest.approx(moments["m10"] / moments["m00"])
    assert cy == pytest.approx(moments["m01"] / moments["m00"])
    assert colors.mask_center(np.zeros((5, 5), np.uint8)) is None


def test_layers_partition_the_image():
    image = synth.stop_sign()
    layers = colors.split_colors(image, min_coverage=0, min_object_fraction=0)
    stacked = sum((layer.mask // 255).astype(np.int32) for layer in layers)
    assert np.all(stacked == 1)
    assert sum(layer.pixels for layer in layers) == image.shape[0] * image.shape[1]


def test_isolated_layer_keeps_only_its_pixels():
    image = synth.stop_sign()
    red = next(layer for layer in colors.split_colors(image) if layer.name == "red")
    isolated = red.isolate(image)
    assert np.all(isolated[red.mask == 0] == 0)
    assert np.array_equal(isolated[red.mask > 0], image[red.mask > 0])
    bgra = red.to_bgra(image)
    assert bgra.shape[2] == 4 and np.array_equal(bgra[:, :, 3], red.mask)


def test_stop_sign_is_a_red_octagon_at_the_right_place():
    image = synth.stop_sign(width=900, height=675)
    layers = {layer.name: layer for layer in colors.split_colors(image)}
    assert {"red", "white", "blue", "green"} <= set(layers)
    sign = layers["red"].objects[0]
    assert sign.shape == "octagon"
    assert sign.center[0] == pytest.approx(450, abs=3)
    assert sign.center[1] == pytest.approx(0.40 * 675, abs=6)  # letters punch holes, shifting it slightly


def test_every_target_found_with_color_shape_and_center(targets):
    image, truths = targets
    layers = {layer.name: layer for layer in colors.split_colors(image)}
    for truth in truths:
        objects = layers[truth.color].objects
        best = min(objects, key=lambda o: np.hypot(o.center[0] - truth.center[0], o.center[1] - truth.center[1]))
        # The printed letter removes pixels, so allow a few pixels of shift.
        assert np.hypot(best.center[0] - truth.center[0], best.center[1] - truth.center[1]) < 6
        assert best.shape == truth.shape


def test_target_letters_are_their_own_objects(targets):
    image, truths = targets
    layers = {layer.name: layer for layer in colors.split_colors(image)}
    for truth in truths:
        letters = layers[truth.letter_color].objects
        assert any(np.hypot(o.center[0] - truth.center[0], o.center[1] - truth.center[1]) < truth.radius * 0.4 for o in letters)


def test_auto_mode_discovers_palette():
    image = np.zeros((200, 300, 3), np.uint8)
    image[:, :100] = (30, 30, 200)
    image[:, 100:200] = (40, 180, 40)
    image[:, 200:] = (200, 90, 20)
    image[60:140, 130:170] = (240, 240, 240)
    noisy = np.clip(image + np.random.default_rng(0).normal(0, 4, image.shape), 0, 255).astype(np.uint8)
    layers = colors.split_colors(noisy, mode="auto")
    assert sorted(layer.name for layer in layers) == ["blue", "green", "red", "white"]
    white = next(layer for layer in layers if layer.name == "white")
    assert white.objects[0].shape == "rectangle"
    assert white.objects[0].center == pytest.approx((149.5, 99.5), abs=0.6)


def test_auto_mode_with_fixed_k():
    image = synth.apple()
    layers = colors.split_colors(image, mode="auto", k=3)
    assert 2 <= len(layers) <= 3
    assert sum(layer.pixels for layer in layers) <= image.shape[0] * image.shape[1]


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        colors.split_colors(np.zeros((10, 10), np.uint8))
    with pytest.raises(ValueError):
        colors.split_colors(np.zeros((10, 10, 3), np.float32))
    with pytest.raises(ValueError):
        colors.split_colors(np.zeros((10, 10, 3), np.uint8), mode="rainbow")


def test_load_image_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        colors.load_image(tmp_path / "missing.png")
    bogus = tmp_path / "bogus.png"
    bogus.write_text("not an image")
    with pytest.raises(ValueError):
        colors.load_image(bogus)


def test_report_and_outputs(tmp_path, targets):
    image, _ = targets
    layers = colors.split_colors(image)
    report = colors.format_report(layers)
    assert "octagon" in report and "object 1: center" in report
    written = colors.save_layers(image, layers, tmp_path)
    names = {path.name for path in written}
    assert "contact_sheet.png" in names and "centers.png" in names and "layer_red.png" in names
    layer = cv2.imread(str(tmp_path / "layer_red.png"), cv2.IMREAD_UNCHANGED)
    assert layer.shape == (*image.shape[:2], 4)
    summary = layers[0].summary()
    assert "mask" not in summary and summary["name"] == layers[0].name
