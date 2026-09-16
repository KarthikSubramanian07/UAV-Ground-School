import json

from week02 import benchmark, synth


def test_benchmark_writes_report(tmp_path, monkeypatch):
    tiny = [
        benchmark.Scenario("tiny", "2 passes", synth.FlightPlan(frame_width=320, frame_height=180, passes=2, speed=8.0, seed=3), 110),
        benchmark.Scenario("tiny_lens", "2 passes with distortion", synth.FlightPlan.hard(frame_width=320, frame_height=180, passes=2, speed=8.0, seed=3), 120),
    ]
    monkeypatch.setattr(benchmark, "scenarios", lambda seed: tiny)
    original = synth.voxel_world
    monkeypatch.setattr(synth, "voxel_world", lambda height_blocks, seed: original(width_blocks=200, height_blocks=height_blocks, seed=seed))

    def small_stitcher(frames, max_frames=None):
        raise benchmark.stitching.StitchError("skipped in tests")

    monkeypatch.setattr(benchmark.stitching, "stitch_with_opencv", small_stitcher)
    rows = benchmark.run_benchmark(tmp_path, seed=3, log=lambda message: None)

    variants = [row["variant"] for row in rows]
    assert "sequential chaining" in variants and "+ lens undistortion (full pipeline)" in variants
    assert any(row.get("status") == "skipped in tests" for row in rows)
    assert all(row["rmse_px"] < 5 for row in rows if row["scenario"] == "tiny" and "rmse_px" in row)
    assert json.loads((tmp_path / "benchmark.json").read_text()) == rows
    markdown = (tmp_path / "benchmark.md").read_text()
    assert "| tiny | sequential chaining |" in markdown
    for name in ["tiny_world.jpg", "tiny_frame.jpg", "tiny_sequential.jpg", "tiny_panorama.jpg", "tiny_path.jpg"]:
        assert (tmp_path / name).exists(), name
