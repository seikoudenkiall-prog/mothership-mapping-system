import json

import numpy as np
import pytest

from src.benchmark import generate_benchmark_report
from src.exporter import export_point_cloud, export_voxels


def _read_ply_vertices(path):
    lines = path.read_text(encoding="ascii").splitlines()
    count = int(next(line.split()[2] for line in lines if line.startswith("element vertex")))
    header_end = lines.index("end_header")
    return np.loadtxt(lines[header_end + 1:], dtype=float).reshape(count, 3)


def _read_obj_vertices(path):
    vertices = [
        [float(value) for value in line.split()[1:4]]
        for line in path.read_text(encoding="ascii").splitlines()
        if line.startswith("v ")
    ]
    return np.asarray(vertices, dtype=float).reshape(-1, 3)


def test_ply_and_obj_exports_preserve_vertices_within_30mm(tmp_path):
    points = np.array([
        [15.0, 15.0, 1.8],
        [30.0, 25.0, 2.0],
        [20.0, 32.0, 1.4],
    ])
    ply_path = export_point_cloud(points, tmp_path / "scan.ply")
    obj_path = export_point_cloud(points, tmp_path / "scan.obj")

    for restored in (_read_ply_vertices(ply_path), _read_obj_vertices(obj_path)):
        assert restored.shape == points.shape
        assert np.all(np.max(np.abs(restored - points), axis=0) * 1000 <= 30.0)


def test_voxel_grid_exports_occupied_voxel_centers(tmp_path):
    voxels = np.zeros((2, 2, 2), dtype=bool)
    voxels[0, 1, 1] = True
    voxels[1, 0, 0] = True

    path = export_voxels(
        voxels, tmp_path / "occupied.obj", voxel_size=0.1,
        origin=(1.0, 2.0, 3.0),
    )

    assert np.allclose(
        _read_obj_vertices(path),
        [[1.05, 2.15, 3.15], [1.15, 2.05, 3.05]],
    )


def test_export_rejects_invalid_point_data(tmp_path):
    with pytest.raises(ValueError, match="Nx3"):
        export_point_cloud([[1.0, 2.0]], tmp_path / "invalid.ply")
    with pytest.raises(ValueError, match="finite"):
        export_point_cloud([[1.0, float("nan"), 3.0]], tmp_path / "invalid.obj")


def test_benchmark_writes_json_and_markdown_reports(tmp_path):
    report = generate_benchmark_report(tmp_path, agent_count=2, seed=42)
    json_report = json.loads((tmp_path / "benchmark_report.json").read_text())
    markdown = (tmp_path / "benchmark_report.md").read_text(encoding="utf-8")

    assert json_report == report
    assert report["environment"] == {
        "area_m2": 2000,
        "max_obstacle_height_m": 2.0,
    }
    assert report["simulation"]["total_distance_m"] > 0
    assert len(report["simulation"]["distance_by_agent_m"]) == 2
    assert report["simulation"]["elapsed_time_s"] > 0
    assert report["communication"]["total_bytes"] > 0
    assert report["accuracy_mm"]["max_xy"] < 30.0
    assert report["accuracy_mm"]["max_z"] < 30.0
    assert report["accuracy_mm"]["max_3d"] < 30.0
    assert "Total travel distance" in markdown
    assert "Communication" in markdown