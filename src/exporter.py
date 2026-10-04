from pathlib import Path

import numpy as np


def _validate_points(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an Nx3 array")
    if not np.isfinite(points).all():
        raise ValueError("points must contain only finite coordinates")
    return points


def voxel_grid_to_points(voxels, voxel_size, origin=(0.0, 0.0, 0.0)):
    grid = np.asarray(voxels)
    if grid.ndim != 3:
        raise ValueError("voxels must be a 3D occupancy grid")
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be positive and finite")
    origin = np.asarray(origin, dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("origin must contain three finite coordinates")

    occupied = np.argwhere(np.isfinite(grid) & (grid != 0))
    return origin + (occupied.astype(float) + 0.5) * voxel_size


def export_point_cloud(points, filepath):
    points = _validate_points(points)
    path = Path(filepath)
    extension = path.suffix.lower()
    if extension not in {".ply", ".obj"}:
        raise ValueError("filepath extension must be .ply or .obj")
    path.parent.mkdir(parents=True, exist_ok=True)

    if extension == ".ply":
        with path.open("w", encoding="ascii", newline="\n") as output:
            output.write("ply\nformat ascii 1.0\n")
            output.write(f"element vertex {len(points)}\n")
            output.write("property float x\nproperty float y\nproperty float z\n")
            output.write("end_header\n")
            for point in points:
                output.write("{:.9g} {:.9g} {:.9g}\n".format(*point))
    else:
        with path.open("w", encoding="ascii", newline="\n") as output:
            output.write("# point cloud vertices\n")
            for point in points:
                output.write("v {:.9g} {:.9g} {:.9g}\n".format(*point))
    return path


def export_voxels(voxels, filepath, voxel_size, origin=(0.0, 0.0, 0.0)):
    points = voxel_grid_to_points(voxels, voxel_size, origin)
    return export_point_cloud(points, filepath)