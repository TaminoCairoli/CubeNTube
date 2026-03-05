import numpy as np


def ijk_region_points(ijk_min, ni, nj, nk):
    """Return flattened ijk points for an axis-aligned voxel region."""
    i_vals = ijk_min[0] + np.arange(ni, dtype=np.float64)
    j_vals = ijk_min[1] + np.arange(nj, dtype=np.float64)
    k_vals = ijk_min[2] + np.arange(nk, dtype=np.float64)
    ii, jj, kk = np.meshgrid(i_vals, j_vals, k_vals, indexing='ij')
    return np.column_stack([ii.ravel(), jj.ravel(), kk.ravel()])


def ijk_points_to_volume_xyz(grid_data, ijk_points):
    """Map ijk points to the grid's local xyz coordinates."""
    origin = np.array(grid_data.origin, dtype=np.float64)
    step = np.array(grid_data.step, dtype=np.float64)
    if hasattr(grid_data, 'rotation') and grid_data.rotation is not None:
        grid_rotation = np.array(grid_data.rotation, dtype=np.float64)
        return origin + np.dot(ijk_points * step, grid_rotation.T)
    return origin + ijk_points * step


def ijk_region_scene_xyz(grid_data, volume_scene_position, ijk_min, ni, nj, nk):
    """Map an ijk voxel region to scene-space xyz points."""
    ijk_points = ijk_region_points(ijk_min, ni, nj, nk)
    volume_xyz = ijk_points_to_volume_xyz(grid_data, ijk_points)
    return volume_scene_position.transform_points(volume_xyz)
