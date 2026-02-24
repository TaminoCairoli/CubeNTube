# CubeNTube -- shape eraser plugin for UCSF ChimeraX
# Copyright 2026 Tamino Cairoli <tcairoli@ethz.ch>
#
# Based on the built-in sphere Map Eraser from ChimeraX
# (chimerax.map_eraser), developed by the UCSF Resource for Biocomputing,
# Visualization, and Informatics (https://www.rbvi.ucsf.edu/chimerax/).
#
# Written with AI assistance (Claude / Cursor), February 2026.
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np

CUSTOM_ERASER_COLOR = (255, 153, 204, 128)  # transparent pink (matches sphere eraser)


# -------------------------------------------------------------------------
#
def _extract_isosurface(volume):
    '''
    Extract isosurface mesh and metadata from a Volume.
    Returns (vertices, normals, triangles, threshold) or (None,)*4.
    Vertices are in the volume's local xyz coordinate system.
    '''
    threshold = volume.minimum_surface_level
    if threshold is None:
        return None, None, None, None

    surfs = volume.surfaces
    if not surfs:
        return None, None, None, None

    surf = surfs[0]
    verts = surf.vertices
    tris = surf.triangles
    if verts is None or tris is None or len(verts) == 0:
        return None, None, None, None

    from chimerax.surface import calculate_vertex_normals
    norms = calculate_vertex_normals(verts, tris)

    return verts.copy(), norms.copy(), tris.copy(), threshold


# -------------------------------------------------------------------------
#
def _erase_with_custom_shape(target_grid_data, shape_model, volume_scene_position,
                             mask_array, mask_xyz_to_ijk, centroid, scale,
                             threshold, value=0, outside=False):
    '''
    Erase target volume using a custom volume shape as the mask.
    For each target voxel: scene -> eraser-local -> unscale -> mask-volume xyz
    -> sample mask data -> compare threshold.
    '''
    from chimerax.map_data import GridSubregion, interpolate_volume_data
    from numpy import putmask

    if outside:
        dmatrix = target_grid_data.full_matrix()
        ni, nj, nk = target_grid_data.size
        ijk_min = (0, 0, 0)
    else:
        ijk_min, ijk_max = _custom_grid_bounds(target_grid_data, shape_model,
                                               volume_scene_position)
        subgrid = GridSubregion(target_grid_data, ijk_min, ijk_max)
        dmatrix = subgrid.full_matrix()
        ni = ijk_max[0] - ijk_min[0] + 1
        nj = ijk_max[1] - ijk_min[1] + 1
        nk = ijk_max[2] - ijk_min[2] + 1

    i_vals = ijk_min[0] + np.arange(ni, dtype=np.float64)
    j_vals = ijk_min[1] + np.arange(nj, dtype=np.float64)
    k_vals = ijk_min[2] + np.arange(nk, dtype=np.float64)
    ii, jj, kk = np.meshgrid(i_vals, j_vals, k_vals, indexing='ij')
    ijk_points = np.column_stack([ii.ravel(), jj.ravel(), kk.ravel()])

    origin = np.array(target_grid_data.origin, dtype=np.float64)
    step = np.array(target_grid_data.step, dtype=np.float64)
    if hasattr(target_grid_data, 'rotation') and target_grid_data.rotation is not None:
        grid_rotation = np.array(target_grid_data.rotation, dtype=np.float64)
        volume_xyz = origin + np.dot(ijk_points * step, grid_rotation.T)
    else:
        volume_xyz = origin + ijk_points * step

    scene_xyz = volume_scene_position.transform_points(volume_xyz)

    scene_to_eraser = shape_model.scene_position.inverse()
    eraser_local = scene_to_eraser.transform_points(scene_xyz)

    mask_xyz = eraser_local / scale + centroid

    mask_xyz_f32 = mask_xyz.astype(np.float32)
    sampled, oob_indices = interpolate_volume_data(mask_xyz_f32, mask_xyz_to_ijk,
                                                   mask_array)

    inside = sampled >= threshold
    if len(oob_indices) > 0:
        inside[oob_indices] = False

    mask = inside.reshape(ni, nj, nk)
    mask = np.transpose(mask, (2, 1, 0))
    if outside:
        mask = ~mask
    putmask(dmatrix, mask, value)

    target_grid_data.values_changed()


# -------------------------------------------------------------------------
#
def _custom_grid_bounds(target_grid_data, shape_model, volume_scene_position):
    '''
    Compute axis-aligned bounding box (in target volume ijk) of the eraser shape.
    Uses the scaled mesh extents transformed through scene coords.
    '''
    from math import floor, ceil

    base_verts = shape_model.base_vertices
    s = shape_model.scale
    scaled_min = base_verts.min(axis=0) * s
    scaled_max = base_verts.max(axis=0) * s

    corners_local = np.array([
        [scaled_min[0], scaled_min[1], scaled_min[2]],
        [scaled_max[0], scaled_min[1], scaled_min[2]],
        [scaled_min[0], scaled_max[1], scaled_min[2]],
        [scaled_max[0], scaled_max[1], scaled_min[2]],
        [scaled_min[0], scaled_min[1], scaled_max[2]],
        [scaled_max[0], scaled_min[1], scaled_max[2]],
        [scaled_min[0], scaled_max[1], scaled_max[2]],
        [scaled_max[0], scaled_max[1], scaled_max[2]],
    ], dtype=float)

    eraser_to_scene = shape_model.scene_position
    scene_to_volume = volume_scene_position.inverse()
    corners_scene = np.array([eraser_to_scene * c for c in corners_local])
    corners_volume = np.array([scene_to_volume * c for c in corners_scene])

    xyz_min = corners_volume.min(axis=0)
    xyz_max = corners_volume.max(axis=0)

    ijk_min_f = target_grid_data.xyz_to_ijk(xyz_min)
    ijk_max_f = target_grid_data.xyz_to_ijk(xyz_max)

    ijk_min = [max(int(floor(i)), 0) for i in ijk_min_f]
    ijk_max = [min(int(ceil(i)), s - 1)
               for i, s in zip(ijk_max_f, target_grid_data.size)]

    return ijk_min, ijk_max


from chimerax.core.models import Surface


class CustomShapeModel(Surface):
    '''Surface model displaying a scaled copy of a volume isosurface.'''
    SESSION_SAVE = False

    def __init__(self, name, session, color, position, centered_vertices,
                 normals, triangles, centroid):
        '''
        centered_vertices: mesh vertices already centered at origin (in mask-vol local).
        position: a Place for the model's scene position.
        centroid: the centroid in mask-volume local xyz (used for erase mapping).
        '''
        Surface.__init__(self, name, session)

        self._base_vertices = centered_vertices.astype(np.float32)
        self._base_normals = normals.astype(np.float32)
        self._base_triangles = triangles.astype(np.int32)
        self._centroid = np.array(centroid, dtype=np.float64)
        self._scale = 1.0

        self._update_geometry()
        self.color = color
        self.position = position
        session.models.add([self])

    def _update_geometry(self):
        verts = self._base_vertices * self._scale
        self.set_geometry(verts, self._base_normals, self._base_triangles)

    @property
    def base_vertices(self):
        return self._base_vertices

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, s):
        if s != self._scale:
            self._scale = s
            self._update_geometry()

    @property
    def centroid(self):
        return self._centroid

