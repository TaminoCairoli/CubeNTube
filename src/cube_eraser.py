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
from .grid_math import ijk_region_scene_xyz

CUBE_ERASER_COLOR = (255, 153, 204, 128)  # transparent pink


# -------------------------------------------------------------------------
#
def volume_cube_erase(session, volumes, center, size_x, size_y, size_z,
                      coordinate_system=None,
                      outside=False, value=0):
    '''
    Erase a volume inside or outside a cube.
    Rotation is automatically handled via cube model scene_position.
    '''

    ev = []
    cscene = center.scene_coordinates(coordinate_system)

    for volume in volumes:

        v = volume.writable_copy()
        ev.append(v)

        from chimerax.geometry import translation
        cube_position = translation(cscene)

        class TempCube:
            pass

        cube = TempCube()
        cube.scene_position = cube_position
        cube.size_x = size_x
        cube.size_y = size_y
        cube.size_z = size_z

        _erase_with_cube_model(v.data, cube, volume.scene_position, value=value, outside=outside)

    return ev[0] if len(ev) == 1 else ev

# -----------------------------------------------------------------------------
#
def _erase_with_cube_model(grid_data, cube_model, volume_scene_position, value=0, outside=False):
    '''
    Core erase logic. Rotation-safe and vectorized.
    Transforms: volume_local -> scene -> cube_local, then tests |local| <= half_size.
    For "erase outside" the full grid is used so all voxels outside the shape are zeroed.
    Returns True iff at least one voxel value is changed.
    '''
    from chimerax.map_data import GridSubregion
    from numpy import putmask

    scene_to_cube = cube_model.scene_position.inverse()
    hx = cube_model.size_x / 2.0
    hy = cube_model.size_y / 2.0
    hz = cube_model.size_z / 2.0

    if outside:
        dmatrix = grid_data.full_matrix()
        ni, nj, nk = grid_data.size
        ijk_min = (0, 0, 0)
    else:
        ijk_min, ijk_max = _cube_grid_bounds(grid_data, cube_model, volume_scene_position)
        subgrid = GridSubregion(grid_data, ijk_min, ijk_max)
        dmatrix = subgrid.full_matrix()
        ni = ijk_max[0] - ijk_min[0] + 1
        nj = ijk_max[1] - ijk_min[1] + 1
        nk = ijk_max[2] - ijk_min[2] + 1

    scene_xyz = ijk_region_scene_xyz(
        grid_data, volume_scene_position, ijk_min, ni, nj, nk)
    cube_local = scene_to_cube.transform_points(scene_xyz)

    inside = (
        (np.abs(cube_local[:, 0]) <= hx)
        & (np.abs(cube_local[:, 1]) <= hy)
        & (np.abs(cube_local[:, 2]) <= hz)
    )
    mask = inside.reshape(ni, nj, nk)
    mask = np.transpose(mask, (2, 1, 0))
    if outside:
        mask = ~mask
    changed = bool(np.any(mask & (dmatrix != value)))
    if not changed:
        return False
    putmask(dmatrix, mask, value)

    grid_data.values_changed()
    return True


# -----------------------------------------------------------------------------
#
def _cube_grid_bounds(grid_data, cube_model, volume_scene_position):
    '''
    Compute axis-aligned bounding box (in volume ijk) of the rotated cube.
    Cube corners are transformed: cube_local -> scene -> volume_local,
    then xyz_to_ijk uses volume-local coordinates.
    '''
    from math import floor, ceil

    hx = cube_model.size_x / 2.0
    hy = cube_model.size_y / 2.0
    hz = cube_model.size_z / 2.0

    corners_local = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz],
        [-hx, hy, -hz], [hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz],
        [-hx, hy, hz], [hx, hy, hz],
    ], dtype=float)

    cube_to_scene = cube_model.scene_position
    scene_to_volume = volume_scene_position.inverse()
    corners_scene = cube_to_scene.transform_points(corners_local)
    corners_volume = scene_to_volume.transform_points(corners_scene)

    xyz_min = corners_volume.min(axis=0)
    xyz_max = corners_volume.max(axis=0)

    ijk_min = grid_data.xyz_to_ijk(xyz_min)
    ijk_max = grid_data.xyz_to_ijk(xyz_max)

    ijk_min = [max(int(floor(i)), 0) for i in ijk_min]
    ijk_max = [
        min(int(ceil(i)), s - 1)
        for i, s in zip(ijk_max, grid_data.size)
    ]

    return ijk_min, ijk_max


# -------------------------------------------------------------------------
#
def register_volume_cube_erase_command(logger):
    from chimerax.core.commands import CmdDesc, register, FloatArg, CenterArg, CoordSysArg, BoolArg
    from chimerax.map import MapsArg
    desc = CmdDesc(
        required=[('volumes', MapsArg), ],
        keyword=[('center', CenterArg),
                 ('size_x', FloatArg),
                 ('size_y', FloatArg),
                 ('size_z', FloatArg),
                 ('coordinate_system', CoordSysArg),
                 ('outside', BoolArg),
                 ('value', FloatArg),
                 ],
        required_arguments=['center', 'size_x', 'size_y', 'size_z'],
        synopsis='Set map values to zero inside a cube'
    )
    register('volume cube erase', desc, volume_cube_erase, logger=logger)


from chimerax.core.models import Surface

_CUBE_UNIT_FACE_VERTICES = np.array([
    # Back face
    [-1, -1, -1], [1, 1, -1], [1, -1, -1],
    [-1, -1, -1], [-1, 1, -1], [1, 1, -1],
    # Front face
    [-1, -1, 1], [1, -1, 1], [1, 1, 1],
    [-1, -1, 1], [1, 1, 1], [-1, 1, 1],
    # Bottom face
    [-1, -1, -1], [1, -1, -1], [1, -1, 1],
    [-1, -1, -1], [1, -1, 1], [-1, -1, 1],
    # Top face
    [1, 1, -1], [-1, 1, -1], [-1, 1, 1],
    [1, 1, -1], [-1, 1, 1], [1, 1, 1],
    # Left face
    [-1, -1, -1], [-1, -1, 1], [-1, 1, 1],
    [-1, -1, -1], [-1, 1, 1], [-1, 1, -1],
    # Right face
    [1, -1, -1], [1, 1, -1], [1, 1, 1],
    [1, -1, -1], [1, 1, 1], [1, -1, 1],
], dtype=np.float32)

_CUBE_FACE_NORMALS = np.array([
    [0, 0, -1], [0, 0, -1], [0, 0, -1],
    [0, 0, -1], [0, 0, -1], [0, 0, -1],
    [0, 0, 1], [0, 0, 1], [0, 0, 1],
    [0, 0, 1], [0, 0, 1], [0, 0, 1],
    [0, -1, 0], [0, -1, 0], [0, -1, 0],
    [0, -1, 0], [0, -1, 0], [0, -1, 0],
    [0, 1, 0], [0, 1, 0], [0, 1, 0],
    [0, 1, 0], [0, 1, 0], [0, 1, 0],
    [-1, 0, 0], [-1, 0, 0], [-1, 0, 0],
    [-1, 0, 0], [-1, 0, 0], [-1, 0, 0],
    [1, 0, 0], [1, 0, 0], [1, 0, 0],
    [1, 0, 0], [1, 0, 0], [1, 0, 0],
], dtype=np.float32)

_CUBE_FACE_TRIANGLES = np.array([
    [0, 1, 2], [3, 4, 5],
    [6, 7, 8], [9, 10, 11],
    [12, 13, 14], [15, 16, 17],
    [18, 19, 20], [21, 22, 23],
    [24, 25, 26], [27, 28, 29],
    [30, 31, 32], [33, 34, 35],
], dtype=np.int32)


class CubeModel(Surface):
    SESSION_SAVE = False

    def __init__(self, name, session, color, center, size_x, size_y, size_z):
        Surface.__init__(self, name, session)

        self._size_x = size_x
        self._size_y = size_y
        self._size_z = size_z

        self._update_geometry()
        self.color = color
        from chimerax.geometry import translation
        self.position = translation(center)
        session.models.add([self])

    def _update_geometry(self):
        '''Create box geometry with current sizes.'''

        scale = np.array([self._size_x * 0.5,
                          self._size_y * 0.5,
                          self._size_z * 0.5], dtype=np.float32)
        face_vertices = _CUBE_UNIT_FACE_VERTICES * scale
        self.set_geometry(face_vertices, _CUBE_FACE_NORMALS, _CUBE_FACE_TRIANGLES)

    def _get_size_x(self):
        return self._size_x

    def _set_size_x(self, s):
        if s != self._size_x:
            self._size_x = s
            self._update_geometry()

    size_x = property(_get_size_x, _set_size_x)

    def _get_size_y(self):
        return self._size_y

    def _set_size_y(self, s):
        if s != self._size_y:
            self._size_y = s
            self._update_geometry()

    size_y = property(_get_size_y, _set_size_y)

    def _get_size_z(self):
        return self._size_z

    def _set_size_z(self, s):
        if s != self._size_z:
            self._size_z = s
            self._update_geometry()

    size_z = property(_get_size_z, _set_size_z)

