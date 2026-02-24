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

CYLINDER_ERASER_COLOR = (255, 153, 204, 128)  # transparent pink (matches sphere eraser)


# -------------------------------------------------------------------------
#
def volume_cylinder_erase(session, volumes, center, radius_top, radius_bottom,
                          length, coordinate_system=None, outside=False, value=0):
    '''Erase a volume inside or outside a cylinder (frustum).'''
    ev = []
    cscene = center.scene_coordinates(coordinate_system)

    for volume in volumes:
        v = volume.writable_copy()
        ev.append(v)

        from chimerax.geometry import translation
        class TempCylinder:
            pass
        cyl = TempCylinder()
        cyl.scene_position = translation(cscene)
        cyl.radius_top = radius_top
        cyl.radius_bottom = radius_bottom
        cyl.length = length

        _erase_with_cylinder_model(v.data, cyl, volume.scene_position,
                                   value=value, outside=outside)

    return ev[0] if len(ev) == 1 else ev


# -------------------------------------------------------------------------
#
def _erase_with_cylinder_model(grid_data, cylinder_model, volume_scene_position,
                               value=0, outside=False):
    '''
    Core erase logic for cylinder. Rotation-safe and vectorized.
    Transforms: volume_local -> scene -> cylinder_local, then tests cylindrical bounds.
    For "erase outside" the full grid is used so all voxels outside the shape are zeroed.
    '''
    from chimerax.map_data import GridSubregion
    from numpy import putmask

    scene_to_cyl = cylinder_model.scene_position.inverse()
    rt = cylinder_model.radius_top
    rb = cylinder_model.radius_bottom
    h = cylinder_model.length / 2.0

    if outside:
        dmatrix = grid_data.full_matrix()
        ni, nj, nk = grid_data.size
        ijk_min = (0, 0, 0)
    else:
        ijk_min, ijk_max = _cylinder_grid_bounds(grid_data, cylinder_model,
                                                 volume_scene_position)
        subgrid = GridSubregion(grid_data, ijk_min, ijk_max)
        dmatrix = subgrid.full_matrix()
        ni = ijk_max[0] - ijk_min[0] + 1
        nj = ijk_max[1] - ijk_min[1] + 1
        nk = ijk_max[2] - ijk_min[2] + 1

    i_vals = ijk_min[0] + np.arange(ni, dtype=np.float64)
    j_vals = ijk_min[1] + np.arange(nj, dtype=np.float64)
    k_vals = ijk_min[2] + np.arange(nk, dtype=np.float64)
    ii, jj, kk = np.meshgrid(i_vals, j_vals, k_vals, indexing='ij')
    ijk_points = np.column_stack([ii.ravel(), jj.ravel(), kk.ravel()])

    origin = np.array(grid_data.origin, dtype=np.float64)
    step = np.array(grid_data.step, dtype=np.float64)
    if hasattr(grid_data, 'rotation') and grid_data.rotation is not None:
        grid_rotation = np.array(grid_data.rotation, dtype=np.float64)
        volume_xyz = origin + np.dot(ijk_points * step, grid_rotation.T)
    else:
        volume_xyz = origin + ijk_points * step

    scene_xyz = volume_scene_position.transform_points(volume_xyz)
    cyl_local = scene_to_cyl.transform_points(scene_xyz)

    z = cyl_local[:, 2]
    inside_z = np.abs(z) <= h

    if h > 0:
        t = np.clip((z + h) / (2.0 * h), 0, 1)
        r_at_z = rb + (rt - rb) * t
    else:
        r_at_z = np.full_like(z, max(rt, rb))

    xy_dist_sq = cyl_local[:, 0]**2 + cyl_local[:, 1]**2
    inside = inside_z & (xy_dist_sq <= r_at_z**2)

    mask = inside.reshape(ni, nj, nk)
    mask = np.transpose(mask, (2, 1, 0))
    if outside:
        mask = ~mask
    putmask(dmatrix, mask, value)

    grid_data.values_changed()


# -------------------------------------------------------------------------
#
def _cylinder_grid_bounds(grid_data, cylinder_model, volume_scene_position):
    '''
    Compute axis-aligned bounding box (in volume ijk) of the rotated cylinder.
    Cylinder bounding box corners transformed: cylinder_local -> scene -> volume_local.
    '''
    from math import floor, ceil

    rt = cylinder_model.radius_top
    rb = cylinder_model.radius_bottom
    max_r = max(rt, rb)
    h = cylinder_model.length / 2.0

    corners_local = np.array([
        [-max_r, -max_r, -h], [max_r, -max_r, -h],
        [-max_r, max_r, -h],  [max_r, max_r, -h],
        [-max_r, -max_r, h],  [max_r, -max_r, h],
        [-max_r, max_r, h],   [max_r, max_r, h],
    ], dtype=float)

    cyl_to_scene = cylinder_model.scene_position
    scene_to_volume = volume_scene_position.inverse()
    corners_scene = np.array([cyl_to_scene * c for c in corners_local])
    corners_volume = np.array([scene_to_volume * c for c in corners_scene])

    xyz_min = corners_volume.min(axis=0)
    xyz_max = corners_volume.max(axis=0)

    ijk_min_f = grid_data.xyz_to_ijk(xyz_min)
    ijk_max_f = grid_data.xyz_to_ijk(xyz_max)

    ijk_min = [max(int(floor(i)), 0) for i in ijk_min_f]
    ijk_max = [min(int(ceil(i)), s - 1)
               for i, s in zip(ijk_max_f, grid_data.size)]

    return ijk_min, ijk_max


# -------------------------------------------------------------------------
#
def register_volume_cylinder_erase_command(logger):
    from chimerax.core.commands import (CmdDesc, register, FloatArg,
                                        CenterArg, CoordSysArg, BoolArg)
    from chimerax.map import MapsArg
    desc = CmdDesc(
        required=[('volumes', MapsArg)],
        keyword=[('center', CenterArg),
                 ('radius_top', FloatArg),
                 ('radius_bottom', FloatArg),
                 ('length', FloatArg),
                 ('coordinate_system', CoordSysArg),
                 ('outside', BoolArg),
                 ('value', FloatArg)],
        required_arguments=['center', 'radius_top', 'radius_bottom', 'length'],
        synopsis='Set map values to zero inside a cylinder'
    )
    register('volume cylinder erase', desc, volume_cylinder_erase, logger=logger)


from chimerax.core.models import Surface


class CylinderModel(Surface):
    '''Cylinder (frustum) surface for the eraser widget.'''
    SESSION_SAVE = False

    def __init__(self, name, session, color, center, radius_top, radius_bottom,
                 length):
        Surface.__init__(self, name, session)
        self._radius_top = radius_top
        self._radius_bottom = radius_bottom
        self._length = length
        self._update_geometry()
        self.color = color
        from chimerax.geometry import translation
        self.position = translation(center)
        session.models.add([self])

    def _update_geometry(self):
        import math
        rt = self._radius_top
        rb = self._radius_bottom
        h = self._length / 2.0
        n = 36

        cos = [math.cos(2 * math.pi * i / n) for i in range(n)]
        sin = [math.sin(2 * math.pi * i / n) for i in range(n)]

        verts = []
        norms = []
        tris = []

        side_base = 0
        dr = rb - rt
        height = 2.0 * h
        slant_len = math.sqrt(height ** 2 + dr ** 2) or 1.0
        nr = height / slant_len
        nz = dr / slant_len
        for i in range(n):
            j = (i + 1) % n
            cx0, sy0 = cos[i], sin[i]
            cx1, sy1 = cos[j], sin[j]
            n0 = [nr * cx0, nr * sy0, nz]
            n1 = [nr * cx1, nr * sy1, nz]
            idx = side_base + i * 4
            verts.extend([
                [rb * cx0, rb * sy0, -h],
                [rb * cx1, rb * sy1, -h],
                [rt * cx0, rt * sy0,  h],
                [rt * cx1, rt * sy1,  h],
            ])
            norms.extend([n0, n1, n0, n1])
            tris.extend([
                [idx, idx + 1, idx + 2],
                [idx + 2, idx + 1, idx + 3],
            ])

        bot_base = len(verts)
        bot_center_idx = bot_base
        verts.append([0, 0, -h])
        norms.append([0, 0, -1])
        for i in range(n):
            verts.append([rb * cos[i], rb * sin[i], -h])
            norms.append([0, 0, -1])
        for i in range(n):
            j = (i + 1) % n
            tris.append([bot_center_idx, bot_base + 1 + j, bot_base + 1 + i])

        top_base = len(verts)
        top_center_idx = top_base
        verts.append([0, 0, h])
        norms.append([0, 0, 1])
        for i in range(n):
            verts.append([rt * cos[i], rt * sin[i], h])
            norms.append([0, 0, 1])
        for i in range(n):
            j = (i + 1) % n
            tris.append([top_center_idx, top_base + 1 + i, top_base + 1 + j])

        self.set_geometry(np.array(verts, dtype=np.float32),
                          np.array(norms, dtype=np.float32),
                          np.array(tris, dtype=np.int32))

    @property
    def radius_top(self):
        return self._radius_top

    @radius_top.setter
    def radius_top(self, r):
        if r != self._radius_top:
            self._radius_top = r
            self._update_geometry()

    @property
    def radius_bottom(self):
        return self._radius_bottom

    @radius_bottom.setter
    def radius_bottom(self, r):
        if r != self._radius_bottom:
            self._radius_bottom = r
            self._update_geometry()

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, L):
        if L != self._length:
            self._length = L
            self._update_geometry()

