# CubeNTube -- shape eraser plugin for UCSF ChimeraX
# Copyright 2026 Tamino Cairoli <tcairoli@ethz.ch>
#
# Dust eraser: uses ChimeraX's own surface-dust detection, then zeros
# the underlying voxels so the cleaned map can be saved as .mrc.
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np

DUST_HIGHLIGHT_COLOR = (255, 153, 204, 128)  # translucent pink
_CONTOUR_CACHE = {
    'volume': None,
    'level': None,
    'matrix_id': None,
    'verts': None,
    'norms': None,
    'tris': None,
}


def create_dust_highlight(session, volume, size_limit, existing=None):
    """
    Build/update dust preview directly from contouring the volume array.
    This bypasses volume.surfaces and avoids render-timing coupling.
    Returns (highlight_model, preview_data_dict).
    """
    preview = _hidden_dust_geometry(volume, size_limit)
    if preview is None:
        if existing is not None and not existing.deleted:
            existing.display = False
        return existing, None
    pv, pn, pt, hidden_xyz, level = preview
    if hidden_xyz is None:
        if existing is not None and not existing.deleted:
            existing.display = False
        return existing, {
            'volume': volume,
            'level': level,
            'size_limit': float(size_limit),
            'hidden_xyz': None,
        }

    from chimerax.core.models import Surface
    if existing is None or existing.deleted:
        m = Surface('dust preview', session)
        session.models.add([m])
    else:
        m = existing
    m.set_geometry(pv, pn, pt)
    m.position = volume.scene_position
    m.color = DUST_HIGHLIGHT_COLOR
    m.selected = False
    m.pickable = False
    m.display = True

    return m, {
        'volume': volume,
        'level': level,
        'size_limit': float(size_limit),
        'hidden_xyz': hidden_xyz,
    }


def compute_dust_voxel_mask(volume, size_limit, preview_data=None):
    """
    Compute boolean dust mask to erase.
    Uses preview_data hidden vertices when valid; otherwise recomputes.
    """
    level = volume.minimum_surface_level
    if level is None:
        return None

    hidden_xyz = None
    if preview_data is not None:
        if (preview_data.get('volume') is volume
                and preview_data.get('level') == level
                and abs(preview_data.get('size_limit', -1.0) - float(size_limit)) < 1e-12):
            hidden_xyz = preview_data.get('hidden_xyz')

    if hidden_xyz is None:
        preview = _hidden_dust_geometry(volume, size_limit)
        if preview is not None:
            _, _, _, hidden_xyz, _ = preview

    if hidden_xyz is None or len(hidden_xyz) == 0:
        return None

    mask = _vertices_to_voxel_mask(hidden_xyz, volume.data)
    padded_level = level * 0.97
    above = volume.data.full_matrix() >= padded_level
    mask &= above
    return mask if np.any(mask) else None


def _hidden_dust_geometry(volume, size_limit):
    """
    Return hidden dust geometry from an array contour as
    (preview_verts, preview_norms, preview_tris, hidden_xyz, level).
    hidden_xyz is None when there is no dust at current size threshold.
    """
    level = volume.minimum_surface_level
    if level is None:
        return None

    verts, norms, tris = _contour_for_volume_level(volume, level)
    if verts is None or tris is None or len(tris) == 0:
        return None

    from chimerax.surface.dust import Blob_Masker
    tmask = Blob_Masker(verts, tris).triangle_mask('size', float(size_limit))
    hidden_idx = np.where(~tmask)[0]
    if len(hidden_idx) == 0:
        return None, None, None, None, level

    hidden_tris = tris[hidden_idx]
    used = np.unique(hidden_tris.ravel())
    remap = np.full(len(verts), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    pv = verts[used].astype(np.float32)
    pn = norms[used].astype(np.float32) if norms is not None else None
    pt = remap[hidden_tris].astype(np.int32)
    return pv, pn, pt, pv, level


def _contour_for_volume_level(volume, level):
    c = _CONTOUR_CACHE
    matrix_id = getattr(volume, '_matrix_id', None)
    if (c['volume'] is volume and c['level'] == level
            and c['matrix_id'] == matrix_id):
        return c['verts'], c['norms'], c['tris']

    matrix = volume.matrix()
    from .custom_eraser import contour_from_array
    verts, norms, tris = contour_from_array(
        matrix, level, volume.matrix_indices_to_xyz_transform())
    c['volume'] = volume
    c['level'] = level
    c['matrix_id'] = matrix_id
    c['verts'] = verts
    c['norms'] = norms
    c['tris'] = tris
    return verts, norms, tris


def _vertices_to_voxel_mask(xyz, grid_data):
    """Convert vertex XYZ positions to a filled boolean voxel mask."""
    ijk_float = grid_data.xyz_to_ijk_transform.transform_points(xyz)
    ijk = np.rint(ijk_float).astype(np.int64)
    shape = tuple(reversed(grid_data.size))  # (nk, nj, ni)
    i = np.clip(ijk[:, 0], 0, shape[2] - 1)
    j = np.clip(ijk[:, 1], 0, shape[1] - 1)
    k = np.clip(ijk[:, 2], 0, shape[0] - 1)

    mask = np.zeros(shape, dtype=bool)
    mask[k, j, i] = True

    from scipy.ndimage import binary_dilation, binary_fill_holes
    mask = binary_dilation(mask)
    mask = binary_fill_holes(mask)
    return mask
