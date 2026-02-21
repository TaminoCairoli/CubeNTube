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

from chimerax.core.undo import UndoAction

CUSTOM_ERASER_COLOR = (255, 153, 204, 128)  # transparent pink (matches sphere eraser)


class _VolumeEraseUndo(UndoAction):
    """Single-step undo/redo for volume erase operations."""

    def __init__(self, name, grid_data, saved_matrix):
        super().__init__(name, can_redo=True)
        self._grid_data = grid_data
        self._saved_before = saved_matrix
        self._saved_after = None

    def undo(self):
        m = self._grid_data.full_matrix()
        self._saved_after = m.copy()
        m[:] = self._saved_before
        self._grid_data.values_changed()

    def redo(self):
        if self._saved_after is None:
            return
        m = self._grid_data.full_matrix()
        m[:] = self._saved_after
        self._grid_data.values_changed()


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

    # Unscale and shift to mask-volume local coordinates
    mask_xyz = eraser_local / scale + centroid

    # Sample the mask volume data via trilinear interpolation
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

    base_verts = shape_model._base_vertices
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


# -------------------------------------------------------------------------
#
from chimerax.mouse_modes import MouseMode


class MapCustomEraser(MouseMode):
    name = 'custom eraser'
    icon_file = 'cubentube.png'

    def __init__(self, session):
        MouseMode.__init__(self, session)

    @property
    def settings(self):
        from .gui_panel import map_shape_eraser_panel
        sp = map_shape_eraser_panel(self.session, create=False)
        if sp is not None:
            return sp
        return map_custom_eraser_panel(self.session)

    def enable(self):
        from .gui_panel import map_shape_eraser_panel
        sp = map_shape_eraser_panel(self.session)
        sp._shape_combo.setCurrentIndex(2)
        sp.show()

    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)

    def mouse_drag(self, event):
        dx, dy = self.mouse_motion(event)
        settings = self.settings
        sm = settings.custom_shape_model
        if sm is None:
            return
        c = sm.scene_position.origin()
        v = self.session.main_view
        s = v.pixel_size(c)
        if event.shift_down():
            shift = (0, 0, s * dy)
        else:
            shift = (s * dx, -s * dy, 0)
        dxyz = v.camera.position.transform_vector(shift)
        settings.move_shape(dxyz)

    def mouse_up(self, event):
        MouseMode.mouse_up(self, event)

    def vr_motion(self, event):
        settings = self.settings
        sm = settings.custom_shape_model
        if sm is None:
            return
        c = sm.scene_position.origin()
        delta_xyz = event.motion * c - c
        settings.move_shape(delta_xyz)


# -------------------------------------------------------------------------
#
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


# -------------------------------------------------------------------------
# Panel for erasing parts of map with a custom volume shape.
#
from chimerax.core.tools import ToolInstance


class MapCustomEraserSettings(ToolInstance):
    help = "help:user/tools/mapcustomeraser.html"

    def __init__(self, session, tool_name):

        self._default_color = CUSTOM_ERASER_COLOR
        self._max_slider_value = 1000
        self._max_slider_scale = 5.0  # max scale factor
        self._min_slider_scale = 0.1
        self._block_text_update = False

        self._custom_shape_model = None
        self._mask_array = None
        self._mask_xyz_to_ijk = None
        self._threshold = None
        self._last_undo_action = None

        ToolInstance.__init__(self, session, tool_name)
        self.display_name = 'Custom Eraser'

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        self.tool_window = tw
        parent = tw.ui_area

        from Qt.QtWidgets import (QVBoxLayout, QHBoxLayout, QFrame,
                                  QCheckBox, QLabel, QPushButton,
                                  QLineEdit, QSlider)
        from Qt.QtCore import Qt

        layout = QVBoxLayout(parent)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        parent.setLayout(layout)

        # --- Row 1: Show checkbox, color ---
        sf = QFrame(parent)
        layout.addWidget(sf)
        slayout = QHBoxLayout(sf)
        slayout.setContentsMargins(0, 0, 0, 0)
        slayout.setSpacing(10)

        self._show_eraser = se = QCheckBox('Show eraser shape', sf)
        se.setCheckState(Qt.Checked)
        se.stateChanged.connect(self._show_eraser_cb)
        slayout.addWidget(se)

        from chimerax.ui.widgets import ColorButton
        self._shape_color = sc = ColorButton(sf, max_size=(16, 16),
                                             has_alpha_channel=True)
        sc.color = CUSTOM_ERASER_COLOR
        sc.color_changed.connect(self._change_color_cb)
        slayout.addWidget(sc)
        slayout.addStretch(1)

        # --- Row 2: Volume selector + Set button ---
        vf = QFrame(parent)
        layout.addWidget(vf)
        vlayout = QHBoxLayout(vf)
        vlayout.setContentsMargins(0, 0, 0, 0)
        vlayout.setSpacing(10)

        vlayout.addWidget(QLabel('Mask volume:', vf))

        from chimerax.ui.widgets import ModelMenuButton
        from chimerax.map import Volume
        self._volume_menu = vm = ModelMenuButton(
            session, class_filter=Volume,
            no_value_button_text='Choose volume...')
        vlayout.addWidget(vm)

        self._set_btn = sb = QPushButton('Set as eraser', vf)
        sb.clicked.connect(self._set_eraser_cb)
        vlayout.addWidget(sb)
        vlayout.addStretch(1)

        # --- Row 3: Scale slider ---
        scf = QFrame(parent)
        layout.addWidget(scf)
        sclayout = QHBoxLayout(scf)
        sclayout.setContentsMargins(0, 0, 0, 0)
        sclayout.setSpacing(4)

        sclayout.addWidget(QLabel('Scale', scf))
        self._scale_entry = sce = QLineEdit('1.0', scf)
        sce.setMaximumWidth(40)
        sce.returnPressed.connect(self._scale_changed_cb)
        sclayout.addWidget(sce)
        self._scale_slider = scs = QSlider(Qt.Horizontal, scf)
        scs.setRange(0, self._max_slider_value)
        scs.valueChanged.connect(self._scale_slider_cb)
        sclayout.addWidget(scs)

        # Set slider to scale=1.0
        self._set_scale_slider(1.0)

        # --- Row 4: Erase buttons ---
        ef = QFrame(parent)
        layout.addWidget(ef)
        elayout = QHBoxLayout(ef)
        elayout.setContentsMargins(0, 0, 0, 0)
        elayout.setSpacing(30)

        eb = QPushButton('Erase inside shape', ef)
        eb.clicked.connect(self._erase_inside)
        elayout.addWidget(eb)

        eo = QPushButton('Erase outside shape', ef)
        eo.clicked.connect(self._erase_outside)
        elayout.addWidget(eo)

        rb = QPushButton('Reduce map bounds', ef)
        rb.clicked.connect(self._crop_map)
        elayout.addWidget(rb)

        elayout.addStretch(1)
        layout.addStretch(1)

        tw.manage(placement='side')

    def delete(self):
        sm = self._custom_shape_model
        if sm and not sm.deleted:
            self.session.models.close([sm])
        self._custom_shape_model = None
        self._mask_array = None
        self._mask_xyz_to_ijk = None
        ToolInstance.delete(self)

    @classmethod
    def get_singleton(cls, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, MapCustomEraserSettings,
                                   'Custom Eraser', create=create)

    def show(self):
        self.tool_window.shown = True

    def hide(self):
        self.tool_window.shown = False

    @property
    def custom_shape_model(self):
        return self._custom_shape_model

    # --- Set eraser from selected volume ---
    def _set_eraser_cb(self):
        vol = self._volume_menu.value
        if vol is None:
            self.session.logger.warning('No volume selected.')
            return

        verts, norms, tris, threshold = _extract_isosurface(vol)
        if verts is None:
            self.session.logger.warning(
                'Selected volume has no displayed isosurface. '
                'Display an isosurface first (e.g. volume #%s level <value>).'
                % vol.id_string)
            return

        # Store mask data for erase operations
        self._mask_array = vol.data.full_matrix().copy()
        self._mask_xyz_to_ijk = vol.data.xyz_to_ijk_transform
        self._threshold = threshold

        # Remove old shape if present
        old = self._custom_shape_model
        if old and not old.deleted:
            self.session.models.close([old])

        # Vertices from the isosurface are in mask-volume local xyz.
        # Center them at origin; store centroid for erase coordinate mapping.
        centroid = verts.mean(axis=0)
        centered_verts = verts - centroid

        # Position the eraser so its mesh aligns with the original volume:
        #   eraser_pos * centered_vert = vol.scene_pos * (centered_vert + centroid)
        # So: eraser_pos = vol.scene_pos * translation(centroid)
        from chimerax.geometry import translation
        eraser_pos = vol.scene_position * translation(centroid)

        sm = CustomShapeModel('custom eraser', self.session,
                              self._default_color, eraser_pos,
                              centered_verts, norms, tris, centroid)
        self._custom_shape_model = sm

        # Reset scale slider
        self._block_text_update = True
        self._scale_entry.setText('1.0')
        self._set_scale_slider(1.0)
        self._block_text_update = False

        from chimerax.core.commands import run
        run(self.session, 'hide #%s models' % vol.id_string)

        self.session.logger.info(
            'Custom eraser set from #%s at level %.4g (%d triangles)'
            % (vol.id_string, threshold, len(tris)))

    # --- Scale slider ---
    def _scale_value(self):
        try:
            return max(self._min_slider_scale,
                       min(float(self._scale_entry.text()),
                           self._max_slider_scale))
        except ValueError:
            return 1.0

    def _set_scale_slider(self, s):
        frac = ((s - self._min_slider_scale)
                / (self._max_slider_scale - self._min_slider_scale))
        self._scale_slider.setValue(
            int(frac * self._max_slider_value))

    def _scale_from_slider(self, sval):
        frac = sval / self._max_slider_value
        return (self._min_slider_scale
                + frac * (self._max_slider_scale - self._min_slider_scale))

    def _scale_changed_cb(self):
        if self._block_text_update:
            return
        s = self._scale_value()
        self._block_text_update = True
        self._set_scale_slider(s)
        self._block_text_update = False
        sm = self._custom_shape_model
        if sm and not sm.deleted:
            sm.scale = s

    def _scale_slider_cb(self, val):
        if self._block_text_update:
            return
        s = self._scale_from_slider(val)
        self._scale_entry.setText('%.3g' % s)
        sm = self._custom_shape_model
        if sm and not sm.deleted:
            sm.scale = s

    # --- Movement ---
    def move_shape(self, delta_xyz):
        sm = self._custom_shape_model
        if sm is None or sm.deleted:
            return
        dxyz = sm.scene_position.inverse().transform_vector(delta_xyz)
        from chimerax.geometry import translation
        sm.position = sm.position * translation(dxyz)

    # --- Show / color ---
    def _show_eraser_cb(self, show):
        sm = self._custom_shape_model
        if sm and not sm.deleted:
            sm.display = show

    def _change_color_cb(self, color):
        sm = self._custom_shape_model
        if sm and not sm.deleted:
            sm.color = color

    # --- Erase ---
    def _erase_inside(self):
        self._erase(outside=False)

    def _erase_outside(self):
        self._erase(outside=True)

    def _erase(self, outside=False):
        sm = self._custom_shape_model
        if sm is None or sm.deleted:
            self.session.logger.warning('No custom eraser shape set.')
            return
        if self._mask_array is None:
            self.session.logger.warning('No mask data loaded.')
            return

        v = self._shown_volume()
        if v is None:
            self.session.logger.warning(
                'No single displayed volume for custom erase')
            return

        vcopy = v.writable_copy()
        grid_data = vcopy.data

        saved = grid_data.full_matrix().copy()

        _erase_with_custom_shape(
            grid_data, sm, v.scene_position,
            self._mask_array, self._mask_xyz_to_ijk,
            sm.centroid, sm.scale,
            self._threshold,
            value=0, outside=outside
        )

        if np.array_equal(saved, grid_data.full_matrix()):
            return

        if self._last_undo_action is not None:
            try:
                self.session.undo.deregister(self._last_undo_action,
                                             delete_history=False)
            except Exception:
                pass
        action = _VolumeEraseUndo('custom erase', grid_data, saved)
        self._last_undo_action = action
        self.session.undo.register(action)

    def _crop_map(self):
        v = self._shown_volume()
        if v is None:
            self.session.logger.warning(
                'No single displayed volume for crop')
            return
        sm = self._custom_shape_model
        if sm is None or sm.deleted:
            self.session.logger.warning('No custom eraser shape set.')
            return
        ijk_min, ijk_max = _custom_grid_bounds(v.data, sm,
                                               v.scene_position)
        region = ','.join(['%d,%d,%d' % tuple(ijk_min),
                           '%d,%d,%d' % tuple(ijk_max)])
        cmd = 'volume copy #%s subregion %s' % (v.id_string, region)
        from chimerax.core.commands import run
        run(self.session, cmd)

    def _shown_volume(self):
        from chimerax.map import Volume
        sm = self._custom_shape_model
        vlist = [m for m in self.session.models.list(type=Volume)
                 if m.visible and m is not sm]
        return vlist[0] if len(vlist) == 1 else None


# -------------------------------------------------------------------------
#
def map_custom_eraser_panel(session, create=True):
    return MapCustomEraserSettings.get_singleton(session, create=create)


# -------------------------------------------------------------------------
#
def register_mousemode(session):
    mm = session.ui.mouse_modes
    mm.add_mode(MapCustomEraser(session))
