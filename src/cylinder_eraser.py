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

CYLINDER_ERASER_COLOR = (255, 153, 204, 128)  # transparent pink (matches sphere eraser)


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

    # Reshape to meshgrid order (ni, nj, nk), then transpose to dmatrix order (nk, nj, ni)
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


# -------------------------------------------------------------------------
#
from chimerax.mouse_modes import MouseMode


class MapCylinderEraser(MouseMode):
    name = 'cylinder'
    icon_file = 'cubentube.png'

    def __init__(self, session):
        MouseMode.__init__(self, session)

    @property
    def settings(self):
        from .shape_eraser import map_shape_eraser_panel
        sp = map_shape_eraser_panel(self.session, create=False)
        if sp is not None:
            return sp
        return map_cylinder_eraser_panel(self.session)

    def enable(self):
        from .shape_eraser import map_shape_eraser_panel
        sp = map_shape_eraser_panel(self.session)
        sp._shape_combo.setCurrentIndex(1)
        sp.show()

    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)

    def mouse_drag(self, event):
        dx, dy = self.mouse_motion(event)
        settings = self.settings
        c = settings.cylinder_center
        v = self.session.main_view
        s = v.pixel_size(c)
        if event.shift_down():
            shift = (0, 0, s * dy)
        else:
            shift = (s * dx, -s * dy, 0)
        dxyz = v.camera.position.transform_vector(shift)
        settings.move_cylinder(dxyz)

    def mouse_up(self, event):
        MouseMode.mouse_up(self, event)

    def vr_motion(self, event):
        settings = self.settings
        c = settings.cylinder_center
        delta_xyz = event.motion * c - c
        settings.move_cylinder(delta_xyz)


# -------------------------------------------------------------------------
#
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

        # --- Side wall (separate vertices so caps get sharp edges) ---
        # Each side quad: 2 bottom + 2 top verts with outward-pointing normals.
        side_base = 0
        slant_z = rb - rt
        slant_len = math.sqrt(max(rb, rt) ** 2 + slant_z ** 2) or 1.0
        for i in range(n):
            j = (i + 1) % n
            cx0, sy0 = cos[i], sin[i]
            cx1, sy1 = cos[j], sin[j]
            n0 = [cx0 / slant_len, sy0 / slant_len, slant_z / slant_len]
            n1 = [cx1 / slant_len, sy1 / slant_len, slant_z / slant_len]
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

        # --- Bottom cap (flat normal pointing -Z) ---
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

        # --- Top cap (flat normal pointing +Z) ---
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


# -------------------------------------------------------------------------
# Panel for erasing parts of map in cylinder
#
from chimerax.core.tools import ToolInstance


class MapCylinderEraserSettings(ToolInstance):
    help = "help:user/tools/mapcylindereraser.html"

    def __init__(self, session, tool_name):

        self._default_color = CYLINDER_ERASER_COLOR
        self._max_slider_value = 1000
        self._max_slider_size = 100.0
        self._block_text_update = False
        self._lock_radii = True
        self._last_undo_action = None

        b = session.main_view.drawing_bounds()
        vradius = 100 if b is None else b.radius()
        self._max_slider_size = vradius
        center = b.center() if b else (0, 0, 0)
        initial_r = 0.2 * vradius
        initial_len = 0.2 * vradius
        self._cylinder_model = CylinderModel(
            'eraser cylinder', session, self._default_color,
            center, initial_r, initial_r, initial_len
        )

        ToolInstance.__init__(self, session, tool_name)
        self.display_name = 'Cylinder Eraser'

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

        # --- Top row: show checkbox, color, lock radii ---
        sf = QFrame(parent)
        layout.addWidget(sf)
        slayout = QHBoxLayout(sf)
        slayout.setContentsMargins(0, 0, 0, 0)
        slayout.setSpacing(10)

        self._show_eraser = se = QCheckBox('Show eraser cylinder', sf)
        se.setCheckState(Qt.Checked)
        se.stateChanged.connect(self._show_eraser_cb)
        slayout.addWidget(se)

        from chimerax.ui.widgets import ColorButton
        self._cyl_color = sc = ColorButton(sf, max_size=(16, 16),
                                           has_alpha_channel=True)
        sc.color = CYLINDER_ERASER_COLOR
        sc.color_changed.connect(self._change_color_cb)
        slayout.addWidget(sc)

        self._lock_cb = lc = QCheckBox('Lock Radii', sf)
        lc.setCheckState(Qt.Checked)
        lc.stateChanged.connect(self._lock_toggled_cb)
        slayout.addWidget(lc)
        slayout.addStretch(1)

        # --- Radius Top slider ---
        rtf = QFrame(parent)
        layout.addWidget(rtf)
        rtl = QHBoxLayout(rtf)
        rtl.setContentsMargins(0, 0, 0, 0)
        rtl.setSpacing(4)

        rtl.addWidget(QLabel('Radius Top', rtf))
        self._radius_top_entry = rte = QLineEdit('', rtf)
        rte.setMaximumWidth(40)
        rte.returnPressed.connect(self._radius_top_changed_cb)
        rtl.addWidget(rte)
        self._radius_top_slider = rts = QSlider(Qt.Horizontal, rtf)
        rts.setRange(0, self._max_slider_value)
        rts.valueChanged.connect(self._radius_top_slider_cb)
        rtl.addWidget(rts)

        # --- Radius Bottom slider ---
        rbf = QFrame(parent)
        layout.addWidget(rbf)
        rbl = QHBoxLayout(rbf)
        rbl.setContentsMargins(0, 0, 0, 0)
        rbl.setSpacing(4)

        rbl.addWidget(QLabel('Radius Bot', rbf))
        self._radius_bottom_entry = rbe = QLineEdit('', rbf)
        rbe.setMaximumWidth(40)
        rbe.returnPressed.connect(self._radius_bottom_changed_cb)
        rbl.addWidget(rbe)
        self._radius_bottom_slider = rbs = QSlider(Qt.Horizontal, rbf)
        rbs.setRange(0, self._max_slider_value)
        rbs.valueChanged.connect(self._radius_bottom_slider_cb)
        rbl.addWidget(rbs)

        # --- Length slider ---
        lf = QFrame(parent)
        layout.addWidget(lf)
        ll = QHBoxLayout(lf)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(4)

        ll.addWidget(QLabel('Length', lf))
        self._length_entry = le = QLineEdit('', lf)
        le.setMaximumWidth(40)
        le.returnPressed.connect(self._length_changed_cb)
        ll.addWidget(le)
        self._length_slider = ls = QSlider(Qt.Horizontal, lf)
        ls.setRange(0, self._max_slider_value)
        ls.valueChanged.connect(self._length_slider_cb)
        ll.addWidget(ls)

        # Initialize text values
        rte.setText('%.4g' % self._cylinder_model.radius_top)
        rbe.setText('%.4g' % self._cylinder_model.radius_bottom)
        le.setText('%.4g' % self._cylinder_model.length)
        self._radius_top_changed_cb()
        self._radius_bottom_changed_cb()
        self._length_changed_cb()

        # --- Erase buttons ---
        ef = QFrame(parent)
        layout.addWidget(ef)
        elayout = QHBoxLayout(ef)
        elayout.setContentsMargins(0, 0, 0, 0)
        elayout.setSpacing(30)

        eb = QPushButton('Erase inside cylinder', ef)
        eb.clicked.connect(self._erase_in_cylinder)
        elayout.addWidget(eb)

        eo = QPushButton('Erase outside cylinder', ef)
        eo.clicked.connect(self._erase_outside_cylinder)
        elayout.addWidget(eo)

        rb_btn = QPushButton('Reduce map bounds', ef)
        rb_btn.clicked.connect(self._crop_map)
        elayout.addWidget(rb_btn)

        elayout.addStretch(1)
        layout.addStretch(1)

        tw.manage(placement='side')

        from chimerax.core.models import MODEL_DISPLAY_CHANGED
        h_handler = session.triggers.add_handler(MODEL_DISPLAY_CHANGED,
                                                 self._model_display_change)
        self._model_display_change_handler = h_handler

    def delete(self):
        ses = self.session
        ses.triggers.remove_handler(self._model_display_change_handler)
        cm = self._cylinder_model
        if cm and not cm.deleted:
            ses.models.close([cm])
        self._cylinder_model = None
        ToolInstance.delete(self)

    @classmethod
    def get_singleton(cls, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, MapCylinderEraserSettings,
                                   'Cylinder Eraser', create=create)

    def show(self):
        self.tool_window.shown = True

    def hide(self):
        self.tool_window.shown = False

    @property
    def cylinder_model(self):
        cm = self._cylinder_model
        if cm is None or cm.deleted:
            b = self.session.main_view.drawing_bounds()
            center = b.center() if b else (0, 0, 0)
            cm = CylinderModel('eraser cylinder', self.session,
                               CYLINDER_ERASER_COLOR, center,
                               self._radius_top_value(),
                               self._radius_bottom_value(),
                               self._length_value())
            self._cylinder_model = cm
        return cm

    # --- Lock toggle ---
    def _lock_toggled_cb(self, state):
        self._lock_radii = bool(state)
        if self._lock_radii:
            self._set_both_radii(self._radius_top_value())

    def _set_both_radii(self, r):
        '''Set both radii to the same value (used when lock is on).'''
        self._block_text_update = True
        sval = int((r / self._max_slider_size) * self._max_slider_value)
        self._radius_top_entry.setText('%.4g' % r)
        self._radius_top_slider.setValue(sval)
        self._radius_bottom_entry.setText('%.4g' % r)
        self._radius_bottom_slider.setValue(sval)
        self._block_text_update = False
        cm = self.cylinder_model
        cm.radius_top = r
        cm.radius_bottom = r

    # --- Radius Top ---
    def _radius_top_changed_cb(self):
        if self._block_text_update:
            return
        r = self._radius_top_value()
        if self._lock_radii:
            self._set_both_radii(r)
            return
        self.cylinder_model.radius_top = r
        sval = int((r / self._max_slider_size) * self._max_slider_value)
        self._block_text_update = True
        self._radius_top_slider.setValue(sval)
        self._block_text_update = False

    def _radius_top_value(self):
        try:
            return float(self._radius_top_entry.text())
        except ValueError:
            return 10

    def _radius_top_slider_cb(self, val):
        if self._block_text_update:
            return
        r = (val / self._max_slider_value) * self._max_slider_size
        if self._lock_radii:
            self._set_both_radii(r)
            return
        self._radius_top_entry.setText('%.4g' % r)
        self.cylinder_model.radius_top = r

    # --- Radius Bottom ---
    def _radius_bottom_changed_cb(self):
        if self._block_text_update:
            return
        r = self._radius_bottom_value()
        if self._lock_radii:
            self._set_both_radii(r)
            return
        self.cylinder_model.radius_bottom = r
        sval = int((r / self._max_slider_size) * self._max_slider_value)
        self._block_text_update = True
        self._radius_bottom_slider.setValue(sval)
        self._block_text_update = False

    def _radius_bottom_value(self):
        try:
            return float(self._radius_bottom_entry.text())
        except ValueError:
            return 10

    def _radius_bottom_slider_cb(self, val):
        if self._block_text_update:
            return
        r = (val / self._max_slider_value) * self._max_slider_size
        if self._lock_radii:
            self._set_both_radii(r)
            return
        self._radius_bottom_entry.setText('%.4g' % r)
        self.cylinder_model.radius_bottom = r

    # --- Length ---
    def _length_changed_cb(self):
        if self._block_text_update:
            return
        L = self._length_value()
        self.cylinder_model.length = L
        sval = int((L / self._max_slider_size) * self._max_slider_value)
        self._block_text_update = True
        self._length_slider.setValue(sval)
        self._block_text_update = False

    def _length_value(self):
        try:
            return float(self._length_entry.text())
        except ValueError:
            return 10

    def _length_slider_cb(self, val):
        if self._block_text_update:
            return
        L = (val / self._max_slider_value) * self._max_slider_size
        self._length_entry.setText('%.4g' % L)
        self.cylinder_model.length = L

    # --- Other methods ---
    def _model_display_change(self, name, data):
        v = self._shown_volume()
        if v:
            self._adjust_slider_range(v)

    def _adjust_slider_range(self, volume):
        xyz_min, xyz_max = volume.xyz_bounds(subregion='all')
        smax = max([x1 - x0 for x0, x1 in zip(xyz_min, xyz_max)])
        if smax != self._max_slider_size:
            self._max_slider_size = smax
            self._radius_top_changed_cb()
            self._radius_bottom_changed_cb()
            self._length_changed_cb()

    @property
    def cylinder_center(self):
        return self.cylinder_model.scene_position.origin()

    def move_cylinder(self, delta_xyz):
        cm = self.cylinder_model
        dxyz = cm.scene_position.inverse().transform_vector(delta_xyz)
        from chimerax.geometry import translation
        cm.position = cm.position * translation(dxyz)

    def _show_eraser_cb(self, show):
        self.cylinder_model.display = show

    def _change_color_cb(self, color):
        self.cylinder_model.color = color

    def _erase_in_cylinder(self):
        self._erase()

    def _erase_outside_cylinder(self):
        self._erase(outside=True)

    def _erase(self, outside=False):
        v = self._shown_volume()
        if v is None:
            self.session.logger.warning(
                'No single displayed volume for cylinder erase')
            return
        cyl = self.cylinder_model
        vcopy = v.writable_copy()
        grid_data = vcopy.data

        saved = grid_data.full_matrix().copy()

        _erase_with_cylinder_model(
            grid_data, cyl, v.scene_position,
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
        action = _VolumeEraseUndo('cylinder erase', grid_data, saved)
        self._last_undo_action = action
        self.session.undo.register(action)

    def _crop_map(self):
        v = self._shown_volume()
        if v is None:
            self.session.logger.warning(
                'No single displayed volume for crop')
            return
        cyl = self.cylinder_model
        ijk_min, ijk_max = _cylinder_grid_bounds(v.data, cyl,
                                                 v.scene_position)
        region = ','.join(['%d,%d,%d' % tuple(ijk_min),
                           '%d,%d,%d' % tuple(ijk_max)])
        cmd = 'volume copy #%s subregion %s' % (v.id_string, region)
        from chimerax.core.commands import run
        run(self.session, cmd)

    def _shown_volume(self):
        from chimerax.map import Volume
        vlist = [m for m in self.session.models.list(type=Volume)
                 if m.visible]
        return vlist[0] if len(vlist) == 1 else None


# -------------------------------------------------------------------------
#
def map_cylinder_eraser_panel(session, create=True):
    return MapCylinderEraserSettings.get_singleton(session, create=create)


# -------------------------------------------------------------------------
#
def register_mousemode(session):
    mm = session.ui.mouse_modes
    mm.add_mode(MapCylinderEraser(session))
