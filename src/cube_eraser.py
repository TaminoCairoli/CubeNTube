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

CUBE_ERASER_COLOR = (255, 153, 204, 128)  # transparent pink (matches sphere eraser)


class _VolumeEraseUndo(UndoAction):
    """Single-step undo/redo for volume erase operations.
    Snapshots the full grid matrix before erasing so Cmd+Z can restore it.
    """

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

        # Create temporary cube model transform
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
    cube_local = scene_to_cube.transform_points(scene_xyz)

    inside = (
        (np.abs(cube_local[:, 0]) <= hx)
        & (np.abs(cube_local[:, 1]) <= hy)
        & (np.abs(cube_local[:, 2]) <= hz)
    )
    # Reshape to meshgrid order (ni, nj, nk), then transpose to dmatrix order (nk, nj, ni)
    mask = inside.reshape(ni, nj, nk)
    mask = np.transpose(mask, (2, 1, 0))
    if outside:
        mask = ~mask
    putmask(dmatrix, mask, value)

    grid_data.values_changed()


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
    corners_scene = np.array([cube_to_scene * c for c in corners_local])
    corners_volume = np.array([scene_to_volume * c for c in corners_scene])

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


# -------------------------------------------------------------------------
#
from chimerax.mouse_modes import MouseMode


class MapCubeEraser(MouseMode):
    name = 'cube'
    icon_file = 'cubentube.png'

    def __init__(self, session):
        MouseMode.__init__(self, session)

    @property
    def settings(self):
        from .shape_eraser import map_shape_eraser_panel
        sp = map_shape_eraser_panel(self.session, create=False)
        if sp is not None:
            return sp
        return map_cube_eraser_panel(self.session)

    def enable(self):
        from .shape_eraser import map_shape_eraser_panel
        sp = map_shape_eraser_panel(self.session)
        sp._shape_combo.setCurrentIndex(0)
        sp.show()

    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)

    def mouse_drag(self, event):
        dx, dy = self.mouse_motion(event)
        settings = self.settings
        # Compute motion in scene coords of cube center.
        c = settings.cube_center
        v = self.session.main_view
        s = v.pixel_size(c)
        if event.shift_down():
            shift = (0, 0, s * dy)  # Move in z if shift key held.
        else:
            shift = (s * dx, -s * dy, 0)

        dxyz = v.camera.position.transform_vector(shift)
        settings.move_cube(dxyz)

    def mouse_up(self, event):
        MouseMode.mouse_up(self, event)

    def vr_motion(self, event):
        settings = self.settings
        c = settings.cube_center
        delta_xyz = event.motion * c - c
        settings.move_cube(delta_xyz)


# -----------------------------------------------------------------------------
# Panel for erasing parts of map in cube with map cube eraser mouse mode.
#
from chimerax.core.tools import ToolInstance


class MapCubeEraserSettings(ToolInstance):
    help = "help:user/tools/mapcubeeraser.html"

    def __init__(self, session, tool_name):

        self._default_color = CUBE_ERASER_COLOR
        self._max_slider_value = 1000  # QSlider only handles integer values
        self._max_slider_size = 100.0  # Float maximum size value, scene units
        self._block_text_update = False
        self._block_slider_update = False
        self._lock_dimensions = False
        self._last_undo_action = None

        b = session.main_view.drawing_bounds()
        vradius = 100 if b is None else b.radius()
        self._max_slider_size = vradius
        center = b.center() if b else (0, 0, 0)
        initial_size = 0.2 * vradius
        self._cube_model = CubeModel('eraser cube', session, self._default_color,
                                     center, initial_size, initial_size, initial_size)

        ToolInstance.__init__(self, session, tool_name)

        self.display_name = 'Cube Eraser'

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        self.tool_window = tw
        parent = tw.ui_area

        from Qt.QtWidgets import QVBoxLayout, QHBoxLayout, QFrame, QCheckBox, QLabel, QPushButton, QLineEdit, QSlider
        from Qt.QtCore import Qt

        layout = QVBoxLayout(parent)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        parent.setLayout(layout)

        # Show cube checkbox and color picker
        sf = QFrame(parent)
        layout.addWidget(sf)

        slayout = QHBoxLayout(sf)
        slayout.setContentsMargins(0, 0, 0, 0)
        slayout.setSpacing(10)

        self._show_eraser = se = QCheckBox('Show map eraser cube', sf)
        se.setCheckState(Qt.Checked)
        se.stateChanged.connect(self._show_eraser_cb)
        slayout.addWidget(se)

        from chimerax.ui.widgets import ColorButton
        self._cube_color = sc = ColorButton(sf, max_size=(16, 16), has_alpha_channel=True)
        sc.color = CUBE_ERASER_COLOR
        sc.color_changed.connect(self._change_color_cb)
        slayout.addWidget(sc)

        self._lock_cb = lc = QCheckBox('Lock XYZ', sf)
        lc.setCheckState(Qt.Unchecked)
        lc.stateChanged.connect(self._lock_toggled_cb)
        slayout.addWidget(lc)
        slayout.addStretch(1)

        # X size slider
        xf = QFrame(parent)
        layout.addWidget(xf)
        xlayout = QHBoxLayout(xf)
        xlayout.setContentsMargins(0, 0, 0, 0)
        xlayout.setSpacing(4)

        xl = QLabel('Size X', xf)
        xlayout.addWidget(xl)
        self._size_x_entry = xv = QLineEdit('', xf)
        xv.setMaximumWidth(40)
        xv.returnPressed.connect(self._size_x_changed_cb)
        xlayout.addWidget(xv)
        self._size_x_slider = xs = QSlider(Qt.Horizontal, xf)
        smax = self._max_slider_value
        xs.setRange(0, smax)
        xs.valueChanged.connect(self._size_x_slider_moved_cb)
        xlayout.addWidget(xs)

        # Y size slider
        yf = QFrame(parent)
        layout.addWidget(yf)
        ylayout = QHBoxLayout(yf)
        ylayout.setContentsMargins(0, 0, 0, 0)
        ylayout.setSpacing(4)

        yl = QLabel('Size Y', yf)
        ylayout.addWidget(yl)
        self._size_y_entry = yv = QLineEdit('', yf)
        yv.setMaximumWidth(40)
        yv.returnPressed.connect(self._size_y_changed_cb)
        ylayout.addWidget(yv)
        self._size_y_slider = ys = QSlider(Qt.Horizontal, yf)
        ys.setRange(0, smax)
        ys.valueChanged.connect(self._size_y_slider_moved_cb)
        ylayout.addWidget(ys)

        # Z size slider
        zf = QFrame(parent)
        layout.addWidget(zf)
        zlayout = QHBoxLayout(zf)
        zlayout.setContentsMargins(0, 0, 0, 0)
        zlayout.setSpacing(4)

        zl = QLabel('Size Z', zf)
        zlayout.addWidget(zl)
        self._size_z_entry = zv = QLineEdit('', zf)
        zv.setMaximumWidth(40)
        zv.returnPressed.connect(self._size_z_changed_cb)
        zlayout.addWidget(zv)
        self._size_z_slider = zs = QSlider(Qt.Horizontal, zf)
        zs.setRange(0, smax)
        zs.valueChanged.connect(self._size_z_slider_moved_cb)
        zlayout.addWidget(zs)

        # Initialize size values
        xv.setText('%.4g' % self._cube_model.size_x)
        yv.setText('%.4g' % self._cube_model.size_y)
        zv.setText('%.4g' % self._cube_model.size_z)
        self._size_x_changed_cb()
        self._size_y_changed_cb()
        self._size_z_changed_cb()

        # Erase buttons
        ef = QFrame(parent)
        layout.addWidget(ef)
        elayout = QHBoxLayout(ef)
        elayout.setContentsMargins(0, 0, 0, 0)
        elayout.setSpacing(30)

        eb = QPushButton('Erase inside cube', ef)
        eb.clicked.connect(self._erase_in_cube)
        elayout.addWidget(eb)

        eo = QPushButton('Erase outside cube', ef)
        eo.clicked.connect(self._erase_outside_cube)
        elayout.addWidget(eo)

        rb = QPushButton('Reduce map bounds', ef)
        rb.clicked.connect(self._crop_map)
        elayout.addWidget(rb)

        elayout.addStretch(1)

        layout.addStretch(1)

        tw.manage(placement="side")

        # When displayed models change update size slider range.
        from chimerax.core.models import MODEL_DISPLAY_CHANGED
        h = session.triggers.add_handler(MODEL_DISPLAY_CHANGED, self._model_display_change)
        self._model_display_change_handler = h

    def delete(self):
        ses = self.session
        ses.triggers.remove_handler(self._model_display_change_handler)
        cm = self._cube_model
        if cm and not cm.deleted:
            ses.models.close([cm])
        self._cube_model = None
        ToolInstance.delete(self)

    @classmethod
    def get_singleton(self, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, MapCubeEraserSettings, 'Cube Eraser', create=create)

    def show(self):
        self.tool_window.shown = True

    def hide(self):
        self.tool_window.shown = False

    @property
    def cube_model(self):
        cm = self._cube_model
        if cm is None or cm.deleted:
            b = self.session.main_view.drawing_bounds()
            center = b.center() if b else (0, 0, 0)
            cm = CubeModel('eraser cube', self.session, CUBE_ERASER_COLOR,
                           center, self._size_x_value(), self._size_y_value(), self._size_z_value())
            self._cube_model = cm
        return cm

    # --- Lock toggle ---
    def _lock_toggled_cb(self, state):
        self._lock_dimensions = bool(state)
        if self._lock_dimensions:
            self._set_all_sizes(self._size_x_value())

    def _set_all_sizes(self, s):
        '''Set all three dimensions to the same value (used when lock is on).'''
        self._block_text_update = True
        sval = int((s / self._max_slider_size) * self._max_slider_value)
        self._size_x_entry.setText('%.4g' % s)
        self._size_x_slider.setValue(sval)
        self._size_y_entry.setText('%.4g' % s)
        self._size_y_slider.setValue(sval)
        self._size_z_entry.setText('%.4g' % s)
        self._size_z_slider.setValue(sval)
        self._block_text_update = False
        cm = self.cube_model
        cm.size_x = s
        cm.size_y = s
        cm.size_z = s

    # --- Size X methods ---
    def _size_x_changed_cb(self):
        if self._block_text_update:
            return
        s = self._size_x_value()
        if self._lock_dimensions:
            self._set_all_sizes(s)
            return
        self.cube_model.size_x = s
        sval = int((s / self._max_slider_size) * self._max_slider_value)
        self._block_text_update = True
        self._size_x_slider.setValue(sval)
        self._block_text_update = False

    def _size_x_value(self):
        rt = self._size_x_entry.text()
        try:
            r = float(rt)
        except ValueError:
            self.session.logger.warning('Cannot parse map eraser size X value "%s"' % rt)
            return 10
        return r

    def _size_x_slider_moved_cb(self, event):
        if self._block_text_update:
            return
        sval = self._size_x_slider.value()
        s = (sval / self._max_slider_value) * self._max_slider_size
        if self._lock_dimensions:
            self._set_all_sizes(s)
            return
        self._size_x_entry.setText('%.4g' % s)
        self.cube_model.size_x = s

    # --- Size Y methods ---
    def _size_y_changed_cb(self):
        if self._block_text_update:
            return
        s = self._size_y_value()
        if self._lock_dimensions:
            self._set_all_sizes(s)
            return
        self.cube_model.size_y = s
        sval = int((s / self._max_slider_size) * self._max_slider_value)
        self._block_text_update = True
        self._size_y_slider.setValue(sval)
        self._block_text_update = False

    def _size_y_value(self):
        rt = self._size_y_entry.text()
        try:
            r = float(rt)
        except ValueError:
            self.session.logger.warning('Cannot parse map eraser size Y value "%s"' % rt)
            return 10
        return r

    def _size_y_slider_moved_cb(self, event):
        if self._block_text_update:
            return
        sval = self._size_y_slider.value()
        s = (sval / self._max_slider_value) * self._max_slider_size
        if self._lock_dimensions:
            self._set_all_sizes(s)
            return
        self._size_y_entry.setText('%.4g' % s)
        self.cube_model.size_y = s

    # --- Size Z methods ---
    def _size_z_changed_cb(self):
        if self._block_text_update:
            return
        s = self._size_z_value()
        if self._lock_dimensions:
            self._set_all_sizes(s)
            return
        self.cube_model.size_z = s
        sval = int((s / self._max_slider_size) * self._max_slider_value)
        self._block_text_update = True
        self._size_z_slider.setValue(sval)
        self._block_text_update = False

    def _size_z_value(self):
        rt = self._size_z_entry.text()
        try:
            r = float(rt)
        except ValueError:
            self.session.logger.warning('Cannot parse map eraser size Z value "%s"' % rt)
            return 10
        return r

    def _size_z_slider_moved_cb(self, event):
        if self._block_text_update:
            return
        sval = self._size_z_slider.value()
        s = (sval / self._max_slider_value) * self._max_slider_size
        if self._lock_dimensions:
            self._set_all_sizes(s)
            return
        self._size_z_entry.setText('%.4g' % s)
        self.cube_model.size_z = s

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
            self._size_x_changed_cb()
            self._size_y_changed_cb()
            self._size_z_changed_cb()

    @property
    def cube_center(self):
        return self.cube_model.scene_position.origin()
    
    def move_cube(self, delta_xyz):
        cm = self.cube_model
        dxyz = cm.scene_position.inverse().transform_vector(delta_xyz)
        from chimerax.geometry import translation
        cm.position = cm.position * translation(dxyz)

    def _show_eraser_cb(self, show):
        self.cube_model.display = show

    def _erase_in_cube(self):
        self._erase()

    def _erase(self, outside=False):
        v = self._shown_volume()
        if v is None:
            self.session.logger.warning('No single displayed volume for cube erase')
            return
        cube = self.cube_model
        vcopy = v.writable_copy()
        grid_data = vcopy.data

        saved = grid_data.full_matrix().copy()

        _erase_with_cube_model(
            grid_data, cube, v.scene_position,
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
        action = _VolumeEraseUndo('cube erase', grid_data, saved)
        self._last_undo_action = action
        self.session.undo.register(action)

    def _erase_outside_cube(self):
        self._erase(outside=True)

    def _crop_map(self):
        v = self._shown_volume()
        if v is None:
            self.session.logger.warning('No single displayed volume for crop')
            return
        cube = self.cube_model
        ijk_min, ijk_max = _cube_grid_bounds(v.data, cube, v.scene_position)
        region = ','.join(['%d,%d,%d' % tuple(ijk_min),
                           '%d,%d,%d' % tuple(ijk_max)])
        cmd = 'volume copy #%s subregion %s' % (v.id_string, region)
        from chimerax.core.commands import run
        run(self.session, cmd)

    def _shown_volume(self):
        ses = self.session
        from chimerax.map import Volume
        vlist = [m for m in ses.models.list(type=Volume) if m.visible]
        v = vlist[0] if len(vlist) == 1 else None
        return v

    def _change_color_cb(self, color):
        self.cube_model.color = color


# -------------------------------------------------------------------------
#
from chimerax.core.models import Surface


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

        # Half sizes
        hx = self._size_x / 2
        hy = self._size_y / 2
        hz = self._size_z / 2

        # 8 vertices of the box
        vertices = np.array([
            [-hx, -hy, -hz],  # 0: back bottom left
            [hx, -hy, -hz],   # 1: back bottom right
            [hx, hy, -hz],    # 2: back top right
            [-hx, hy, -hz],   # 3: back top left
            [-hx, -hy, hz],   # 4: front bottom left
            [hx, -hy, hz],    # 5: front bottom right
            [hx, hy, hz],     # 6: front top right
            [-hx, hy, hz],    # 7: front top left
        ], dtype=np.float32)

        # 12 triangles (2 per face, 6 faces) with proper face vertices and normals
        face_vertices = np.array([
            # Back face
            [-hx, -hy, -hz], [hx, hy, -hz], [hx, -hy, -hz],
            [-hx, -hy, -hz], [-hx, hy, -hz], [hx, hy, -hz],
            # Front face
            [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz],
            [-hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
            # Bottom face
            [-hx, -hy, -hz], [hx, -hy, -hz], [hx, -hy, hz],
            [-hx, -hy, -hz], [hx, -hy, hz], [-hx, -hy, hz],
            # Top face
            [hx, hy, -hz], [-hx, hy, -hz], [-hx, hy, hz],
            [hx, hy, -hz], [-hx, hy, hz], [hx, hy, hz],
            # Left face
            [-hx, -hy, -hz], [-hx, -hy, hz], [-hx, hy, hz],
            [-hx, -hy, -hz], [-hx, hy, hz], [-hx, hy, -hz],
            # Right face
            [hx, -hy, -hz], [hx, hy, -hz], [hx, hy, hz],
            [hx, -hy, -hz], [hx, hy, hz], [hx, -hy, hz],
        ], dtype=np.float32)

        face_normals = np.array([
            # Back face (6 vertices)
            [0, 0, -1], [0, 0, -1], [0, 0, -1],
            [0, 0, -1], [0, 0, -1], [0, 0, -1],
            # Front face
            [0, 0, 1], [0, 0, 1], [0, 0, 1],
            [0, 0, 1], [0, 0, 1], [0, 0, 1],
            # Bottom face
            [0, -1, 0], [0, -1, 0], [0, -1, 0],
            [0, -1, 0], [0, -1, 0], [0, -1, 0],
            # Top face
            [0, 1, 0], [0, 1, 0], [0, 1, 0],
            [0, 1, 0], [0, 1, 0], [0, 1, 0],
            # Left face
            [-1, 0, 0], [-1, 0, 0], [-1, 0, 0],
            [-1, 0, 0], [-1, 0, 0], [-1, 0, 0],
            # Right face
            [1, 0, 0], [1, 0, 0], [1, 0, 0],
            [1, 0, 0], [1, 0, 0], [1, 0, 0],
        ], dtype=np.float32)

        face_triangles = np.array([
            [0, 1, 2], [3, 4, 5],       # Back
            [6, 7, 8], [9, 10, 11],     # Front
            [12, 13, 14], [15, 16, 17], # Bottom
            [18, 19, 20], [21, 22, 23], # Top
            [24, 25, 26], [27, 28, 29], # Left
            [30, 31, 32], [33, 34, 35], # Right
        ], dtype=np.int32)

        self.set_geometry(face_vertices, face_normals, face_triangles)

    # Size X property
    def _get_size_x(self):
        return self._size_x

    def _set_size_x(self, s):
        if s != self._size_x:
            self._size_x = s
            self._update_geometry()

    size_x = property(_get_size_x, _set_size_x)

    # Size Y property
    def _get_size_y(self):
        return self._size_y

    def _set_size_y(self, s):
        if s != self._size_y:
            self._size_y = s
            self._update_geometry()

    size_y = property(_get_size_y, _set_size_y)

    # Size Z property
    def _get_size_z(self):
        return self._size_z

    def _set_size_z(self, s):
        if s != self._size_z:
            self._size_z = s
            self._update_geometry()

    size_z = property(_get_size_z, _set_size_z)


# -------------------------------------------------------------------------
#
def map_cube_eraser_panel(session, create=True):
    return MapCubeEraserSettings.get_singleton(session, create=create)


# -------------------------------------------------------------------------
#
def register_mousemode(session):
    mm = session.ui.mouse_modes
    mm.add_mode(MapCubeEraser(session))