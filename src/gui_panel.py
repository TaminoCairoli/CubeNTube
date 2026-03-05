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

from chimerax.core.tools import ToolInstance
from .undo import VolumeEraseUndo


# Index constants for the stacked widget pages
_CUBE, _CYLINDER, _CUSTOM, _DUST = 0, 1, 2, 3


class MapShapeEraserSettings(ToolInstance):
    """Unified panel for cube, cylinder, custom and dust volume erasers."""

    help = "help:user/tools/shapeeraser.html"

    def __init__(self, session, tool_name):

        self._max_slider_value = 1000
        self._max_slider_size = 100.0
        self._block_text_update = False
        self._lock_dimensions = False
        self._lock_radii = True
        self._last_undo_action = None

        self._custom_shape_model = None
        self._mask_array = None
        self._mask_xyz_to_ijk = None
        self._mask_matrix_id = None
        self._threshold = None
        self._mask_volume = None
        self._threshold_min = 0.0
        self._threshold_max = 1.0
        self._custom_contour_cache = {'level': None, 'matrix_id': None,
                                      'verts': None, 'norms': None, 'tris': None}

        self._max_slider_scale = 5.0
        self._min_slider_scale = 0.1

        # Dust state
        self._dust_active = False
        self._dust_highlight_model = None
        self._dust_preview_data = None
        self._dust_size_range = (1, 1000)
        self._dust_last_level = None
        self._dust_last_volume = None

        b = session.main_view.drawing_bounds()
        vradius = 100 if b is None else b.radius()
        self._max_slider_size = vradius
        center = b.center() if b else (0, 0, 0)
        initial = 0.2 * vradius

        from .cube_eraser import CUBE_ERASER_COLOR
        from .cylinder_eraser import CYLINDER_ERASER_COLOR
        from .custom_eraser import CUSTOM_ERASER_COLOR

        self._cube_color = CUBE_ERASER_COLOR
        self._cyl_color = CYLINDER_ERASER_COLOR
        self._custom_color = CUSTOM_ERASER_COLOR

        self._cube_model = None
        self._cylinder_model = None
        self._prev_right_mode = None

        ToolInstance.__init__(self, session, tool_name)
        self.display_name = "Cube'n Tube"

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        self.tool_window = tw
        parent = tw.ui_area

        from Qt.QtWidgets import (QVBoxLayout, QHBoxLayout, QFrame,
                                  QCheckBox, QLabel, QPushButton,
                                  QLineEdit, QSlider, QComboBox,
                                  QStackedWidget)
        from Qt.QtCore import Qt

        layout = QVBoxLayout(parent)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        parent.setLayout(layout)

        # ---- Top row: show, color, shape selector ----
        tf = QFrame(parent)
        layout.addWidget(tf)
        tl = QHBoxLayout(tf)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(10)

        self._show_cb = sc = QCheckBox('Show eraser', tf)
        sc.setCheckState(Qt.Checked)
        sc.stateChanged.connect(self._show_eraser_cb)
        tl.addWidget(sc)

        from chimerax.ui.widgets import ColorButton
        self._color_btn = cb = ColorButton(tf, max_size=(16, 16),
                                           has_alpha_channel=True)
        cb.color = CUBE_ERASER_COLOR
        cb.color_changed.connect(self._change_color_cb)
        tl.addWidget(cb)

        tl.addWidget(QLabel('Shape:', tf))
        self._shape_combo = combo = QComboBox(tf)
        combo.addItems(['Cube', 'Cylinder', 'Custom', 'Dust'])
        combo.currentIndexChanged.connect(self._shape_changed)
        tl.addWidget(combo)
        tl.addStretch(1)

        # ---- Stacked widget for shape-specific controls ----
        self._stack = stack = QStackedWidget(parent)
        layout.addWidget(stack)

        # -- Page 0: Cube --
        cube_page = QFrame()
        cpl = QVBoxLayout(cube_page)
        cpl.setContentsMargins(0, 0, 0, 0)
        cpl.setSpacing(0)

        clf = QFrame(cube_page)
        cpl.addWidget(clf)
        cll = QHBoxLayout(clf)
        cll.setContentsMargins(0, 0, 0, 0)
        cll.setSpacing(10)
        self._cube_lock_cb = clc = QCheckBox('Lock XYZ', clf)
        clc.setCheckState(Qt.Unchecked)
        clc.stateChanged.connect(self._cube_lock_toggled)
        cll.addWidget(clc)
        cll.addStretch(1)

        smax = self._max_slider_value
        for axis in ('X', 'Y', 'Z'):
            f = QFrame(cube_page)
            cpl.addWidget(f)
            hl = QHBoxLayout(f)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(4)
            hl.addWidget(QLabel('Size ' + axis, f))
            entry = QLineEdit('%.4g' % initial, f)
            entry.setMaximumWidth(40)
            hl.addWidget(entry)
            slider = QSlider(Qt.Horizontal, f)
            slider.setRange(0, smax)
            hl.addWidget(slider)
            setattr(self, '_cube_size_%s_entry' % axis.lower(), entry)
            setattr(self, '_cube_size_%s_slider' % axis.lower(), slider)
            entry.returnPressed.connect(
                getattr(self, '_cube_size_%s_text' % axis.lower()))
            slider.valueChanged.connect(
                getattr(self, '_cube_size_%s_slide' % axis.lower()))

        self._cube_size_x_text()
        self._cube_size_y_text()
        self._cube_size_z_text()
        stack.addWidget(cube_page)

        # -- Page 1: Cylinder --
        cyl_page = QFrame()
        cypl = QVBoxLayout(cyl_page)
        cypl.setContentsMargins(0, 0, 0, 0)
        cypl.setSpacing(0)

        cylf = QFrame(cyl_page)
        cypl.addWidget(cylf)
        cyll = QHBoxLayout(cylf)
        cyll.setContentsMargins(0, 0, 0, 0)
        cyll.setSpacing(10)
        self._cyl_lock_cb = cylc = QCheckBox('Lock Radii', cylf)
        cylc.setCheckState(Qt.Checked)
        cylc.stateChanged.connect(self._cyl_lock_toggled)
        cyll.addWidget(cylc)
        cyll.addStretch(1)

        for label, attr in (('Radius Top', 'rt'), ('Radius Bot', 'rb'),
                            ('Length', 'ln')):
            f = QFrame(cyl_page)
            cypl.addWidget(f)
            hl = QHBoxLayout(f)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(4)
            hl.addWidget(QLabel(label, f))
            entry = QLineEdit('%.4g' % initial, f)
            entry.setMaximumWidth(40)
            hl.addWidget(entry)
            slider = QSlider(Qt.Horizontal, f)
            slider.setRange(0, smax)
            hl.addWidget(slider)
            setattr(self, '_cyl_%s_entry' % attr, entry)
            setattr(self, '_cyl_%s_slider' % attr, slider)
            entry.returnPressed.connect(
                getattr(self, '_cyl_%s_text' % attr))
            slider.valueChanged.connect(
                getattr(self, '_cyl_%s_slide' % attr))

        self._cyl_rt_text()
        self._cyl_rb_text()
        self._cyl_ln_text()
        stack.addWidget(cyl_page)

        # -- Page 2: Custom --
        cust_page = QFrame()
        cupl = QVBoxLayout(cust_page)
        cupl.setContentsMargins(0, 0, 0, 0)
        cupl.setSpacing(0)

        vf = QFrame(cust_page)
        cupl.addWidget(vf)
        vl = QHBoxLayout(vf)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(10)
        vl.addWidget(QLabel('Mask volume:', vf))
        from chimerax.ui.widgets import ModelMenuButton
        from chimerax.map import Volume
        self._volume_menu = vm = ModelMenuButton(
            session, class_filter=Volume,
            no_value_button_text='Choose volume...')
        vl.addWidget(vm)
        self._set_btn = sb = QPushButton('Set as eraser', vf)
        sb.clicked.connect(self._set_eraser_cb)
        vl.addWidget(sb)
        vl.addStretch(1)

        thf = QFrame(cust_page)
        cupl.addWidget(thf)
        thl = QHBoxLayout(thf)
        thl.setContentsMargins(0, 0, 0, 0)
        thl.setSpacing(4)
        thl.addWidget(QLabel('Threshold', thf))
        self._threshold_entry = te = QLineEdit('0', thf)
        te.setMaximumWidth(60)
        te.returnPressed.connect(self._threshold_text_cb)
        thl.addWidget(te)
        self._threshold_slider = ths = QSlider(Qt.Horizontal, thf)
        ths.setRange(0, smax)
        ths.valueChanged.connect(self._threshold_slide_cb)
        thl.addWidget(ths)

        sf = QFrame(cust_page)
        cupl.addWidget(sf)
        sl = QHBoxLayout(sf)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(4)
        sl.addWidget(QLabel('Scale', sf))
        self._scale_entry = se = QLineEdit('1.0', sf)
        se.setMaximumWidth(40)
        se.returnPressed.connect(self._scale_text_cb)
        sl.addWidget(se)
        self._scale_slider = ss = QSlider(Qt.Horizontal, sf)
        ss.setRange(0, smax)
        ss.valueChanged.connect(self._scale_slide_cb)
        sl.addWidget(ss)
        self._set_scale_slider(1.0)
        stack.addWidget(cust_page)

        # -- Page 3: Dust --
        dust_page = QFrame()
        dpl = QVBoxLayout(dust_page)
        dpl.setContentsMargins(0, 0, 0, 0)
        dpl.setSpacing(0)

        dvf = QFrame(dust_page)
        dpl.addWidget(dvf)
        dvl = QHBoxLayout(dvf)
        dvl.setContentsMargins(0, 0, 0, 0)
        dvl.setSpacing(10)
        dvl.addWidget(QLabel('Dust surface:', dvf))
        self._dust_volume_menu = ModelMenuButton(
            session, class_filter=Volume,
            no_value_button_text='Choose volume...')
        dvl.addWidget(self._dust_volume_menu)
        dvl.addStretch(1)

        from chimerax.ui.widgets import LogSlider
        self._dust_slider = LogSlider(
            dust_page, label='Size limit',
            range=self._dust_size_range,
            value_change_cb=self._dust_size_changed,
            release_cb=self._dust_slider_released)
        dpl.addWidget(self._dust_slider.frame)
        stack.addWidget(dust_page)

        self._dust_frame_handler = None

        # ---- Bottom row: erase buttons ----
        ef = QFrame(parent)
        layout.addWidget(ef)
        el = QHBoxLayout(ef)
        el.setContentsMargins(0, 0, 0, 0)
        el.setSpacing(30)

        self._erase_inside_btn = eb = QPushButton('Erase inside', ef)
        eb.clicked.connect(lambda: self._erase(outside=False))
        el.addWidget(eb)

        self._erase_outside_btn = eo = QPushButton('Erase outside', ef)
        eo.clicked.connect(lambda: self._erase(outside=True))
        el.addWidget(eo)

        self._crop_btn = rb = QPushButton('Reduce map bounds', ef)
        rb.clicked.connect(self._crop_map)
        el.addWidget(rb)

        el.addStretch(1)
        layout.addStretch(1)

        tw.manage(placement='side')

        from chimerax.core.models import MODEL_DISPLAY_CHANGED, MODEL_SELECTION_CHANGED
        self._mdch = session.triggers.add_handler(
            MODEL_DISPLAY_CHANGED, self._model_display_change)
        self._msch = session.triggers.add_handler(
            MODEL_SELECTION_CHANGED, self._model_selection_change)

    # ================================================================
    #  Lifecycle
    # ================================================================

    def delete(self):
        self.session.triggers.remove_handler(self._mdch)
        self.session.triggers.remove_handler(self._msch)
        self._restore_mouse_mode()
        self._deactivate_dust()
        for m in (self._cube_model, self._cylinder_model,
                  self._custom_shape_model, self._dust_highlight_model):
            if m and not m.deleted:
                self.session.models.close([m])
        self._cube_model = None
        self._cylinder_model = None
        self._custom_shape_model = None
        self._dust_highlight_model = None
        self._dust_preview_data = None
        self._mask_array = None
        self._mask_xyz_to_ijk = None
        self._mask_matrix_id = None
        self._mask_volume = None
        self._custom_contour_cache = {'level': None, 'matrix_id': None,
                                      'verts': None, 'norms': None, 'tris': None}
        ToolInstance.delete(self)

    @classmethod
    def get_singleton(cls, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, MapShapeEraserSettings,
                                   "Cube'n Tube", create=create)

    def show(self):
        self.tool_window.shown = True
        self._activate_mouse_mode()

    def hide(self):
        self.tool_window.shown = False
        self._restore_mouse_mode()

    # ================================================================
    #  Active shape helpers
    # ================================================================

    @property
    def _active_index(self):
        return self._shape_combo.currentIndex()

    @property
    def _active_model(self):
        idx = self._active_index
        if idx == _CUBE:
            return self.cube_model
        elif idx == _CYLINDER:
            return self.cylinder_model
        elif idx == _CUSTOM:
            return self._custom_shape_model
        return None

    # ================================================================
    #  Shape combo switching
    # ================================================================

    def _shape_changed(self, idx):
        prev_dust = self._dust_active
        if prev_dust and idx != _DUST:
            self._deactivate_dust()

        self._stack.setCurrentIndex(idx)

        if idx == _DUST:
            self._activate_dust()
            self._erase_inside_btn.setText('Erase dust')
            self._erase_outside_btn.setVisible(False)
            self._crop_btn.setVisible(False)
            self._show_cb.setVisible(False)
            self._color_btn.setVisible(False)
        else:
            self._erase_inside_btn.setText('Erase inside')
            self._erase_outside_btn.setVisible(True)
            self._crop_btn.setVisible(True)
            self._show_cb.setVisible(True)
            self._color_btn.setVisible(True)
            colors = [self._cube_color, self._cyl_color, self._custom_color]
            if idx < len(colors):
                self._color_btn.color = colors[idx]

        show = self._show_cb.isChecked()
        for i, m in enumerate((self._cube_model, self._cylinder_model,
                               self._custom_shape_model)):
            if m and not m.deleted:
                m.display = show and (i == idx)

    # ================================================================
    #  Show / Color (shared)
    # ================================================================

    def _show_eraser_cb(self, state):
        if self._active_index == _DUST:
            return
        m = self._active_model
        if m and not m.deleted:
            m.display = bool(state)

    def _change_color_cb(self, color):
        idx = self._active_index
        if idx == _DUST:
            return
        if idx == _CUBE:
            self._cube_color = tuple(color)
        elif idx == _CYLINDER:
            self._cyl_color = tuple(color)
        elif idx == _CUSTOM:
            self._custom_color = tuple(color)
        m = self._active_model
        if m and not m.deleted:
            m.color = color

    # ================================================================
    #  Model accessors (with lazy re-creation)
    # ================================================================

    @property
    def cube_model(self):
        cm = self._cube_model
        if cm is None or cm.deleted:
            from .cube_eraser import CubeModel, CUBE_ERASER_COLOR
            b = self.session.main_view.drawing_bounds()
            c = b.center() if b else (0, 0, 0)
            cm = CubeModel('eraser cube', self.session, CUBE_ERASER_COLOR,
                           c, self._cube_size_val('x'),
                           self._cube_size_val('y'),
                           self._cube_size_val('z'))
            self._cube_model = cm
            if self._active_index != _CUBE:
                cm.display = False
        return cm

    @property
    def cylinder_model(self):
        cm = self._cylinder_model
        if cm is None or cm.deleted:
            from .cylinder_eraser import CylinderModel, CYLINDER_ERASER_COLOR
            b = self.session.main_view.drawing_bounds()
            c = b.center() if b else (0, 0, 0)
            cm = CylinderModel('eraser cylinder', self.session,
                               CYLINDER_ERASER_COLOR, c,
                               self._cyl_val('rt'), self._cyl_val('rb'),
                               self._cyl_val('ln'))
            self._cylinder_model = cm
            if self._active_index != _CYLINDER:
                cm.display = False
        return cm

    @property
    def custom_shape_model(self):
        return self._custom_shape_model

    # ================================================================
    #  Mouse-mode interface (used by unified MapShapeEraser mode)
    # ================================================================

    @property
    def cube_center(self):
        return self.cube_model.scene_position.origin()

    def _move_model(self, model, delta_xyz):
        if model is None or model.deleted:
            return
        dxyz = model.scene_position.inverse().transform_vector(delta_xyz)
        from chimerax.geometry import translation
        model.position = model.position * translation(dxyz)

    def move_cube(self, delta_xyz):
        self._move_model(self.cube_model, delta_xyz)

    @property
    def cylinder_center(self):
        return self.cylinder_model.scene_position.origin()

    def move_cylinder(self, delta_xyz):
        self._move_model(self.cylinder_model, delta_xyz)

    def move_shape(self, delta_xyz):
        self._move_model(self._custom_shape_model, delta_xyz)

    def move_active_shape(self, delta_xyz):
        idx = self._active_index
        if idx == _CUBE:
            self.move_cube(delta_xyz)
        elif idx == _CYLINDER:
            self.move_cylinder(delta_xyz)
        elif idx == _CUSTOM:
            self.move_shape(delta_xyz)

    def active_center(self):
        idx = self._active_index
        if idx == _CUBE:
            return self.cube_center
        if idx == _CYLINDER:
            return self.cylinder_center
        if idx == _CUSTOM:
            sm = self._custom_shape_model
            if sm is None or sm.deleted:
                return None
            return sm.scene_position.origin()
        return None

    def _activate_mouse_mode(self):
        mm = self.session.ui.mouse_modes
        shape_mode = mm.named_mode(MapShapeEraser.name)
        if shape_mode is None:
            return
        current = mm.mode(button='right', modifiers=[], exact=True)
        if current is shape_mode:
            return
        self._prev_right_mode = current
        mm.bind_mouse_mode(mouse_button='right', mouse_modifiers=[],
                           mode=shape_mode)

    def _restore_mouse_mode(self):
        mm = self.session.ui.mouse_modes
        shape_mode = mm.named_mode(MapShapeEraser.name)
        if shape_mode is None:
            self._prev_right_mode = None
            return
        current = mm.mode(button='right', modifiers=[], exact=True)
        if current is shape_mode:
            restore = self._prev_right_mode
            if restore is None:
                restore = mm.named_mode('translate')
            mm.bind_mouse_mode(mouse_button='right', mouse_modifiers=[],
                               mode=restore)
        self._prev_right_mode = None

    # ================================================================
    #  Cube slider callbacks
    # ================================================================

    def _cube_size_val(self, axis):
        entry = getattr(self, '_cube_size_%s_entry' % axis)
        try:
            return float(entry.text())
        except ValueError:
            return 10

    def _cube_set_all(self, s):
        self._block_text_update = True
        sval = int((s / self._max_slider_size) * self._max_slider_value)
        for a in ('x', 'y', 'z'):
            getattr(self, '_cube_size_%s_entry' % a).setText('%.4g' % s)
            getattr(self, '_cube_size_%s_slider' % a).setValue(sval)
        self._block_text_update = False
        cm = self.cube_model
        cm.size_x = s
        cm.size_y = s
        cm.size_z = s

    def _cube_size_text(self, axis):
        if self._block_text_update:
            return
        s = self._cube_size_val(axis)
        if self._lock_dimensions:
            self._cube_set_all(s)
            return
        setattr(self.cube_model, 'size_' + axis, s)
        self._block_text_update = True
        getattr(self, '_cube_size_%s_slider' % axis).setValue(
            int((s / self._max_slider_size) * self._max_slider_value))
        self._block_text_update = False

    def _cube_size_slide(self, axis, val):
        if self._block_text_update:
            return
        s = (val / self._max_slider_value) * self._max_slider_size
        if self._lock_dimensions:
            self._cube_set_all(s)
            return
        getattr(self, '_cube_size_%s_entry' % axis).setText('%.4g' % s)
        setattr(self.cube_model, 'size_' + axis, s)

    def _cube_size_x_text(self):
        self._cube_size_text('x')

    def _cube_size_y_text(self):
        self._cube_size_text('y')

    def _cube_size_z_text(self):
        self._cube_size_text('z')

    def _cube_size_x_slide(self, val):
        self._cube_size_slide('x', val)

    def _cube_size_y_slide(self, val):
        self._cube_size_slide('y', val)

    def _cube_size_z_slide(self, val):
        self._cube_size_slide('z', val)

    def _cube_lock_toggled(self, state):
        self._lock_dimensions = bool(state)
        if self._lock_dimensions:
            self._cube_set_all(self._cube_size_val('x'))

    # ================================================================
    #  Cylinder slider callbacks
    # ================================================================

    def _cyl_val(self, attr):
        entry = getattr(self, '_cyl_%s_entry' % attr)
        try:
            return float(entry.text())
        except ValueError:
            return 10

    def _cyl_set_both_radii(self, r):
        self._block_text_update = True
        sval = int((r / self._max_slider_size) * self._max_slider_value)
        self._cyl_rt_entry.setText('%.4g' % r)
        self._cyl_rt_slider.setValue(sval)
        self._cyl_rb_entry.setText('%.4g' % r)
        self._cyl_rb_slider.setValue(sval)
        self._block_text_update = False
        cm = self.cylinder_model
        cm.radius_top = r
        cm.radius_bottom = r

    def _cyl_rt_text(self):
        if self._block_text_update:
            return
        r = self._cyl_val('rt')
        if self._lock_radii:
            self._cyl_set_both_radii(r); return
        self.cylinder_model.radius_top = r
        self._block_text_update = True
        self._cyl_rt_slider.setValue(
            int((r / self._max_slider_size) * self._max_slider_value))
        self._block_text_update = False

    def _cyl_rb_text(self):
        if self._block_text_update:
            return
        r = self._cyl_val('rb')
        if self._lock_radii:
            self._cyl_set_both_radii(r); return
        self.cylinder_model.radius_bottom = r
        self._block_text_update = True
        self._cyl_rb_slider.setValue(
            int((r / self._max_slider_size) * self._max_slider_value))
        self._block_text_update = False

    def _cyl_ln_text(self):
        if self._block_text_update:
            return
        L = self._cyl_val('ln')
        self.cylinder_model.length = L
        self._block_text_update = True
        self._cyl_ln_slider.setValue(
            int((L / self._max_slider_size) * self._max_slider_value))
        self._block_text_update = False

    def _cyl_rt_slide(self, val):
        if self._block_text_update:
            return
        r = (val / self._max_slider_value) * self._max_slider_size
        if self._lock_radii:
            self._cyl_set_both_radii(r); return
        self._cyl_rt_entry.setText('%.4g' % r)
        self.cylinder_model.radius_top = r

    def _cyl_rb_slide(self, val):
        if self._block_text_update:
            return
        r = (val / self._max_slider_value) * self._max_slider_size
        if self._lock_radii:
            self._cyl_set_both_radii(r); return
        self._cyl_rb_entry.setText('%.4g' % r)
        self.cylinder_model.radius_bottom = r

    def _cyl_ln_slide(self, val):
        if self._block_text_update:
            return
        L = (val / self._max_slider_value) * self._max_slider_size
        self._cyl_ln_entry.setText('%.4g' % L)
        self.cylinder_model.length = L

    def _cyl_lock_toggled(self, state):
        self._lock_radii = bool(state)
        if self._lock_radii:
            self._cyl_set_both_radii(self._cyl_val('rt'))

    # ================================================================
    #  Custom eraser callbacks
    # ================================================================

    def _clear_custom_contour_cache(self):
        self._custom_contour_cache = {'level': None, 'matrix_id': None,
                                      'verts': None, 'norms': None, 'tris': None}

    def _custom_contour_at_level(self, volume, level):
        from .custom_eraser import contour_from_array
        matrix_id = getattr(volume, '_matrix_id', None)
        c = self._custom_contour_cache
        if c['matrix_id'] == matrix_id and c['level'] == level:
            return c['verts'], c['norms'], c['tris']
        verts, norms, tris = contour_from_array(
            volume.matrix(), level, volume.matrix_indices_to_xyz_transform())
        c['matrix_id'] = matrix_id
        c['level'] = level
        c['verts'] = verts
        c['norms'] = norms
        c['tris'] = tris
        self._mask_matrix_id = matrix_id
        return verts, norms, tris

    def _set_eraser_cb(self):
        vol = self._volume_menu.value
        if vol is None:
            self.session.logger.warning('No volume selected.')
            return
        from .custom_eraser import CustomShapeModel
        threshold = vol.minimum_surface_level
        if threshold is None:
            self.session.logger.warning(
                'Selected volume has no displayed isosurface. '
                'Display an isosurface first (e.g. volume #%s level <value>).'
                % vol.id_string)
            return

        self._mask_array = vol.data.full_matrix().copy()
        self._mask_xyz_to_ijk = vol.data.xyz_to_ijk_transform
        self._clear_custom_contour_cache()
        verts, norms, tris = self._custom_contour_at_level(vol, threshold)
        if verts is None or tris is None or len(tris) == 0:
            self.session.logger.warning(
                'Could not contour selected volume at level %.4g.' % threshold)
            return
        self._threshold = threshold
        self._mask_volume = vol

        data = self._mask_array
        self._threshold_min = float(data.min())
        self._threshold_max = float(data.max())

        old = self._custom_shape_model
        if old and not old.deleted:
            self.session.models.close([old])

        centroid = verts.mean(axis=0)
        centered_verts = (verts - centroid.astype(verts.dtype, copy=False)) * np.float32(1.03)
        from chimerax.geometry import translation
        eraser_pos = vol.scene_position * translation(centroid)

        sm = CustomShapeModel('custom eraser', self.session,
                              self._custom_color, eraser_pos,
                              centered_verts, norms, tris, centroid)
        self._custom_shape_model = sm

        self._block_text_update = True
        self._scale_entry.setText('1.0')
        self._set_scale_slider(1.0)
        self._threshold_entry.setText('%.4g' % threshold)
        self._set_threshold_slider(threshold)
        self._block_text_update = False

        from chimerax.core.commands import run
        run(self.session, 'hide #%s models' % vol.id_string)

        self.session.logger.info(
            'Custom eraser set from #%s at level %.4g (%d triangles)'
            % (vol.id_string, threshold, len(tris)))

    def _set_scale_slider(self, s):
        frac = ((s - self._min_slider_scale)
                / (self._max_slider_scale - self._min_slider_scale))
        self._scale_slider.setValue(int(frac * self._max_slider_value))

    def _scale_text_cb(self):
        if self._block_text_update:
            return
        try:
            s = max(self._min_slider_scale,
                    min(float(self._scale_entry.text()),
                        self._max_slider_scale))
        except ValueError:
            s = 1.0
        self._block_text_update = True
        self._set_scale_slider(s)
        self._block_text_update = False
        sm = self._custom_shape_model
        if sm and not sm.deleted:
            sm.scale = s

    def _scale_slide_cb(self, val):
        if self._block_text_update:
            return
        frac = val / self._max_slider_value
        s = (self._min_slider_scale
             + frac * (self._max_slider_scale - self._min_slider_scale))
        self._scale_entry.setText('%.3g' % s)
        sm = self._custom_shape_model
        if sm and not sm.deleted:
            sm.scale = s

    # ================================================================
    #  Custom eraser threshold callbacks
    # ================================================================

    def _set_threshold_slider(self, t):
        rng = self._threshold_max - self._threshold_min
        if rng == 0:
            self._threshold_slider.setValue(0)
            return
        frac = (t - self._threshold_min) / rng
        frac = max(0.0, min(1.0, frac))
        self._threshold_slider.setValue(int(frac * self._max_slider_value))

    def _threshold_text_cb(self):
        if self._block_text_update:
            return
        try:
            t = float(self._threshold_entry.text())
            t = max(self._threshold_min, min(t, self._threshold_max))
        except ValueError:
            return
        self._block_text_update = True
        self._set_threshold_slider(t)
        self._block_text_update = False
        self._update_custom_threshold(t)

    def _threshold_slide_cb(self, val):
        if self._block_text_update:
            return
        frac = val / self._max_slider_value
        t = self._threshold_min + frac * (self._threshold_max - self._threshold_min)
        self._threshold_entry.setText('%.4g' % t)
        self._update_custom_threshold(t)

    def _update_custom_threshold(self, new_threshold):
        sm = self._custom_shape_model
        if sm is None or sm.deleted:
            return
        mv = self._mask_volume
        if self._mask_array is None or mv is None or mv.deleted:
            return

        current_matrix_id = getattr(mv, '_matrix_id', None)
        if self._mask_matrix_id != current_matrix_id:
            self._clear_custom_contour_cache()
        verts, norms, tris = self._custom_contour_at_level(mv, new_threshold)
        if verts is not None and len(verts) > 0:
            centered = (verts - sm.centroid.astype(verts.dtype, copy=False)) * np.float32(1.03)
            sm.update_mesh(centered, norms, tris)
            self._threshold = new_threshold

    # ================================================================
    #  Dust eraser callbacks
    # ================================================================

    def _dust_volume(self):
        """Return the volume selected in the dust dropdown."""
        v = self._dust_volume_menu.value
        if v is None or v.deleted:
            return None
        return v

    def _update_dust_slider_range(self):
        v = self._dust_volume()
        if v is None:
            return
        min_size = min(v.data.step)
        r = (min_size, 1000 * min_size)
        if r != self._dust_size_range:
            self._dust_size_range = r
            self._dust_slider.set_range(r[0], r[1])

    def _activate_dust(self):
        v = self._dust_volume()
        if v is None:
            v = self._shown_volume()
            if v is not None:
                self._dust_volume_menu.value = v
        self._update_dust_slider_range()
        self._dust_slider.value = 6 * self._dust_size_range[0]
        self._dust_active = True
        self._dust_last_level = None
        self._dust_last_volume = None
        self._apply_dust_hiding()
        if self._dust_frame_handler is None:
            self._dust_frame_handler = self.session.triggers.add_handler(
                'new frame', self._dust_refresh_check)

    def _deactivate_dust(self):
        if not self._dust_active:
            return
        self._dust_active = False
        if self._dust_frame_handler is not None:
            self.session.triggers.remove_handler(self._dust_frame_handler)
            self._dust_frame_handler = None
        self._dust_preview_data = None
        self._close_dust_highlight()

    def _dust_size_changed(self, size, slider_down):
        if self._dust_active:
            self._apply_dust_hiding()

    def _dust_slider_released(self):
        pass

    def _apply_dust_hiding(self):
        v = self._dust_volume()
        if v is None:
            return
        from .dust_eraser import create_dust_highlight
        self._dust_highlight_model, self._dust_preview_data = create_dust_highlight(
            self.session, v, self._dust_slider.value, self._dust_highlight_model)
        self._dust_last_level = v.minimum_surface_level
        self._dust_last_volume = v

    def _dust_refresh_check(self, *_):
        """Per-frame check: refresh highlight if surface level or volume changed."""
        if not self._dust_active:
            return
        v = self._dust_volume()
        if v is None:
            return
        hm = self._dust_highlight_model
        if hm is not None and not hm.deleted:
            vpos = v.scene_position
            if hm.position != vpos:
                hm.position = vpos

        old_v = self._dust_last_volume
        if v is not old_v:
            self._update_dust_slider_range()
            self._dust_preview_data = None
            self._apply_dust_hiding()
        elif v.minimum_surface_level != self._dust_last_level:
            self._dust_preview_data = None
            self._apply_dust_hiding()

    def _close_dust_highlight(self):
        hm = self._dust_highlight_model
        if hm is not None and not hm.deleted:
            self.session.models.close([hm])
        self._dust_highlight_model = None
        self._dust_preview_data = None

    def _erase_dust(self):
        v = self._dust_volume()
        if v is None:
            self.session.logger.warning(
                'No volume selected for dust erase')
            return

        from .dust_eraser import compute_dust_voxel_mask
        mask = compute_dust_voxel_mask(
            v, self._dust_slider.value, self._dust_preview_data)
        if mask is None:
            self.session.logger.info('No dust voxels to erase.')
            return

        vcopy = v.writable_copy()
        grid_data = vcopy.data
        saved = grid_data.full_matrix().copy()
        grid_data.full_matrix()[mask] = 0
        grid_data.values_changed()

        self._register_undo('dust erase', grid_data, saved)

        self._close_dust_highlight()
        self._dust_preview_data = None

        self._dust_volume_menu.value = vcopy
        self._dust_last_volume = vcopy
        self._dust_last_level = None

    # ================================================================
    #  Slider range adjustment
    # ================================================================

    def _model_display_change(self, name, data):
        v = self._shown_volume()
        if v:
            self._adjust_slider_range(v)

    def _model_selection_change(self, name, data):
        hm = self._dust_highlight_model
        if hm is not None and not hm.deleted and hm.selected:
            hm.selected = False

    def _adjust_slider_range(self, volume):
        xyz_min, xyz_max = volume.xyz_bounds(subregion='all')
        smax = max([x1 - x0 for x0, x1 in zip(xyz_min, xyz_max)])
        if smax != self._max_slider_size:
            self._max_slider_size = smax

    # ================================================================
    #  Erase / crop (shared)
    # ================================================================

    def _erase(self, outside=False):
        idx = self._active_index
        if idx == _DUST:
            self._erase_dust()
            return
        if idx == _CUSTOM:
            self._erase_custom(outside)
        else:
            self._erase_shape(outside)

    def _erase_shape(self, outside):
        v = self._shown_volume()
        if v is None:
            self.session.logger.warning(
                'No single displayed volume for erase')
            return
        vcopy = v.writable_copy()
        grid_data = vcopy.data
        saved = grid_data.full_matrix().copy()

        idx = self._active_index
        changed = False
        if idx == _CUBE:
            from .cube_eraser import _erase_with_cube_model
            changed = _erase_with_cube_model(grid_data, self.cube_model,
                                             v.scene_position, value=0,
                                             outside=outside)
        elif idx == _CYLINDER:
            from .cylinder_eraser import _erase_with_cylinder_model
            changed = _erase_with_cylinder_model(grid_data, self.cylinder_model,
                                                 v.scene_position, value=0,
                                                 outside=outside)

        if not changed:
            return
        self._register_undo('shape erase', grid_data, saved)

    def _erase_custom(self, outside):
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

        from .custom_eraser import _erase_with_custom_shape
        changed = _erase_with_custom_shape(
            grid_data, sm, v.scene_position,
            self._mask_array, self._mask_xyz_to_ijk,
            sm.centroid, sm.scale,
            self._threshold,
            value=0, outside=outside)

        if not changed:
            return
        self._register_undo('custom erase', grid_data, saved)

    def _register_undo(self, name, grid_data, saved):
        if self._last_undo_action is not None:
            try:
                self.session.undo.deregister(self._last_undo_action,
                                             delete_history=False)
            except Exception:
                pass
        action = VolumeEraseUndo(name, grid_data, saved)
        self._last_undo_action = action
        self.session.undo.register(action)

    def _crop_map(self):
        v = self._shown_volume()
        if v is None:
            self.session.logger.warning(
                'No single displayed volume for crop')
            return
        idx = self._active_index
        if idx == _CUBE:
            from .cube_eraser import _cube_grid_bounds
            ijk_min, ijk_max = _cube_grid_bounds(
                v.data, self.cube_model, v.scene_position)
        elif idx == _CYLINDER:
            from .cylinder_eraser import _cylinder_grid_bounds
            ijk_min, ijk_max = _cylinder_grid_bounds(
                v.data, self.cylinder_model, v.scene_position)
        elif idx == _CUSTOM:
            sm = self._custom_shape_model
            if sm is None or sm.deleted:
                self.session.logger.warning('No custom eraser shape set.')
                return
            from .custom_eraser import _custom_grid_bounds
            ijk_min, ijk_max = _custom_grid_bounds(
                v.data, sm, v.scene_position)
        elif idx == _DUST:
            self.session.logger.warning(
                'Crop is not applicable for dust mode.')
            return
        else:
            return
        region = ','.join(['%d,%d,%d' % tuple(ijk_min),
                           '%d,%d,%d' % tuple(ijk_max)])
        cmd = 'volume copy #%s subregion %s' % (v.id_string, region)
        from chimerax.core.commands import run
        run(self.session, cmd)

    def _shown_volume(self):
        from chimerax.map import Volume
        exclude = {self._custom_shape_model, self._dust_highlight_model}
        only_visible = None
        for m in self.session.models.list(type=Volume):
            if not m.visible or m in exclude:
                continue
            if only_visible is not None:
                return None
            only_visible = m
        return only_visible


# -------------------------------------------------------------------------
def map_shape_eraser_panel(session, create=True):
    return MapShapeEraserSettings.get_singleton(session, create=create)


# -------------------------------------------------------------------------
from chimerax.mouse_modes import MouseMode


class MapShapeEraser(MouseMode):
    """Single mouse mode that moves the currently selected CubeNTube shape."""

    name = "cube'n tube"
    icon_file = 'cubentube.png'

    def __init__(self, session):
        MouseMode.__init__(self, session)

    @property
    def settings(self):
        return map_shape_eraser_panel(self.session)

    def enable(self):
        sp = map_shape_eraser_panel(self.session)
        if sp is not None:
            sp.show()

    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)

    def mouse_drag(self, event):
        settings = self.settings
        c = settings.active_center()
        if c is None:
            return
        dx, dy = self.mouse_motion(event)
        v = self.session.main_view
        s = v.pixel_size(c)
        if event.shift_down():
            shift = (0, 0, s * dy)
        else:
            shift = (s * dx, -s * dy, 0)
        dxyz = v.camera.position.transform_vector(shift)
        settings.move_active_shape(dxyz)

    def mouse_up(self, event):
        MouseMode.mouse_up(self, event)

    def vr_motion(self, event):
        settings = self.settings
        c = settings.active_center()
        if c is None:
            return
        delta_xyz = event.motion * c - c
        settings.move_active_shape(delta_xyz)


# -------------------------------------------------------------------------
def register_mousemode(session):
    mm = session.ui.mouse_modes
    mm.add_mode(MapShapeEraser(session))
