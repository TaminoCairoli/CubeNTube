from chimerax.core.toolshed import BundleAPI


class _CubeNTubeAPI(BundleAPI):

    @staticmethod
    def start_tool(session, tool_name):
        if tool_name == "CubeNTube":
            from .gui_panel import map_shape_eraser_panel
            return map_shape_eraser_panel(session)

    @staticmethod
    def initialize(session, bundle_info):
        """Register mouse modes (same pattern as map_eraser bundle)."""
        if not session.ui.is_gui:
            return
        import traceback
        # Cube eraser
        try:
            from . import cube_eraser
            reg = getattr(cube_eraser, 'register_mousemode', None)
            if reg is not None:
                reg(session)
            else:
                session.logger.warning('CubeNTube: cube_eraser has no register_mousemode.')
        except Exception as e:
            session.logger.warning('CubeNTube: cube eraser registration failed: %s' % e)
            session.logger.info(traceback.format_exc())
        # Cylinder eraser
        try:
            from . import cylinder_eraser
            reg = getattr(cylinder_eraser, 'register_mousemode', None)
            if reg is not None:
                reg(session)
            else:
                session.logger.warning(
                    'CubeNTube: cylinder_eraser has no register_mousemode (old install?).'
                )
        except Exception as e:
            session.logger.warning('CubeNTube: cylinder eraser registration failed: %s' % e)
            session.logger.info(traceback.format_exc())
        # Custom eraser
        try:
            from . import custom_eraser
            reg = getattr(custom_eraser, 'register_mousemode', None)
            if reg is not None:
                reg(session)
            else:
                session.logger.warning(
                    'CubeNTube: custom_eraser has no register_mousemode.')
        except Exception as e:
            session.logger.warning('CubeNTube: custom eraser registration failed: %s' % e)
            session.logger.info(traceback.format_exc())

        # Patch built-in sphere eraser with single-step undo support
        try:
            _patch_sphere_eraser_undo(session)
        except Exception as e:
            session.logger.warning(
                'CubeNTube: sphere eraser undo patch failed: %s' % e)
            session.logger.info(traceback.format_exc())

    @staticmethod
    def finish(session, bundle_info):
        _unpatch_sphere_eraser_undo(session)

    @staticmethod
    def register_command(command_name, logger):
        if command_name == "volume cube erase":
            from . import cube_eraser
            cube_eraser.register_volume_cube_erase_command(logger)
        elif command_name == "volume cylinder erase":
            from . import cylinder_eraser
            cylinder_eraser.register_volume_cylinder_erase_command(logger)


def _patch_sphere_eraser_undo(session):
    """Monkey-patch the built-in sphere Map Eraser to support single-step
    undo/redo.  Replaces MapEraserSettings._erase so that Cmd+Z restores
    volume data after an accidental erase."""
    import numpy as np
    from chimerax.core.undo import UndoAction
    from chimerax.map_eraser.eraser import (
        MapEraserSettings,
        _set_data_in_sphere,
        _set_data_outside_sphere,
    )

    class _SphereEraseUndo(UndoAction):
        def __init__(self, grid_data, saved_matrix):
            super().__init__('sphere erase', can_redo=True)
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

    def _erase_with_undo(self, outside=False):
        v, center, radius = self._eraser_region()
        if v is None:
            return

        vcopy = v.writable_copy()
        grid_data = vcopy.data
        saved = grid_data.full_matrix().copy()

        cvol = v.scene_position.inverse() * center
        if outside:
            _set_data_outside_sphere(grid_data, cvol, radius, 0)
        else:
            _set_data_in_sphere(grid_data, cvol, radius, 0)

        if np.array_equal(saved, grid_data.full_matrix()):
            return

        prev = getattr(self, '_last_undo_action', None)
        if prev is not None:
            try:
                self.session.undo.deregister(prev, delete_history=False)
            except Exception:
                pass
        action = _SphereEraseUndo(grid_data, saved)
        self._last_undo_action = action
        self.session.undo.register(action)

    MapEraserSettings._cubentube_orig_erase = MapEraserSettings._erase
    MapEraserSettings._erase = _erase_with_undo
    session.logger.info('CubeNTube: sphere eraser undo support enabled')


def _unpatch_sphere_eraser_undo(session):
    """Restore the original sphere eraser _erase method on bundle unload."""
    try:
        from chimerax.map_eraser.eraser import MapEraserSettings
        orig = getattr(MapEraserSettings, '_cubentube_orig_erase', None)
        if orig is not None:
            MapEraserSettings._erase = orig
            del MapEraserSettings._cubentube_orig_erase
            session.logger.info('CubeNTube: sphere eraser undo patch removed')
    except Exception:
        pass


bundle_api = _CubeNTubeAPI()
