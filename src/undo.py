# CubeNTube -- shape eraser plugin for UCSF ChimeraX
# Copyright 2026 Tamino Cairoli <tcairoli@ethz.ch>
# SPDX-License-Identifier: LGPL-2.1-or-later

from chimerax.core.undo import UndoAction


class VolumeEraseUndo(UndoAction):
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
