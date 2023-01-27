# -*- coding: utf-8 -*-
"""
Created the 05/12/2022

@author: Sebastien Weber
"""
from typing import List, Tuple

import numpy as np

from qtpy import QtCore, QtWidgets
from pymodaq.utils.data import Axis, DataDistribution
from pymodaq.utils.logger import set_logger, get_module_name
from pymodaq.utils import math_utils as mutils
from pymodaq.utils import config as configmod
from pymodaq.utils import gui_utils as gutils
from ..scan_factory import ScannerFactory, ScannerBase, ScanParameterManager
from pymodaq.utils.parameter import utils as putils
from pymodaq.utils.parameter.pymodaq_ptypes import TableViewCustom

logger = set_logger(get_module_name(__file__))
config = configmod.Config()


class TableModelTabular(gutils.TableModel):
    """Table Model for the Model/View Qt framework dedicated to the Tabular scan mode"""
    def __init__(self, data, axes_name=None, **kwargs):
        if axes_name is None:
            if 'header' in kwargs:  # when saved as XML the header will be saved and restored here
                axes_name = [h for h in kwargs['header']]
                kwargs.pop('header')
            else:
                raise Exception('Invalid header')

        header = [name for name in axes_name]
        editable = [True for name in axes_name]
        super().__init__(data, header, editable=editable, **kwargs)

    def __len__(self):
        return len(self._data)

    def add_data(self, row, data=None):
        if data is not None:
            self.insert_data(row, [float(d) for d in data])
        else:
            self.insert_data(row, [0. for name in self.header])

    def remove_data(self, row):
        self.remove_row(row)

    def load_txt(self):
        fname = gutils.select_file(start_path=None, save=False, ext='*')
        if fname is not None and fname != '':
            while self.rowCount(self.index(-1, -1)) > 0:
                self.remove_row(0)

            data = np.loadtxt(fname)
            if len(data.shape) == 1:
                data = data.reshape((data.size, 1))
            self.set_data_all(data)

    def save_txt(self):
        fname = gutils.select_file(start_path=None, save=True, ext='dat')
        if fname is not None and fname != '':
            np.savetxt(fname, self.get_data_all(), delimiter='\t')

    def __repr__(self):
        return f'{self.__class__.__name__} from module {self.__class__.__module__}'

    def validate_data(self, row, col, value):
        """
        make sure the values and signs of the start, stop and step values are "correct"
        Parameters
        ----------
        row: (int) row within the table that is to be changed
        col: (int) col within the table that is to be changed
        value: (float) new value for the value defined by row and col

        Returns
        -------
        bool: True is the new value is fine (change some other values if needed) otherwise False
        """

        return True


@ScannerFactory.register('Tabular', 'Linear')
class TabularScanner(ScannerBase):
    params = [
        {'title': 'Positions', 'name': 'tabular_table', 'type': 'table_view', 'delegate': gutils.SpinBoxDelegate,
         'menu': True},
              ]
    distribution = DataDistribution['spread']

    def __init__(self, actuators: List[str]):
        super().__init__(actuators=actuators)
        self.table_model: TableModelTabular = None
        self.table_view: TableViewCustom = None
        self.update_model()

    @property
    def actuators(self):
        return self._actuators

    @actuators.setter
    def actuators(self, actuators_name):
        self._actuators = actuators_name
        self.update_model()

    def update_model(self, init_data=None):
        if init_data is None:
            init_data = [[0. for name in self._actuators]]

        self.table_model = TableModelTabular(init_data, [name for name in self._actuators])
        self.table_view = putils.get_widget_from_tree(self.settings_tree, TableViewCustom)[0]
        self.settings.child('tabular_table').setValue(self.table_model)

        self.update_table_view()

    def update_table_view(self):
        self.table_view.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table_view.horizontalHeader().setStretchLastSection(True)
        self.table_view.setSelectionBehavior(QtWidgets.QTableView.SelectRows)
        self.table_view.setSelectionMode(QtWidgets.QTableView.SingleSelection)
        styledItemDelegate = QtWidgets.QStyledItemDelegate()
        styledItemDelegate.setItemEditorFactory(gutils.SpinBoxDelegate())
        self.table_view.setItemDelegate(styledItemDelegate)

        self.table_view.setDragEnabled(True)
        self.table_view.setDropIndicatorShown(True)
        self.table_view.setAcceptDrops(True)
        self.table_view.viewport().setAcceptDrops(True)
        self.table_view.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.table_view.setDragDropMode(QtWidgets.QTableView.InternalMove)
        self.table_view.setDragDropOverwriteMode(False)

        self.table_view.add_data_signal[int].connect(self.table_model.add_data)
        self.table_view.remove_row_signal[int].connect(self.table_model.remove_data)
        self.table_view.load_data_signal.connect(self.table_model.load_txt)
        self.table_view.save_data_signal.connect(self.table_model.save_txt)

    def evaluate_steps(self):
        return len(self.table_model)

    def set_scan(self):
        positions = np.array(self.table_model.get_data_all())
        self.get_info_from_positions(positions)

    def update_tabular_positions(self, positions: np.ndarray = None):
        """Convenience function to write positions directly into the tabular table

        Parameters
        ----------
        positions: ndarray
            a 2D ndarray with as many columns as selected actuators
        """
        try:
            if positions is None:
                if self.settings.child('tabular_settings',
                                       'tabular_selection').value() == 'Polylines':  # from ROI
                    viewer = self.scan_selector.scan_selector_source

                    if self.settings.child('tabular_settings', 'tabular_subtype').value() == 'Linear':
                        positions = self.scan_selector.scan_selector.getArrayIndexes(
                            spacing=self.settings.child('tabular_settings', 'tabular_step').value())
                    elif self.settings.child('tabular_settings',
                                             'tabular_subtype').value() == 'Adaptive':
                        positions = self.scan_selector.scan_selector.get_vertex()

                    steps_x, steps_y = zip(*positions)
                    steps_x, steps_y = viewer.scale_axis(np.array(steps_x), np.array(steps_y))
                    positions = np.transpose(np.array([steps_x, steps_y]))
                    self.update_model(init_data=positions)
                else:
                    self.update_model()
            elif isinstance(positions, np.ndarray):
                self.update_model(init_data=positions)
            else:
                pass
        except Exception as e:
            logger.exception(str(e))

    def get_nav_axes(self) -> List[Axis]:
        return [Axis(label=f'{act.title}', units=act.units, data=self.positions[:, ind], index=ind)
                for ind, act in enumerate(self.actuators)]

    def get_indexes_from_scan_index(self, scan_index: int) -> Tuple[int]:
        """To be reimplemented. Calculations of indexes within the scan"""
        return scan_index,

    def get_scan_shape(self) -> Tuple[int]:
        return len(self.table_model),
