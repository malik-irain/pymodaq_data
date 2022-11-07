from collections import OrderedDict
from qtpy import QtWidgets
from qtpy.QtCore import QObject, Signal, Slot

from pymodaq.utils.data import DataToExport, DataRaw
from pymodaq.utils.plotting.utils.filter import Filter
from pymodaq.utils import daq_utils as utils
from pymodaq.utils import gui_utils as gutils
from pymodaq.utils.exceptions import ViewerError
import datetime


DATATYPES = {'Viewer0D': 'Data0D', 'Viewer1D': 'Data1D', 'Viewer2D': 'Data2D', 'ViewerND': 'DataND',
             'ViewerSequential': 'DataSequential'}


class ViewerBase(QObject):
    """Base Class for data viewers implementing all common functionalities

    Parameters
    ----------
    parent: QtWidgets.QWidget
    title: str

    Attributes
    ----------
    view: QObject
        Ui interface of the viewer

    data_to_export_signal: Signal[DataToExport]
    ROI_changed: Signal
    crosshair_dragged: Signal[float, float]
    crosshair_clicked: Signal[bool]
    sig_double_clicked: Signal[float, float]
    status_signal: Signal[str]
    """
    data_to_export_signal = Signal(DataToExport)
    _data_to_show_signal = Signal(DataRaw)

    ROI_changed = Signal()
    crosshair_dragged = Signal(float, float)  # Crosshair position in units of scaled top/right axes
    status_signal = Signal(str)
    crosshair_clicked = Signal(bool)
    sig_double_clicked = Signal(float, float)

    def __init__(self, parent=None, title=''):
        super().__init__()
        self.title = title if title != '' else self.__class__.__name__

        self._raw_data = None
        self.data_to_export: DataToExport = DataToExport(name=self.title)
        self.view = None

        if parent is None:
            parent = QtWidgets.QWidget()
            parent.show()
        self.parent = parent

        self._display_temporary = False

    @property
    def viewer_type(self):
        """str: the viewer data type see DATA_TYPES"""
        return DATATYPES[self.__class__.__name__]

    def show_data(self, data: DataRaw, **kwargs):
        """Entrypoint to display data into the viewer

        Parameters
        ----------
        data: data_mod.DataFromPlugins
        """
        if not(len(data.shape) == 2 or len(data.shape) == 1 or len(data.shape) == 0):
            raise ViewerError(f'Ndarray of dim: {len(data.shape)} cannot be plotted'
                              f' using a {self.viewer_type}')
        self.data_to_export = DataToExport(name=self.title)
        self._raw_data = data

        self._display_temporary = False

        self._show_data(data, **kwargs)

    def show_data_temp(self, data: DataRaw, **kwargs):
        """Entrypoint to display temporary data into the viewer

        No processed data signal is emitted from the viewer

        Parameters
        ----------
        data: data_mod.DataFromPlugins
        """
        self._display_temporary = True
        self.show_data(data, **kwargs)

    def _show_data(self, data: DataRaw):
        """Specific viewers should implement it"""
        raise NotImplementedError

    def add_attributes_from_view(self):
        """Convenience function to add attributes from the view to self"""
        for attribute in self.convenience_attributes:
            if hasattr(self.view, attribute):
                setattr(self, attribute, getattr(self.view, attribute))

    def trigger_action(self, action_name: str):
        """Convenience function to trigger programmatically one of the action of the related view"""
        if self.has_action(action_name):
            self.get_action(action_name).trigger()

    def activate_roi(self, activate=True):
        """Activate the Roi manager using the corresponding action"""
        raise NotImplementedError

    def setVisible(self, show=True):
        """convenience method to show or hide the paretn widget"""
        self.parent.setVisible(show)