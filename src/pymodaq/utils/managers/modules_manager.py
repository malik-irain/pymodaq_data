from collections import OrderedDict
from qtpy.QtCore import QObject, Signal, Slot, QThread
from qtpy import QtWidgets
import time

from pymodaq.utils.logger import set_logger, get_module_name, get_module_name
from pymodaq.utils import daq_utils as utils
from pymodaq.utils.config import Config
from pyqtgraph.parametertree import Parameter, ParameterTree
from pymodaq.utils.managers.parameter_manager import ParameterManager

logger = set_logger(get_module_name(__file__))
config = Config()

class ModulesManager(QObject, ParameterManager):
    """Class to manage DAQ_Viewers and DAQ_Moves with UI to select some

    Easier to connect control modules signals to slots, test, ...

    Parameters
    ----------
    detectors: list of DAQ_Viewer
    actuators: list of DAQ_Move
    selected_detectors: list of DAQ_Viewer
        sublist of detectors
    selected_actuators: list of DAQ_Move
        sublist of actuators
    """
    detectors_changed = Signal(list)
    actuators_changed = Signal(list)
    det_done_signal = Signal(OrderedDict)
    move_done_signal = Signal(OrderedDict)
    timeout_signal = Signal(bool)

    params = [
        {'title': 'Actuators/Detectors Selection', 'name': 'modules', 'type': 'group', 'children': [
            {'title': 'detectors', 'name': 'detectors', 'type': 'itemselect'},
            {'title': 'Actuators', 'name': 'actuators', 'type': 'itemselect'},
        ]},

        {'title': "Moves done?", 'name': 'move_done', 'type': 'led', 'value': False},
        {'title': "Detections done?", 'name': 'det_done', 'type': 'led', 'value': False},

        {'title': 'Data dimensions', 'name': 'data_dimensions', 'type': 'group', 'children': [
            {'title': "Probe detector's data", 'name': 'probe_data', 'type': 'action'},
            {'title': 'Data0D list:', 'name': 'det_data_list0D', 'type': 'itemselect'},
            {'title': 'Data1D list:', 'name': 'det_data_list1D', 'type': 'itemselect'},
            {'title': 'Data2D list:', 'name': 'det_data_list2D', 'type': 'itemselect'},
            {'title': 'DataND list:', 'name': 'det_data_listND', 'type': 'itemselect'},
        ]},
        {'title': 'Actuators positions', 'name': 'actuators_positions', 'type': 'group', 'children': [
            {'title': "Test actuators", 'name': 'test_actuator', 'type': 'action'},
            {'title': 'Positions:', 'name': 'positions_list', 'type': 'itemselect'},
        ]},
    ]

    def __init__(self, detectors=[], actuators=[], selected_detectors=[], selected_actuators=[], **kwargs):
        QObject.__init__(self)
        ParameterManager.__init__(self)

        for mod in selected_actuators:
            assert mod in actuators
        for mod in selected_detectors:
            assert mod in detectors

        self.actuator_timeout = config('actuator', 'timeout')
        self.detector_timeout = config('viewer', 'timeout')

        self.det_done_datas = OrderedDict()
        self.det_done_flag = False
        self.move_done_positions = OrderedDict()
        self.move_done_flag = False

        self.settings.child('data_dimensions', 'probe_data').sigActivated.connect(self.get_det_data_list)
        self.settings.child('actuators_positions', 'test_actuator').sigActivated.connect(self.test_move_actuators)

        self._detectors = []
        self._actuators = []

        self.grab_done_signals = []
        self.det_commands_signal = []

        self.actuators_connected = False
        self.detectors_connected = False

        self.set_actuators(actuators, selected_actuators)
        self.set_detectors(detectors, selected_detectors)

    @classmethod
    def get_names(cls, modules):
        """Get the titles of a list of Control Modules

        Parameters
        ----------
        modules: list of DAQ_Move and/or DAQ_Viewer
        """
        if not hasattr(modules, '__iter__'):
            modules = [modules]
        return [mod.title for mod in modules]

    def get_mods_from_names(self, names, mod='det'):
        """Getter of a list of given modules from their name (title)

        Parameters
        ----------
        names: list of str
        mod: str
            either 'det' for DAQ_Viewer modules or 'act' for DAQ_Move modules
        """
        mods = []
        for name in names:
            d = self.get_mod_from_name(name, mod)
            if d is not None:
                mods.append(d)
        return mods

    def get_mod_from_name(self, name, mod='det'):
        """Getter of a given module from its name (title)

        Parameters
        ----------
        name: str
        mod: str
            either 'det' for DAQ_Viewer modules or 'act' for DAQ_Move modules
        """
        if mod == 'det':
            modules = self._detectors
        else:
            modules = self._actuators

        if name in self.get_names(modules):
            return modules[self.get_names(modules).index(name)]
        else:
            logger.warning(f'No detector with this name: {name}')
            return None

    def set_actuators(self, actuators, selected_actuators):
        """Populates actuators and the subset to be selected in the UI"""
        self._actuators = actuators
        self.settings.child('modules', 'actuators').setValue(dict(all_items=self.get_names(actuators),
                                                                  selected=self.get_names(selected_actuators)))

    def set_detectors(self, detectors, selected_detectors):
        """Populates detectors and the subset to be selected in the UI"""
        self._detectors = detectors
        self.settings.child('modules', 'detectors').setValue(dict(all_items=self.get_names(detectors),
                                                                  selected=self.get_names(selected_detectors)))

    @property
    def detectors(self):
        """Get the list of selected detectors"""
        return self.get_mods_from_names(self.selected_detectors_name)

    @property
    def detectors_all(self):
        """Get the list of all detectors"""
        return self._detectors

    @property
    def actuators(self):
        """Get the list of selected actuators"""
        return self.get_mods_from_names(self.selected_actuators_name, mod='act')

    @property
    def actuators_all(self):
        """Get the list of all actuators"""
        return self._actuators

    @property
    def modules(self):
        """Get the list of all detectors and actuators"""
        return self.detectors_all + self.actuators_all

    @property
    def Ndetectors(self):
        """Get the number of selected detectors"""
        return len(self.detectors)

    @property
    def Nactuators(self):
        """Get the number of selected actuators"""
        return len(self.actuators)

    @property
    def detectors_name(self):
        """Get all the names of the detectors"""
        return self.settings.child('modules', 'detectors').value()['all_items']

    @property
    def selected_detectors_name(self):
        """Get/Set the names of the selected detectors"""
        return self.settings.child('modules', 'detectors').value()['selected']

    @selected_detectors_name.setter
    def selected_detectors_name(self, detectors):
        if set(detectors).issubset(self.detectors_name):
            self.settings.child('modules', 'detectors').setValue(dict(all_items=self.detectors_name,
                                                                      selected=detectors))

    @property
    def actuators_name(self):
        """Get all the names of the actuators"""
        return self.settings.child('modules', 'actuators').value()['all_items']

    @property
    def selected_actuators_name(self):
        """Get/Set the names of the selected actuators"""
        return self.settings.child('modules', 'actuators').value()['selected']

    @selected_actuators_name.setter
    def selected_actuators_name(self, actuators):
        if set(actuators).issubset(self.actuators_name):
            self.settings.child('modules', 'actuators').setValue(dict(all_items=self.actuators_name,
                                                                      selected=actuators))

    def value_changed(self, param):
        if param.name() == 'detectors':
            self.detectors_changed.emit(param.value()['selected'])

        elif param.name() == 'actuators':
            self.actuators_changed.emit(param.value()['selected'])

    def get_det_data_list(self):
        """Do a snap of selected detectors, to get the list of all the data and processed data"""

        self.connect_detectors()
        datas = self.grab_datas()

        data_list0D = []
        data_list1D = []
        data_list2D = []
        data_listND = []

        for k in datas.keys():
            if 'data0D' in datas[k].keys():
                data_list0D.extend([f'{k}/{name}' for name in datas[k]['data0D'].keys()])
            if 'data1D' in datas[k].keys():
                data_list1D.extend([f'{k}/{name}' for name in datas[k]['data1D'].keys()])
            if 'data2D' in datas[k].keys():
                data_list2D.extend([f'{k}/{name}' for name in datas[k]['data2D'].keys()])
            if 'dataND' in datas[k].keys():
                data_listND.extend([f'{k}/{name}' for name in datas[k]['dataND'].keys()])

        self.settings.child('data_dimensions', 'det_data_list0D').setValue(
            dict(all_items=data_list0D, selected=[]))
        self.settings.child('data_dimensions', 'det_data_list1D').setValue(
            dict(all_items=data_list1D, selected=[]))
        self.settings.child('data_dimensions', 'det_data_list2D').setValue(
            dict(all_items=data_list2D, selected=[]))
        self.settings.child('data_dimensions', 'det_data_listND').setValue(
            dict(all_items=data_listND, selected=[]))

        self.connect_detectors(False)

    def get_selected_probed_data(self, dim='0D'):
        """Get the name of selected data names of a given dimensionality

        Parameters
        ----------
        dim: str
            either '0D', '1D', '2D' or 'ND'
        """
        return self.settings.child('data_dimensions', f'det_data_list{dim.upper()}').value()['selected']

    def grab_datas(self, **kwargs):
        """Do a single grab of connected and selected detectors"""
        self.det_done_datas = OrderedDict()
        self.det_done_flag = False
        self.settings.child('det_done').setValue(self.det_done_flag)
        tzero = time.perf_counter()

        for sig in [mod.command_hardware for mod in self.detectors]:
            sig.emit(utils.ThreadCommand("single", [1, kwargs]))

        while not self.det_done_flag:
            # wait for grab done signals to end

            QtWidgets.QApplication.processEvents()
            if time.perf_counter() - tzero > self.detector_timeout:
                self.timeout_signal.emit(True)
                logger.error('Timeout Fired during waiting for data to be acquired')
                break

        self.det_done_signal.emit(self.det_done_datas)
        return self.det_done_datas

    def connect_actuators(self, connect=True, slot=None, signal='move_done'):
        """Connect the selected actuators signal to a given or default slot

        Parameters
        ----------
        connect: bool
        slot: builtin_function_or_method
            method or function the chosen signal will be connected to
            if None, then the default move_done slot is used
        signal: str
            What kind of signal is to be used:

            * 'move_done' will connect the `move_done signal` to the slot
            * 'current_value' will connect the 'current_signal' to the slot

        See Also
        --------
        :meth:`move_done`
        """
        if slot is None:
            slot = self.move_done
        if connect:
            for sig in [mod.move_done_signal if signal == 'move_done' else mod.current_value_signal
                        for mod in self.actuators]:
                sig.connect(slot)

        else:
            try:
                for sig in [mod.move_done_signal if signal == 'move_done' else mod.current_value_signal
                            for mod in self.actuators]:
                    sig.disconnect(slot)
            except Exception as e:
                logger.error(str(e))

        self.actuators_connected = connect

    def connect_detectors(self, connect=True, slot=None):
        """
        Connect selected DAQ_Viewers's grab_done_signal to the given slot

        Parameters
        ----------
        connect: bool
            if True, connect to the given slot (or default slot)
            if False, disconnect all detectors (not only the currently selected ones.
            This is made because when selected detectors changed if you only disconnect those one,
            the previously connected ones will stay connected)
        slot: method
            A method that should be connected, if None self.det_done is connected by default
        """

        if slot is None:
            slot = self.det_done

        if connect:
            for sig in [mod.grab_done_signal for mod in self.detectors]:
                sig.connect(slot)
        else:

            for sig in [mod.grab_done_signal for mod in self.detectors_all]:
                try:
                    sig.disconnect(slot)
                except TypeError as e:
                    # means the slot was not previously connected
                    logger.info(str(e))

        self.detectors_connected = connect

    def test_move_actuators(self):
        """Do a move of selected actuator"""
        positions = dict()
        for act in self.get_names(self.actuators):
            pos, done = QtWidgets.QInputDialog.getDouble(None, f'Enter a target position for actuator {act}',
                                                         'Position:')
            if not done:
                pos = 0.
            positions[act] = pos

        self.connect_actuators()

        positions = self.move_actuators(positions)

        self.settings.child('actuators_positions',
                            'positions_list').setValue(dict(all_items=[f'{k}: {positions[k]}' for k in positions],
                                                            selected=[]))

        self.connect_actuators(False)

    def move_actuators(self, positions, mode='abs', polling=True):
        """will apply positions to each currently selected actuators. By Default the mode is absolute but can be

        Parameters
        ----------
        positions: list
            the list of position to apply. Its length must be equal to the number of selected actutors
        mode: str
            either 'abs' for absolute positionning or 'rel' for relative
        polling: bool
            if True will wait for the selected actuators to reach their target positions (they have to be
            connected to a method checking for the position and letting the programm know the move is done (default
            connection is this object `move_done` method)

        Returns
        -------
        (OrderedDict) with the selected actuators's name as key and current actuators's value as value
        """
        self.move_done_positions = OrderedDict()
        self.move_done_flag = False
        self.settings.child(('move_done')).setValue(self.move_done_flag)

        if mode == 'abs':
            command = 'move_abs'
        elif mode == 'rel':
            command = 'move_rel'
        else:
            logger.error(f'Invalid positioning mode: {mode}')
            return self.move_done_positions

        if not hasattr(positions, '__iter__'):
            positions = [positions]

        if len(positions) == self.Nactuators:
            if isinstance(positions, dict):
                for k in positions:
                    act = self.get_mod_from_name(k, 'act')
                    if act is not None:
                        act.command_hardware.emit(
                            utils.ThreadCommand(command=command, attribute=[positions[k], polling]))
            else:
                for ind, act in enumerate(self.actuators):
                    act.command_hardware.emit(utils.ThreadCommand(command=command, attribute=[positions[ind], polling]))

        else:
            logger.error('Invalid number of positions compared to selected actuators')
            return self.move_done_positions

        tzero = time.perf_counter()
        if polling:
            while not self.move_done_flag:  # polling move done

                QtWidgets.QApplication.processEvents()
                if time.perf_counter() - tzero > self.actuator_timeout:
                    self.timeout_signal.emit(True)
                    logger.error('Timeout Fired during waiting for data to be acquired')
                    break
                QThread.msleep(20)

        self.move_done_signal.emit(self.move_done_positions)
        return self.move_done_positions

    def order_positions(self, positions_as_dict):
        actuators = self.selected_actuators_name
        pos = []
        for act in actuators:
            pos.append(positions_as_dict[act])
        return pos

    @Slot(str, float)
    def move_done(self, name, position):
        try:
            if name not in list(self.move_done_positions.keys()):
                self.move_done_positions[name] = position

            if len(self.move_done_positions.items()) == len(self.actuators):
                self.move_done_flag = True
                self.settings.child('move_done').setValue(self.move_done_flag)
        except Exception as e:
            logger.exception(str(e))

    @Slot(OrderedDict)
    def det_done(self, data):
        try:
            if data['name'] not in list(self.det_done_datas.keys()):
                self.det_done_datas[data['name']] = data
            if len(self.det_done_datas.items()) == len(self.detectors):
                self.det_done_flag = True
                self.settings.child('det_done').setValue(self.det_done_flag)
        except Exception as e:
            logger.exception(str(e))


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    from qtpy.QtCore import QThread
    from pymodaq.utils.gui_utils import DockArea
    from pyqtgraph.dockarea import Dock
    from pymodaq.control_modules.daq_viewer import DAQ_Viewer
    from pymodaq.control_modules.daq_move import DAQ_Move

    win = QtWidgets.QMainWindow()
    area = DockArea()
    win.setCentralWidget(area)
    win.resize(1000, 500)
    win.setWindowTitle('pymodaq main')

    prog = DAQ_Viewer(area, title="Testing2D", DAQ_type='DAQ2D')
    prog2 = DAQ_Viewer(area, title="Testing1D", DAQ_type='DAQ1D')
    prog3 = DAQ_Viewer(area, title="Testing0D", DAQ_type='DAQ0D')

    act1_widget = QtWidgets.QWidget()
    act2_widget = QtWidgets.QWidget()
    act1 = DAQ_Move(act1_widget, title='X_axis')
    act2 = DAQ_Move(act2_widget, title='Y_axis')

    QThread.msleep(1000)
    prog.ui.IniDet_pb.click()
    prog2.ui.IniDet_pb.click()
    prog3.ui.IniDet_pb.click()

    dock1 = Dock('actuator 1')
    dock1.addWidget(act1_widget)
    area.addDock(dock1)

    dock2 = Dock('actuator 2')
    dock2.addWidget(act2_widget)
    area.addDock(dock2)

    act1.ui.IniStage_pb.click()
    act2.ui.IniStage_pb.click()

    QtWidgets.QApplication.processEvents()
    win.show()

    manager = ModulesManager(actuators=[act1, act2], detectors=[prog, prog2, prog3], selected_detectors=[prog2])
    manager.settings_tree.show()

    sys.exit(app.exec_())
