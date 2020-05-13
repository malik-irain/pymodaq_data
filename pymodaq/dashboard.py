#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import datetime
import pickle
import logging
from pathlib import Path

from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtCore import Qt,QObject, pyqtSlot, QThread, pyqtSignal, QLocale

from pyqtgraph.dockarea import Dock
from pyqtgraph.parametertree import Parameter, ParameterTree
import pymodaq.daq_utils.custom_parameter_tree as custom_tree# to be placed after importing Parameter

from pymodaq.daq_utils import daq_utils as utils

from pymodaq.daq_utils import gui_utils as gutils
from pymodaq.daq_utils.pid.pid_controller import DAQ_PID
from pymodaq.version import get_version
from pymodaq.daq_utils.manage_preset import PresetManager
from pymodaq.daq_utils.overshoot_manager import OvershootManager
from pymodaq.daq_utils.roi_saver import ROISaver
from pymodaq.daq_move.daq_move_main import DAQ_Move
from pymodaq.daq_viewer.daq_viewer_main import DAQ_Viewer
from pymodaq.daq_scan import DAQ_Scan
from pymodaq.daq_logger import DAQ_Logger

logger = utils.set_logger(utils.get_module_name(__file__))


local_path = utils.get_set_local_dir()
now = datetime.datetime.now()
preset_path = utils.get_set_preset_path()
log_path = utils.get_set_log_path()
layout_path = utils.get_set_layout_path()
overshoot_path = utils.get_set_overshoot_path()
roi_path = utils.get_set_roi_path()




class DashBoard(QObject):
    """
    Main class initializing a DashBoard interface to display det and move modules and logger """
    status_signal = pyqtSignal(str)

    def __init__(self, dockarea):
        """

        Parameters
        ----------
        parent: (dockarea) instance of the modified pyqtgraph Dockarea (see daq_utils)
        """
        QLocale.setDefault(QLocale(QLocale.English, QLocale.UnitedStates))
        super().__init__()
        logger.info('Initializing Dashboard')
        self.wait_time = 1000
        self.scan_module = None
        self.database_module = None

        self.dockarea = dockarea
        self.dockarea.dock_signal.connect(self.save_layout_state_auto)
        self.mainwindow = dockarea.parent()
        self.title = ''
        splash_path = './splash.png'

        splash = QtGui.QPixmap(splash_path)
        self.splash_sc = QtWidgets.QSplashScreen(splash, Qt.WindowStaysOnTopHint)
        self.overshoot_manager = None
        self.preset_manager = None
        self.roi_saver = None

        self.overshoot = False
        self.preset_file = None
        self.move_modules = []
        self.detector_modules = []
        self.setupUI()

    @pyqtSlot(str)
    def add_status(self, txt):
        """
            Add the QListWisgetItem initialized with txt informations to the User Interface logger_list and to the save_parameters.logger array.

            =============== =========== ======================
            **Parameters**    **Type**   **Description**
            *txt*             string     the log info to add.
            =============== =========== ======================
        """
        try:
            now = datetime.datetime.now()
            new_item = QtWidgets.QListWidgetItem(now.strftime('%Y/%m/%d %H:%M:%S')+": "+txt)
            self.logger_list.addItem(new_item)

        except Exception as e:
            logger.exception(str(e))

    def clear_move_det_controllers(self):
        """
            Remove all docks containing Moves or Viewers.

            See Also
            --------
            quit_fun, update_status
        """
        try:
        #remove all docks containing Moves or Viewers
            if hasattr(self, 'move_modules'):
                if self.move_modules is not None:
                    for module in self.move_modules:
                        module.quit_fun()
                self.move_modules = None

            if hasattr(self, 'detector_modules'):
                if self.detector_modules is not None:
                    for module in self.detector_modules:
                        module.quit_fun()
                self.detector_modules = None
        except Exception as e:
            logger.exception(str(e))

    def load_scan_module(self):
        self.scan_module = DAQ_Scan(dockarea=self.dockarea, dashboard=self)
        self.scan_module.status_signal.connect(self.add_status)
        self.actions_menu.setEnabled(False)

    def load_log_module(self):
        self.log_module = DAQ_Logger(dockarea=self.dockarea, dashboard=self)
        self.log_module.status_signal.connect(self.add_status)
        self.actions_menu.setEnabled(False)

    def create_menu(self, menubar):
        """
            Create the menubar object looking like :
        """
        menubar.clear()

        # %% create Settings menu
        self.file_menu = menubar.addMenu('File')
        self.file_menu.addAction('Show log file', self.show_log)
        self.file_menu.addSeparator()
        quit_action = self.file_menu.addAction('Quit')
        quit_action.triggered.connect(self.quit_fun)

        self.settings_menu = menubar.addMenu('Settings')
        docked_menu = self.settings_menu.addMenu('Docked windows')
        action_load = docked_menu.addAction('Load Layout')
        action_save = docked_menu.addAction('Save Layout')
        action_clear = self.settings_menu.addAction('Clear moves/Detectors')
        action_clear.triggered.connect(self.clear_move_det_controllers)

        action_load.triggered.connect(self.load_layout_state)
        action_save.triggered.connect(self.save_layout_state)

        docked_menu.addSeparator()
        action_show_log = docked_menu.addAction('Show/hide log window')
        action_show_log.setCheckable(True)
        action_show_log.toggled.connect(self.logger_dock.setVisible)

        self.preset_menu = menubar.addMenu('Preset Modes')
        action_new_preset = self.preset_menu.addAction('New preset')
        # action.triggered.connect(lambda: self.show_file_attributes(type_info='preset'))
        action_new_preset.triggered.connect(self.create_preset)
        action_modify_preset = self.preset_menu.addAction('Modify preset')
        action_modify_preset.triggered.connect(self.modify_preset)
        self.preset_menu.addSeparator()
        load_preset = self.preset_menu.addMenu('Load presets')

        slots = dict([])
        for ind_file, file in enumerate(preset_path.iterdir()):
            if file.suffix == '.xml':
                filestem = file.stem
                slots[filestem] = load_preset.addAction(filestem)
                slots[filestem].triggered.connect(
                    self.create_menu_slot(preset_path.joinpath(file)))

        self.overshoot_menu = menubar.addMenu('Overshoot Modes')
        action_new_overshoot = self.overshoot_menu.addAction('New Overshoot')
        # action.triggered.connect(lambda: self.show_file_attributes(type_info='preset'))
        action_new_overshoot.triggered.connect(self.create_overshoot)
        action_modify_overshoot = self.overshoot_menu.addAction('Modify Overshoot')
        action_modify_overshoot.triggered.connect(self.modify_overshoot)
        self.overshoot_menu.addSeparator()
        load_overshoot = self.overshoot_menu.addMenu('Load Overshoots')
        self.overshoot_menu.setEnabled(False)

        slots_over = dict([])
        for ind_file, file in enumerate(utils.get_set_overshoot_path().iterdir()):
            if file.suffix == '.xml':
                filestem = file.stem
                slots_over[filestem] = load_overshoot.addAction(filestem)
                slots_over[filestem].triggered.connect(
                    self.create_menu_slot_over(utils.get_set_overshoot_path().joinpath(file)))


        self.roi_menu = menubar.addMenu('ROI Modes')
        action_new_roi = self.roi_menu.addAction('Save Current ROIs as a file')
        action_new_roi.triggered.connect(self.create_roi_file)
        action_modify_roi = self.roi_menu.addAction('Modify roi config')
        action_modify_roi.triggered.connect(self.modify_roi)
        self.roi_menu.addSeparator()
        load_roi = self.roi_menu.addMenu('Load roi configs')
        self.roi_menu.setEnabled(False)


        slots = dict([])
        for ind_file, file in enumerate(utils.get_set_roi_path().iterdir()):
            if file.suffix == '.xml':
                filestem = file.stem
                slots[filestem] = load_roi.addAction(filestem)
                slots[filestem].triggered.connect(
                    self.create_menu_slot_roi(utils.get_set_roi_path.joinpath(file)))

        #actions menu
        self.actions_menu = menubar.addMenu('Actions')
        action_scan = self.actions_menu.addAction('Do Scans')
        action_scan.triggered.connect(self.load_scan_module)
        action_log = self.actions_menu.addAction('Log data')
        action_log.triggered.connect(self.load_log_module)
        self.actions_menu.setEnabled(False)

        # help menu
        help_menu = menubar.addMenu('?')
        action_about = help_menu.addAction('About')
        action_about.triggered.connect(self.show_about)
        action_help = help_menu.addAction('Help')
        action_help.triggered.connect(self.show_help)
        action_help.setShortcut(QtCore.Qt.Key_F1)



    def create_menu_slot(self, filename):
        return lambda: self.set_preset_mode(filename)

    def create_menu_slot_roi(self, filename):
        return lambda: self.set_roi_configuration(filename)

    def create_menu_slot_over(self, filename):
        return lambda: self.set_overshoot_configuration(filename)

    def create_roi_file(self):
        try:
            if self.preset_file is not None:
                self.roi_saver.set_new_roi(self.preset_file.stem)
                self.create_menu(self.menubar)

        except Exception as e:
            logger.exception(str(e))


    def create_overshoot(self):
        try:
            if self.preset_file is not None:
                self.overshoot_manager.set_new_overshoot(self.preset_file.stem)
                self.create_menu(self.menubar)
        except Exception as e:
            logger.exception(str(e))

    def create_preset(self):
        try:
            self.preset_manager.set_new_preset()
            self.create_menu(self.menubar)
        except Exception as e:
            logger.exception(str(e))

    def load_layout_state(self, file=None):
        """
            Load and restore a layout state from the select_file obtained pathname file.

            See Also
            --------
            utils.select_file
        """
        try:
            if file is None:
                file = gutils.select_file(save=False, ext='dock')
            if file is not None:
                with open(str(file), 'rb') as f:
                    dockstate = pickle.load(f)
                    self.dockarea.restoreState(dockstate)
            file = file.name
            self.settings.child('loaded_files', 'layout_file').setValue(file)
        except Exception as e:
            logger.exception(str(e))

    def modify_overshoot(self):
        try:
            path = gutils.select_file(start_path=utils.get_set_overshoot_path(), save=False, ext='xml')
            if path != '':
                self.overshoot_manager.set_file_overshoot(path)

            else:  # cancel
                pass
        except Exception as e:
            logger.exception(str(e))

    def modify_roi(self):
        try:
            path = gutils.select_file(start_path=utils.get_set_roi_path(), save=False, ext='xml')
            if path != '':
                self.roi_saver.set_file_roi(path)

            else:  # cancel
                pass
        except Exception as e:
            logger.exception(str(e))

    def modify_preset(self):
        try:
            path = gutils.select_file(start_path=preset_path, save=False, ext='xml')
            if path != '':
                self.preset_manager.set_file_preset(path)

                if self.detector_modules != []:
                    mssg = QtWidgets.QMessageBox()
                    mssg.setText('You have to restart the application to take the modifications into account! Quitting the application...')
                    mssg.exec()

                    self.quit_fun()

            else:  # cancel
                pass
        except Exception as e:
            logger.exception(str(e))

    def quit_fun(self):
        """
            Quit the current instance of DAQ_scan and close on cascade move and detector modules.

            See Also
            --------
            quit_fun
        """
        try:

            for mov in self.move_modules:
                mov.init_signal.disconnect(self.update_init_tree)
            for det in self.detector_modules:
                det.init_signal.disconnect(self.update_init_tree)

            for module in self.move_modules:
                try:
                    module.quit_fun()
                    QtWidgets.QApplication.processEvents()
                    QThread.msleep(1000)
                    QtWidgets.QApplication.processEvents()
                except:
                    pass

            for module in self.detector_modules:
                try:
                    module.quit_fun()
                    QtWidgets.QApplication.processEvents()
                    QThread.msleep(1000)
                    QtWidgets.QApplication.processEvents()
                except:
                    pass
            areas = self.dockarea.tempAreas[:]
            for area in areas:
                area.win.close()
                QtWidgets.QApplication.processEvents()
                QThread.msleep(1000)
                QtWidgets.QApplication.processEvents()

            if hasattr(self, 'mainwindow'):
                self.mainwindow.close()

        except Exception as e:
            logger.exception(str(e))

    def save_layout_state(self, file=None):
        """
            Save the current layout state in the select_file obtained pathname file.
            Once done dump the pickle.

            See Also
            --------
            utils.select_file
        """
        try:
            dockstate = self.dockarea.saveState()
            if file is None:
                file = gutils.select_file(start_path=None, save=True, ext='dock')
            if file is not None:
                with open(str(file), 'wb') as f:
                    pickle.dump(dockstate, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.exception(str(e))

    def save_layout_state_auto(self):
        if self.preset_file is not None:
            path = layout_path.joinpath(self.preset_file.stem+'.dock')
            self.save_layout_state(path)

    def open_PID(self):
        area = self.dockarea.addTempArea()
        self.pid_controller = DAQ_PID(area, [], [])

    def set_file_preset(self, filename):
        """
            Set a file preset from the converted xml file given by the filename parameter.


            =============== =========== ===================================================
            **Parameters**    **Type**    **Description**
            *filename*        string      the name of the xml file to be converted/treated
            =============== =========== ===================================================

            Returns
            -------
            (Object list, Object list) tuple
                The updated (Move modules list, Detector modules list).

            See Also
            --------
            custom_tree.XML_file_to_parameter, set_param_from_param, stop_moves, update_status,DAQ_Move_main.daq_move, DAQ_viewer_main.daq_viewer
        """
        move_modules = []
        detector_modules = []
        if not isinstance(filename, Path):
            filename = Path(filename)

        if filename.suffix == '.xml':
            self.preset_file = filename
            self.preset_manager.set_file_preset(filename, show=False)
            move_docks = []
            det_docks_settings = []
            det_docks_viewer = []
            move_forms = []


            # ### set daq scan settings set in presets
            # try:
            #     for child in self.preset_manager.preset_params.child(('saving_options')).children():
            #         if hasattr(self, 'h5saver'):
            #             self.h5saver.settings.child((child.name())).setValue(child.value())
            # except Exception as e:
            #     logger.exception(str(e))

            ## set PID if checked in preset
            try:
                if self.preset_manager.preset_params.child(('use_pid')).value():
                    self.open_PID()
                    QtWidgets.QApplication.processEvents()
                    for child in custom_tree.iter_children_params(self.preset_manager.preset_params.child(('pid_settings')), []):
                        preset_path = self.preset_manager.preset_params.child(('pid_settings')).childPath(child)
                        self.pid_controller.settings.child(*preset_path).setValue(child.value())

                    move_modules.append(self.pid_controller)

            except Exception as e:
                logger.exception(str(e))

            #################################################################
            ###### sort plugins by IDs and within the same IDs by Master and Slave status
            plugins = []
            if isinstance(self.preset_manager.preset_params.child(('Moves')).children()[0], custom_tree.GroupParameterCustom):
                plugins += [{'type': 'move', 'value': child} for child in
                           self.preset_manager.preset_params.child(('Moves')).children()]
            if isinstance(self.preset_manager.preset_params.child(('Detectors')).children()[0], custom_tree.GroupParameterCustom):
                plugins += [{'type': 'det', 'value': child} for child in
                            self.preset_manager.preset_params.child(('Detectors')).children()]

            for plug in plugins:
                plug['ID'] = plug['value'].child('params', 'main_settings', 'controller_ID').value()
                if plug["type"] == 'det':
                    plug['status'] = plug['value'].child('params', 'detector_settings', 'controller_status').value()
                else:
                    plug['status'] = plug['value'].child('params', 'move_settings', 'multiaxes', 'multi_status').value()

            IDs = list(set([plug['ID'] for plug in plugins]))
            # %%
            plugins_sorted = []
            for id in IDs:
                plug_Ids = []
                for plug in plugins:
                    if plug['ID'] == id:
                        plug_Ids.append(plug)
                plug_Ids.sort(key=lambda status: status['status'])
                plugins_sorted.append(plug_Ids)
            #################################################################
            #######################

            ind_move = -1
            ind_det = -1
            for plug_IDs in plugins_sorted:
                for ind_plugin, plugin in enumerate(plug_IDs):

                    plug_name = plugin['value'].child(('name')).value()
                    plug_init = plugin['value'].child(('init')).value()
                    plug_settings = plugin['value'].child(('params'))
                    self.splash_sc.showMessage('Loading {:s} module: {:s}'.format(plugin['type'], plug_name),
                                               color=Qt.white)
                    if plugin['type'] == 'move':
                        ind_move += 1
                        plug_type = plug_settings.child('main_settings', 'move_type').value()
                        move_docks.append(Dock(plug_name, size=(150, 250)))
                        if ind_move == 0:
                            self.dockarea.addDock(move_docks[-1], 'right', self.logger_dock)
                        else:
                            self.dockarea.addDock(move_docks[-1], 'above', move_docks[-2])
                        move_forms.append(QtWidgets.QWidget())
                        mov_mod_tmp = DAQ_Move(move_forms[-1], plug_name)

                        mov_mod_tmp.ui.Stage_type_combo.setCurrentText(plug_type)
                        mov_mod_tmp.ui.Quit_pb.setEnabled(False)
                        QtWidgets.QApplication.processEvents()

                        utils.set_param_from_param(mov_mod_tmp.settings, plug_settings)
                        QtWidgets.QApplication.processEvents()

                        mov_mod_tmp.bounds_signal[bool].connect(self.stop_moves)
                        move_docks[-1].addWidget(move_forms[-1])
                        move_modules.append(mov_mod_tmp)

                        try:
                            if ind_plugin == 0:  # should be a master type plugin
                                if plugin['status'] != "Master":
                                    logger.error('error in the master/slave type for plugin {}'.format(plug_name))
                                if plug_init:
                                    move_modules[-1].ui.IniStage_pb.click()
                                    QtWidgets.QApplication.processEvents()
                                    if 'Mock' in plug_type:
                                        QThread.msleep(500)
                                    else:
                                        QThread.msleep(4000)  # to let enough time for real hardware to init properly
                                    QtWidgets.QApplication.processEvents()
                                    master_controller = move_modules[-1].controller
                            else:
                                if plugin['status'] != "Slave":
                                    logger.error('error in the master/slave type for plugin {}'.format(plug_name))
                                if plug_init:
                                    move_modules[-1].controller = master_controller
                                    move_modules[-1].ui.IniStage_pb.click()
                                    QtWidgets.QApplication.processEvents()
                                    if 'Mock' in plug_type:
                                        QThread.msleep(500)
                                    else:
                                        QThread.msleep(4000)  # to let enough time for real hardware to init properly
                                    QtWidgets.QApplication.processEvents()
                        except Exception as e:
                            logger.exception(str(e))


                    else:
                        ind_det += 1
                        plug_type = plug_settings.child('main_settings', 'DAQ_type').value()
                        plug_subtype = plug_settings.child('main_settings', 'detector_type').value()

                        det_docks_settings.append(Dock(plug_name + " settings", size=(150, 250)))
                        det_docks_viewer.append(Dock(plug_name + " viewer", size=(350, 350)))

                        if ind_det == 0:
                            self.logger_dock.area.addDock(det_docks_settings[-1],
                                                             'bottom')  # dockarea of the logger dock
                        else:
                            self.dockarea.addDock(det_docks_settings[-1], 'right', det_docks_viewer[-2])
                        self.dockarea.addDock(det_docks_viewer[-1], 'right', det_docks_settings[-1])

                        det_mod_tmp = DAQ_Viewer(self.dockarea, dock_settings=det_docks_settings[-1],
                                                 dock_viewer=det_docks_viewer[-1], title=plug_name,
                                                 DAQ_type=plug_type, parent_scan=self)
                        detector_modules.append(det_mod_tmp)
                        detector_modules[-1].ui.Detector_type_combo.setCurrentText(plug_subtype)
                        detector_modules[-1].ui.Quit_pb.setEnabled(False)
                        utils.set_param_from_param(det_mod_tmp.settings, plug_settings)
                        QtWidgets.QApplication.processEvents()

                        try:
                            if ind_plugin == 0:  # should be a master type plugin
                                if plugin['status'] != "Master":
                                    logger.error('error in the master/slave type for plugin {}'.format(plug_name))
                                if plug_init:
                                    detector_modules[-1].ui.IniDet_pb.click()
                                    QtWidgets.QApplication.processEvents()
                                    if 'Mock' in plug_subtype:
                                        QThread.msleep(500)
                                    else:
                                        QThread.msleep(4000)  # to let enough time for real hardware to init properly
                                    QtWidgets.QApplication.processEvents()
                                    master_controller = detector_modules[-1].controller
                            else:
                                if plugin['status'] != "Slave":
                                    logger.error('error in the master/slave type for plugin {}'.format(plug_name))
                                if plug_init:
                                    detector_modules[-1].controller = master_controller
                                    detector_modules[-1].ui.IniDet_pb.click()
                                    QtWidgets.QApplication.processEvents()
                                    if 'Mock' in plug_subtype:
                                        QThread.msleep(500)
                                    else:
                                        QThread.msleep(4000)  # to let enough time for real hardware to init properly
                                    QtWidgets.QApplication.processEvents()
                        except Exception as e:
                            logger.exception(str(e))

                        detector_modules[-1].settings.child('main_settings', 'overshoot').show()
                        detector_modules[-1].overshoot_signal[bool].connect(self.stop_moves)

            QtWidgets.QApplication.processEvents()
            # restore dock state if saved

            self.title = self.preset_file.stem
            path = layout_path.joinpath(self.title + '.dock')
            if path.is_file():
                self.load_layout_state(path)

            self.mainwindow.setWindowTitle(f'PyMoDAQ Dashboard: {self.title}')

            return move_modules, detector_modules
        else:
            logger.error('Invalid file selected')
            return move_modules, detector_modules

    def set_roi_configuration(self, filename):
        if not isinstance(filename, Path):
            filename = Path(filename)
        try:
            if filename.suffix == '.xml':
                file = filename.stem
                self.settings.child('loaded_files', 'roi_file').setValue(file)
                self.update_status('ROI configuration ({}) has been loaded'.format(file),
                                   log_type='log')
                self.roi_saver.set_file_roi(filename, show=False)

        except Exception as e:
            logger.exception(str(e))

    def set_overshoot_configuration(self, filename):
        try:
            if not isinstance(filename, Path):
                filename = Path(filename)

            if filename.suffix == '.xml':
                file = filename.stem
                self.settings.child('loaded_files', 'overshoot_file').setValue(file)
                self.update_status('Overshoot configuration ({}) has been loaded'.format(file),
                                   log_type='log')
                self.overshoot_manager.set_file_overshoot(filename, show=False)

                det_titles = [det.title for det in self.detector_modules]
                move_titles = [move.title for move in self.move_modules]

                for det_param in self.overshoot_manager.overshoot_params.child(('Detectors')).children():
                    if det_param.child(('trig_overshoot')).value():
                        det_index = det_titles.index(det_param.opts['title'])
                        det_module = self.detector_modules[det_index]
                        det_module.settings.child('main_settings', 'overshoot', 'stop_overshoot').setValue(True)
                        det_module.settings.child('main_settings', 'overshoot', 'overshoot_value').setValue(
                            det_param.child(('overshoot_value')).value())
                        for move_param in det_param.child(('params')).children():
                            if move_param.child(('move_overshoot')).value():
                                move_index = move_titles.index(move_param.opts['title'])
                                move_module = self.move_modules[move_index]
                                det_module.overshoot_signal.connect(
                                    self.create_overshoot_fun(move_module, move_param.child(('position')).value()))

        except Exception as e:
            logger.exception(str(e))

    def create_overshoot_fun(self, move_module, position):
        return lambda: move_module.move_Abs(position)

    def set_preset_mode(self, filename):
        """
            | Set the preset mode from the given filename.
            |
            | In case of "mock" or "canon" move, set the corresponding preset calling set_(*)_preset procedure.
            |
            | Else set the preset file using set_file_preset function.
            | Once done connect the move and detector modules to logger to recipe/transmit informations.

            Finally update DAQ_scan_settings tree with :
                * Detectors
                * Move
                * plot_form.

            =============== =========== =============================================
            **Parameters**    **Type**    **Description**
            *filename*        string      the name of the preset file to be treated
            =============== =========== =============================================

            See Also
            --------
            set_Mock_preset, set_canon_preset, set_file_preset, add_status, update_status
        """
        try:
            if not isinstance(filename, Path):
                filename = Path(filename)
            self.mainwindow.setVisible(False)
            for area in self.dockarea.tempAreas:
                area.window().setVisible(False)

            self.splash_sc.show()
            QtWidgets.QApplication.processEvents()
            self.splash_sc.raise_()
            self.splash_sc.showMessage('Loading Modules, please wait', color=Qt.white)
            QtWidgets.QApplication.processEvents()
            self.clear_move_det_controllers()
            QtWidgets.QApplication.processEvents()


            move_modules, detector_modules= self.set_file_preset(filename)
            if not(not move_modules and not detector_modules):
                self.update_status('Preset mode ({}) has been loaded'.format(filename.name), log_type='log')
                self.settings.child('loaded_files', 'preset_file').setValue(filename.name)
                self.move_modules = move_modules
                self.detector_modules = detector_modules
    
                #####################################################
                self.overshoot_manager = OvershootManager(det_modules=[det.title for det in detector_modules],
                                                          move_modules=[move.title for move in move_modules])
                # load overshoot if present
                file = filename.name
                path = overshoot_path.joinpath(file)
                if path.is_file():
                    self.set_overshoot_configuration(path)
    
                self.roi_saver = ROISaver(det_modules=detector_modules)
                #load roi saver if present
                path = roi_path.joinpath(file)
                if path.is_file():
                    self.set_roi_configuration(path)
    
                #connecting to logger
                for mov in move_modules:
                    mov.status_signal[str].connect(self.add_status)
                    mov.init_signal.connect(self.update_init_tree)
                for det in detector_modules:
                    det.status_signal[str].connect(self.add_status)
                    det.init_signal.connect(self.update_init_tree)
    
                self.splash_sc.close()
                self.mainwindow.setVisible(True)
                for area in self.dockarea.tempAreas:
                    area.window().setVisible(True)
    
                self.file_menu.setEnabled(True)
                self.settings_menu.setEnabled(True)
                self.overshoot_menu.setEnabled(True)
                self.actions_menu.setEnabled(True)
                self.roi_menu.setEnabled(True)
                self.update_init_tree()

        except Exception as e:
            logger.exception(str(e))

    def update_init_tree(self):
        for act in self.move_modules:
            if act.title not in custom_tree.iter_children(self.settings.child(('actuators')), []):
                title = act.title
                name = ''.join(title.split())  # remove empty spaces
                self.settings.child(('actuators')).addChild(
                    {'title': title, 'name': name, 'type': 'led', 'value': False})
            QtWidgets.QApplication.processEvents()
            self.settings.child('actuators', name).setValue(act.initialized_state)

        for act in self.detector_modules:
            if act.title not in custom_tree.iter_children(self.settings.child(('detectors')), []):
                title = act.title
                name = ''.join(title.split())  # remove empty spaces
                self.settings.child(('detectors')).addChild(
                    {'title': title, 'name': name, 'type': 'led', 'value': False})
            QtWidgets.QApplication.processEvents()
            self.settings.child('detectors', name).setValue(act.initialized_state)


    pyqtSlot(bool)
    def stop_moves(self,overshoot):
        """
            Foreach module of the move module object list, stop motion.

            See Also
            --------
            stop_scan,  DAQ_Move_main.daq_move.stop_Motion
        """
        self.overshoot = overshoot
        if self.scan_module is not None:
            self.scan_module.stop_scan()

        for mod in self.move_modules:
            mod.stop_Motion()


    def show_log(self):
        import webbrowser
        webbrowser.open(logging.getLogger('pymodaq').handlers[0].baseFilename)

    def setupUI(self):

        # %% create logger dock
        self.logger_dock = Dock("Logger")
        self.logger_list = QtWidgets.QListWidget()
        self.logger_list.setMinimumWidth(300)
        self.init_tree = ParameterTree()
        self.init_tree.setMinimumWidth(300)
        self.logger_dock.addWidget(self.init_tree)
        self.logger_dock.addWidget(self.logger_list)

        self.settings = Parameter.create(name='init_settings', type='group', children=[
            {'title': 'Log level', 'name': 'log_level', 'type': 'list', 'value': 'DEBUG', 'values': ['DEBUG', 'INFO',
                                                                                              'WARNING', 'ERROR',
                                                                                              'CRITICAL']},
            {'title': 'Loaded presets', 'name': 'loaded_files', 'type': 'group', 'children': [
                {'title': 'Preset file', 'name': 'preset_file', 'type': 'str', 'value': '', 'readonly': True},
                {'title': 'Overshoot file', 'name': 'overshoot_file', 'type': 'str', 'value': '', 'readonly': True},
                {'title': 'Layout file', 'name': 'layout_file', 'type': 'str', 'value': '', 'readonly': True},
                {'title': 'ROI file', 'name': 'roi_file', 'type': 'str', 'value': '', 'readonly': True},
            ]},
                {'title': 'Actuators Init.', 'name': 'actuators', 'type': 'group', 'children': []},
                {'title': 'Detectors Init.', 'name': 'detectors', 'type': 'group', 'children': []},
                ])
        self.init_tree.setParameters(self.settings, showTop=False)
        self.dockarea.addDock(self.logger_dock, 'top')
        self.logger_dock.setVisible(True)

        self.preset_manager = PresetManager()

        #creating the menubar
        self.menubar=self.mainwindow.menuBar()
        self.create_menu(self.menubar)

#        connecting
        self.status_signal[str].connect(self.add_status)


        self.file_menu.setEnabled(True)
        self.actions_menu.setEnabled(False)
        self.settings_menu.setEnabled(True)
        self.preset_menu.setEnabled(True)
        self.mainwindow.setVisible(True)

    def parameter_tree_changed(self, param, changes):
        """
            Foreach value changed, update :
                * Viewer in case of **DAQ_type** parameter name
                * visibility of button in case of **show_averaging** parameter name
                * visibility of naverage in case of **live_averaging** parameter name
                * scale of axis **else** (in 2D pymodaq type)

            Once done emit the update settings signal to link the commit.

            =============== =================================== ================================================================
            **Parameters**    **Type**                           **Description**
            *param*           instance of ppyqtgraph parameter   the parameter to be checked
            *changes*         tuple list                         Contain the (param,changes,info) list listing the changes made
            =============== =================================== ================================================================

            See Also
            --------
            change_viewer, daq_utils.custom_parameter_tree.iter_children
        """

        for param, change, data in changes:
            path = self.settings.childPath(param)
            if path is not None:
                childName = '.'.join(path)
            else:
                childName = param.name()
            if change == 'childAdded':
                pass
            elif change == 'value':
                if param.name() == 'log_level':
                    logger.setLevel(getattr(logging, param.value().upper()))
                #TODO set a proper logging scheme: https://docs.python.org/3/howto/logging.html
            elif change == 'parent':
                pass


    def show_about(self):
        self.splash_sc.setVisible(True)
        self.splash_sc.showMessage("PyMoDAQ version {:}\nModular Acquisition with Python\nWritten by Sébastien Weber".format(get_version()), QtCore.Qt.AlignRight, QtCore.Qt.white)

    def show_file_attributes(self, type_info='dataset'):
        """
            Switch the type_info value.

            In case of :
                * *scan* : Set parameters showing top false
                * *dataset* : Set parameters showing top false
                * *preset* : Set parameters showing top false. Add the save/cancel buttons to the accept/reject dialog (to save preset parameters in a xml file).

            Finally, in case of accepted preset type info, save the preset parameters in a xml file.

            =============== =========== ====================================
            **Parameters**    **Type**    **Description**
            *type_info*       string      The file type information between
                                            * scan
                                            * dataset
                                            * preset
            =============== =========== ====================================

            See Also
            --------
            custom_tree.parameter_to_xml_file, create_menu
        """
        dialog = QtWidgets.QDialog()
        vlayout = QtWidgets.QVBoxLayout()
        tree = ParameterTree()
        tree.setMinimumWidth(400)
        tree.setMinimumHeight(500)
        if type_info=='scan':
            tree.setParameters(self.scan_attributes, showTop=False)
        elif type_info=='dataset':
            tree.setParameters(self.dataset_attributes, showTop=False)


        vlayout.addWidget(tree)
        dialog.setLayout(vlayout)
        buttonBox = QtWidgets.QDialogButtonBox(parent=dialog)
        buttonBox.addButton('Cancel', buttonBox.RejectRole)
        buttonBox.addButton('Apply', buttonBox.AcceptRole)
        buttonBox.rejected.connect(dialog.reject)
        buttonBox.accepted.connect(dialog.accept)

        vlayout.addWidget(buttonBox)
        dialog.setWindowTitle('Fill in information about this {}'.format(type_info))
        res=dialog.exec()
        return res

    def show_help(self):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl("http://pymodaq.cnrs.fr"))


    def update_status(self,txt,wait_time=0, log_type=None):
        """
            Show the txt message in the status bar with a delay of wait_time ms.

            =============== =========== =======================
            **Parameters**    **Type**    **Description**
            *txt*             string      The message to show
            *wait_time*       int         the delay of showing
            *log_type*        string      the type of the log
            =============== =========== =======================
        """
        try:
            if log_type is not None:
                self.status_signal.emit(txt)
                logging.info(txt)
        except Exception as e:
            pass


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = QtWidgets.QMainWindow()
    area = gutils.DockArea()
    win.setCentralWidget(area)
    win.resize(1000, 500)
    win.setWindowTitle('PyMoDAQ Dashboard')

    #win.setVisible(False)
    prog = DashBoard(area)
    sys.exit(app.exec_())

