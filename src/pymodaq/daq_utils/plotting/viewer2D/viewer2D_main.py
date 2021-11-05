from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtWidgets import QPushButton, QLabel, QCheckBox
from qtpy.QtGui import QPixmap, QIcon
from qtpy.QtCore import QObject, Slot, Signal, QRectF, QPointF
import sys
from collections import OrderedDict
from pymodaq.daq_utils.managers.roi_manager import ROIManager
import pyqtgraph as pg

from pymodaq.daq_utils.plotting.viewer2D.viewer2D_basic import ImageWidget
from pymodaq.daq_utils.plotting.graph_items import ImageItem, PlotCurveItem, TriangulationItem, AxisItem_Scaled, AXIS_POSITIONS
from pymodaq.daq_utils.plotting.crosshair import Crosshair

import numpy as np
from easydict import EasyDict as edict
import copy
from pymodaq.daq_utils.gui_utils import DockArea, ActionManager

import pymodaq.daq_utils.daq_utils as utils
import datetime
from pymodaq.resources.QtDesigner_Ressources import QtDesigner_ressources_rc

from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

logger = utils.set_logger(utils.get_module_name(__file__))

Gradients.update(OrderedDict([
    ('red', {'ticks': [(0.0, (0, 0, 0, 255)), (1.0, (255, 0, 0, 255))], 'mode': 'rgb'}),
    ('green', {'ticks': [(0.0, (0, 0, 0, 255)), (1.0, (0, 255, 0, 255))], 'mode': 'rgb'}),
    ('blue', {'ticks': [(0.0, (0, 0, 0, 255)), (1.0, (0, 0, 255, 255))], 'mode': 'rgb'}),
    ('spread', {'ticks': [(0.0, (0, 0, 0, 255)), (1.0, (255, 255, 255, 255))], 'mode': 'rgb'}),]))

COLORS_DICT = dict(red=(255, 0, 0), green=(0, 255, 0), blue=(0, 0, 255), spread=(128, 128, 128))
IMAGE_TYPES = ['red', 'green', 'blue', 'spread']
LINEOUT_WIDGETS = ['hor', 'ver', 'int']
COLOR_LIST = utils.plot_colors


def image_item_factory(axisOrder='row-major'):
    image = ImageItem()
    image.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)
    image.setOpts(axisOrder=axisOrder)
    return image


def histogram_factory(image_item=None, gradient='red'):
    """
    Create a pyqtgraph HistogramLUTWidget widget (histogram) and link it to the corresponding image_item
    Parameters
    ----------
    image_item: (ImageItem) the image item to be linked with the histogram
    gradient: (str) either 'red', 'green', 'blue', 'spread' or one of the Gradients

    Returns
    -------
    HistogramLUTWidget instance
    """

    if gradient not in Gradients:
        return KeyError(f'Possible gradient are {Gradients} not {gradient}')

    histo = pg.HistogramLUTWidget()
    if image_item is not None:
        histo.setImageItem(image_item)

    histo.gradient.loadPreset(gradient)
    return histo


def curve_item_factory(pen='red'):
    """
    Create a PlotCurveItem with the given pen
    Parameters
    ----------
    pen: any type of arguments accepted by pyqtgraph.function.mkColor or one of the COLORS_DICT key

    Returns
    -------
    PlotCurveItem
    """
    if isinstance(pen, str):
        if pen in COLORS_DICT:
            pen = COLORS_DICT[pen]

    return PlotCurveItem(pen=pen)


def extract_axis_info(axis):
    label = ''
    units = ''
    if isinstance(axis, dict):
        if 'data' in axis:
            data = axis['data']
        if 'label' in axis:
            label = axis['label']
        if 'units' in axis:
            units = axis['units']
    else:
        data = axis

    if len(data) > 1:
        scaling = data[1] - data[0]
        if scaling > 0:
            offset = np.min(data)
        else:
            offset = np.max(data)
    else:
        scaling = 1.
        offset = 0.
    return scaling, offset, label, units


class Filter:
    def __init__(self, roi_settings, graph_item):
        self.roi_settings = roi_settings
        self.graph_item = graph_item
        self.axes = (0, 1)
        self.data = None

    def data_from_roi(self, data, roi):

        data, coords = roi.getArrayRegion(data, self.graph_item, self.axes, returnMappedCoords=True)
        return data, coords

    def get_xydata_from_roi(self, roi_key, roi):
        image_source = self.roi_settings.child('ROIs', roi_key, 'use_channel').value()
        data = self.data[image_source]
        if data is not None:
            if image_source == 'spread':
                xvals, yvals, data = self.get_xydata_spread(data, roi)
            else:
                xvals, yvals, data = self.get_xydata(data, roi)

            if len(data) > 0:
                if image_source == 'spread':
                    ind_xaxis = np.argsort(xvals)
                    ind_yaxis = np.argsort(yvals)
                    xvals = xvals[ind_xaxis]
                    yvals = yvals[ind_yaxis]
                    data_H = data[ind_xaxis]
                    data_V = data[ind_yaxis]
                else:
                    data_H = np.mean(data, axis=0)
                    data_V = np.mean(data, axis=1)

                    return dict(hor_axis=xvals, ver_axis=yvals, data_hor=data_H, data_ver=data_V)

    def get_xydata(self, data, roi):
        data, coords = self.data_from_roi(data, roi)

        if data is not None:
            xvals = np.linspace(np.min(np.min(coords[1, :, :])), np.max(np.max(coords[1, :, :])),
                                data.shape[1])
            yvals = np.linspace(np.min(np.min(coords[0, :, :])), np.max(np.max(coords[0, :, :])),
                                data.shape[0])
            return xvals, yvals, data

    def get_xydata_spread(self, data, roi):
        xvals = []
        yvals = []
        data_out = []
        for ind in range(data.shape[0]):
            # invoke the QPainterpath of the ROI (from the shape method)
            if roi.shape().contains(QPointF(data[ind, 0] - roi.pos().x(),
                                            data[ind, 1] - roi.pos().y())):
                xvals.append(data[ind, 0])
                yvals.append(data[ind, 1])
                data_out.append(data[ind, 2])
        if len(data_out) != 0:
            data_out = np.array(data_out)
            xvals = np.array(xvals)
            yvals = np.array(yvals)
            return xvals, yvals, data_out


class ActionManager(ActionManager):
    def __init__(self, toolbar=None):
        super().__init__(toolbar=toolbar)

    def setup_actions(self):

        self.addaction('position', '(,)')
        self.addaction('red', 'Red Channel', 'r_icon', tip='Show/Hide Red Channel', checkable=True)
        self.addaction('green', 'Green Channel', 'g_icon', tip='Show/Hide Green Channel', checkable=True)
        self.addaction('blue', 'Blue Channel', 'b_icon', tip='Show/Hide Blue Channel', checkable=True)
        self.addaction('spread', 'Spread Channel', 'grey_icon',
                       tip='Show/Hide Spread Channel', checkable=True)

        self.addaction('histo', 'Histogram', 'Histogram', tip='Show/Hide Histogram', checkable=True)
        self.addaction('roi', 'ROI', 'Region', tip='Show/Hide ROI Manager', checkable=True)
        self.addaction('isocurve', 'IsoCurve', 'meshPlot', tip='Show/Hide Isocurve', checkable=True)
        self.addaction('init_plot', 'Init. Plot', 'Refresh', tip='Initialize the plots')

        self.addaction('aspect_ratio', 'Aspect Ratio', 'Zoom_1_1', tip='Fix Aspect Ratio', checkable=True)
        self.addaction('auto_levels', 'AutoLevels', 'autoscale',
                       tip='Scale Histogram to Min/Max intensity', checkable=True)
        self.addaction('auto_levels_sym', 'AutoLevels Sym.', 'autoscale',
                       tip='Make the autoscale of the histograms symetric with respect to 0', checkable=True)

        self.addaction('crosshair', 'CrossHair', 'reset', tip='Show/Hide data Crosshair', checkable=True)
        self.addaction('ROIselect', 'ROI Select', 'Select_24',
                       tip='Show/Hide ROI selection area', checkable=True)
        self.addaction('flip_ud', 'Flip UD', 'scale_vertically',
                       tip='Flip the image up/down', checkable=True)
        self.addaction('flip_lr', 'Flip LR', 'scale_horizontally',
                       tip='Flip the image left/right', checkable=True)
        self.addaction('rotate', 'Rotate', 'rotation2',
                       tip='Rotate the image', checkable=True)


class View2D(QObject):

    roi_changed = Signal(dict)

    def __init__(self, parent):
        super().__init__()
        # setting the gui
        self.parent = parent
        self.image_widget = ImageWidget()
        self.setupUI()

        self.action_manager = ActionManager(self.toolbar)

        self.roi_manager = ROIManager(self.image_widget, '2D')
        self.crosshair = Crosshair(self.image_widget)

        self._image_items = dict([])
        self._crosshair_hcurves = dict([])
        self._crosshair_vcurves = dict([])
        self._roi_curves = OrderedDict()

        self.setup_images()
        self.setup_histograms()
        self.setup_iso_curve()
        self.setup_crosshair()
        self.setupROI()

        self.prepare_connect_internal_ui()

        self.set_axis_label('bottom', label='', units='Pxls')
        self.set_axis_label('left', label='', units='Pxls')

    def __getattr__(self, item):
        """
        If item is not found in self, try to look for an attribute in ActionManager such as:
        is_action_visible, is_action_checked, set_action_visible, set_action_checked, connect_action
        """
        if self.action_manager is not None:
            if hasattr(self.action_manager, item):
                return getattr(self.action_manager, item)
        else:
            raise AttributeError(f'Attribute {item} cannot be found in self nor in action_manager')


    @property
    def autolevels(self):
        return self.is_action_checked('auto_levels')

    @property
    def autolevels_sym(self):
        return self.is_action_checked('auto_levels_sym')

    @Slot()
    def set_autolevels(self):
        if not self.autolevels:
            for histo in self._histograms.values():
                histo.regionChanged()
        for histo in self._histograms.values():
            histo.region.setVisible(not self.autolevels)

    def prepare_connect_internal_ui(self):
        for key in IMAGE_TYPES:
            self.connect_action(key, self.update_image_visibility)

        self.connect_action('aspect_ratio', self.lock_aspect_ratio)

        self.connect_action('histo', self.show_hide_histogram)

        self.connect_action('roi', self.roi_clicked)

        self.ROIselect.setVisible(False)
        self.connect_action('ROIselect', self.show_ROI_select)

        self.connect_action('isocurve', self.show_hide_iso)
        self.isoLine.sigDragged.connect(self.update_isocurve)
        self.set_action_checked('isocurve', False)
        self.show_hide_iso()

        self.connect_action('crosshair', self.show_hide_crosshair)
        self.show_hide_crosshair()

        self.show_lineouts()

    def update_image_visibility(self):
        for key in IMAGE_TYPES:
            self._image_items[key].setVisible(self._actions[key].isChecked())

    @Slot(dict)
    def display_images(self, kwargs):
        for key, val in kwargs.items():
            self._image_items[key].setImage(val, self.autolevels)

    def prepare_image_widget_for_lineouts(self, ratio=0.7):
        QtGui.QGuiApplication.processEvents()
        self.splitter_VRight.splitterMoved[int, int].emit(int(ratio * self.parent.height()), 1)
        self.splitter.moveSplitter(int(ratio * self.parent.width()), 1)
        self.splitter_VLeft.moveSplitter(int(ratio * self.parent.height()), 1)
        self.splitter_VLeft.splitterMoved[int, int].emit(int(ratio * self.parent.height()), 1)
        QtGui.QGuiApplication.processEvents()

    def get_view_range(self):
        return self.image_widget.view.viewRange()

    def lock_aspect_ratio(self):
        lock = self.is_action_checked('aspect_ratio')
        self.plotitem.vb.setAspectLocked(lock=lock, ratio=1)

    def setupUI(self):
        vertical_layout = QtWidgets.QVBoxLayout()
        vertical_layout.setContentsMargins(5, 5, 5, 5)
        self.parent.setLayout(vertical_layout)
        splitter_vertical = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        vertical_layout.addWidget(splitter_vertical)

        # ###### Actions ##########
        self.toolbar = QtWidgets.QToolBar()
        splitter_vertical.addWidget(self.toolbar)
        # ############################

        # ####### Graphs, ImageItem, Histograms ############
        self.graphs_widget = QtWidgets.QWidget()
        self.graphs_widget.setLayout(QtWidgets.QHBoxLayout())
        self.graphs_widget.layout().setContentsMargins(0, 0, 0, 0)
        self.setupGraphs(self.graphs_widget.layout())
        splitter_vertical.addWidget(self.graphs_widget)

        self.ROIselect = pg.RectROI([0, 0], [10, 10], centered=True, sideScalers=True)
        self.plotitem.addItem(self.ROIselect)

        self.splitter_VLeft.splitterMoved[int, int].connect(self.move_right_splitter)
        self.splitter_VRight.splitterMoved[int, int].connect(self.move_left_splitter)

    @Slot(int, int)
    def move_left_splitter(self, pos, index):
        self.splitter_VLeft.blockSignals(True)
        self.splitter_VLeft.moveSplitter(pos, index)
        self.splitter_VLeft.blockSignals(False)

    @Slot(int, int)
    def move_right_splitter(self, pos, index):
        self.splitter_VRight.blockSignals(True)
        self.splitter_VRight.moveSplitter(pos, index)
        self.splitter_VRight.blockSignals(False)

    def add_histogram(self, histogram):
        self.widget_histo.layout().addWidget(histogram)

    def setupGraphs(self, graphs_layout):
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        graphs_layout.addWidget(self.splitter)

        self.widget_histo = QtWidgets.QWidget()
        graphs_layout.addWidget(self.widget_histo)
        self.widget_histo.setLayout(QtWidgets.QHBoxLayout())

        self.splitter_VLeft = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.splitter_VRight = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        self.splitter.addWidget(self.splitter_VLeft)
        self.splitter.addWidget(self.splitter_VRight)

        self._lineout_widgets = {widg_key: pg.PlotWidget() for widg_key in LINEOUT_WIDGETS}
        
        self.splitter_VLeft.addWidget(self.image_widget)
        self.splitter_VLeft.addWidget(self._lineout_widgets['hor'])
        self.splitter_VRight.addWidget(self._lineout_widgets['ver'])
        self.splitter_VRight.addWidget(self._lineout_widgets['int'])

        self.image_widget.add_scaled_axis('right')
        self.image_widget.add_scaled_axis('top')

    def setup_images(self):
        for img_key in IMAGE_TYPES:
            self._image_items[img_key] = image_item_factory()
        self._image_items['spread'] = TriangulationItem()
        for image in self._image_items:
            self.plotitem.addItem(self._image_items[image])

    def setup_histograms(self):
        self._histograms = dict([])
        for histo_key in IMAGE_TYPES:
            self._histograms[histo_key] = histogram_factory(self._image_items[histo_key], gradient=histo_key)
            self.add_histogram(self._histograms[histo_key])
            self._histograms[histo_key].setVisible(False)

    def show_hide_histogram(self):
        for histo_name, histo in self._histograms.items():
            if self.is_action_checked(histo_name):
                histo.setVisible(self.is_action_checked('histo'))

    def setup_crosshair(self):
        for curve_key in IMAGE_TYPES:
            self._crosshair_hcurves[curve_key] = curve_item_factory(curve_key)
            self._crosshair_vcurves[curve_key] = curve_item_factory(curve_key)

        for curve_key in IMAGE_TYPES:
            self.add_lineout_items(self._crosshair_hcurves[curve_key], self._crosshair_vcurves[curve_key])

    def setupROI(self):
        self.roi_manager.new_ROI_signal.connect(self.add_ROI)
        self.roi_manager.remove_ROI_signal.connect(self.remove_ROI)
        self.roi_manager.roi_settings_changed.connect(self.update_roi)
        self.splitter.addWidget(self.roi_manager.roiwidget)
        self.roi_manager.roiwidget.setVisible(False)

    def setup_iso_curve(self, parent_image_item='red'):
        # TODO provide isocurve for the spread points
        # # Isocurve drawing
        self._isocurve_item = pg.IsocurveItem(level=0.8, pen='g', axisOrder='row-major')
        self._isocurve_item.setParentItem(self._image_items[parent_image_item])
        self._isocurve_item.setZValue(5)

        # # Draggable line for setting isocurve level
        self.isoLine = pg.InfiniteLine(angle=0, movable=True, pen='g')
        self._histograms[parent_image_item].vb.addItem(self.isoLine)
        self._histograms[parent_image_item].vb.setMouseEnabled(y=False)  # makes user interaction a little easier
        self.isoLine.setValue(0.8)
        self.isoLine.setZValue(1000)  # bring iso line above contrast controls

    def update_isocurve(self):
        self._isocurve_item.setLevel(self.isoLine.value())

    def set_isocurve_data(self, data):
        self._isocurve_item.setData(data)

    def get_double_clicked(self):
        return self.image_widget.view.sig_double_clicked

    def get_lineout_widget(self, name):
        if name not in LINEOUT_WIDGETS:
            raise KeyError(f'The lineout_widget reference should be within {LINEOUT_WIDGETS} not {name}')
        return self._lineout_widgets[name]

    def get_axis(self, position='left'):
        return self.image_widget.getAxis(position)

    @property
    def plotitem(self):
        return self.image_widget.plotitem

    def add_roi_lineout_items(self, index, pen):
        """
        Add specifics lineouts generated from ROIs
        Parameters
        ----------
        index: (int) index of the ROI generating these lineouts
        pen: (str, tuple) any argument able to generate a QPen, see pyqtgraph.functions.mkPen
        """
        self._roi_curves[f'ROI_{index:02d}'] =\
            {curv_key: curve_item_factory(pen) for curv_key in LINEOUT_WIDGETS}
        self.add_lineout_items(*self._roi_curves[f'ROI_{index:02d}'].values())

    def remove_roi_lineout_items(self, index):
        """
        Remove specifics lineouts generated from ROI referenced by a unique integer
        Parameters
        ----------
        index: (int) index of the ROI generating these lineouts
        """
        items = self._roi_curves.pop(f'ROI_{index:02d}')
        self.remove_lineout_items(*items.values())

    def add_lineout_items(self, *curve_items):
        """
        Add Curve items sequentially to lineouts widgets: (hor, ver and int)
        Parameters
        ----------
        curve_items: (PlotCurveItem) at most 3 of them
        """
        for ind, curve_item in enumerate(curve_items):
            self.get_lineout_widget(LINEOUT_WIDGETS[ind]).addItem(curve_item)

    def remove_lineout_items(self, *curve_items):
        """
        Remove Curve items sequentially to lineouts widgets: (hor, ver and int)
        Parameters
        ----------
        curve_items: (PlotCurveItem) at most 3 of them
        """

        for ind, curve_item in enumerate(curve_items):
            self.get_lineout_widget(LINEOUT_WIDGETS[ind]).removeItem(curve_item)

    def get_roi_curves(self):
        return self._roi_curves

    def show_hide_iso(self):
        if self.is_action_checked('isocurve'):
            self._isocurve_item.show()
            self.isoLine.show()
            self.set_action_checked('histo', True)
            self.show_hide_histogram()
        else:
            self._isocurve_item.hide()
            self.isoLine.hide()

    def set_gradient(self, histo='red', gradient='grey'):
        """
        Change the color gradient of the specified histogram
        Parameters
        ----------
        histo: (str) either 'red', 'green', 'blue', 'spread' or 'all'
        gradient: (str or Gradient)
        """
        if gradient in Gradients:
            if histo == 'all':
                for key in IMAGE_TYPES:
                    self._histograms[key].item.gradient.loadPreset(gradient)
            else:
                self._histograms[histo].item.gradient.loadPreset(gradient)

    def get_crosshair_signal(self):
        """Convenience function from the Crosshair"""
        return self.crosshair.crosshair_dragged

    def get_crosshair_position(self):
        """Convenience function from the Crosshair"""
        return self.crosshair.get_positions()

    def set_crosshair_position(self, *positions):
        """Convenience function from the Crosshair"""
        self.crosshair.set_crosshair_position(*positions)

    def show_hide_crosshair(self):
        if self.is_action_checked('crosshair'):
            self.crosshair.setVisible(True)
            self.set_action_checked('position', True)

            range = self.get_view_range()
            self.set_crosshair_position(np.mean(np.array(range[0])), np.mean(np.array(range[0])))

        else:
            self.set_action_visible('position', False)
            self.crosshair.setVisible(False)

    def show_crosshair_curve(self, curve_key, show=True):
        self._crosshair_hcurves[curve_key].setVisible(show)
        self._crosshair_vcurves[curve_key].setVisible(show)

    def show_ROI_select(self):
        self.ROIselect.setVisible(self.is_action_checked('ROIselect'))

    def show_lineouts(self):
        state = self.is_action_checked('roi') or self.is_action_checked('crosshair')
        for lineout_name in LINEOUT_WIDGETS:
            lineout = self.get_lineout_widget(lineout_name)
            lineout.setMouseEnabled(state, state)
            lineout.showAxis('left', state)
            lineout.setVisible(state)
            lineout.update()
        self.prepare_image_widget_for_lineouts()

    def set_axis_label(self, position, label='', units=''):
        """
        Convenience method to set label and unit of any view axes
        Parameters
        ----------
        position: (str) any of AXIS_POSITIONS
        label: (str) text of the axis label
        units: (str) units of the axis label
        """
        axis = self.get_axis(position)
        axis.setLabel(text=label, units=units)

    def set_axis_scaling(self, position='top', scaling=1, offset=0, label='', units='Pxls'):
        """
        Method used to update the scaling of the right and top axes in order to translate pixels to real coordinates
        Parameters
        ----------
        position: (str) axis position either one of AXIS_POSITIONS
        scaling: (float) scaling of the axis
        offset: (float) offset of the axis
        label: (str) text of the axis label
        units: (str) units of the axis label
        """

        self.get_axis(position).scaling = scaling
        self.get_axis(position).offset = offset
        self.set_axis_label(position, label=label, units=units)

    @Slot(list)
    def update_roi(self, changes):
        for param, change, param_value in changes:
            if change == 'value':
                if param.name() == 'Color':
                    key = param.parent().name()
                    for curve in self._roi_curves[key].values():
                        curve.setPen(param_value)

            elif change == 'childAdded':
                pass
            elif change == 'parent':
                key = param.name()
                for lineout_name in LINEOUT_WIDGETS:
                    self.view.get_lineout_widget(lineout_name).removeItem(self._roi_curves[key].pop[lineout_name])

        self.roi_changed.emit(self.roi_manager.ROIs)

    @Slot(str)
    def remove_ROI(self, roi_name):
        index = int(roi_name.split('_')[1])
        self.remove_roi_lineout_items(index)

        self.roi_changed.emit(self.roi_manager.ROIs)

    @Slot(int, str)
    def add_ROI(self, newindex, roi_type):
        item = self.roi_manager.ROIs['ROI_{:02d}'.format(newindex)]
        item.sigRegionChanged.connect(lambda: self.roi_changed.emit(self.roi_manager.ROIs))
        item_param = self.roi_manager.settings.child('ROIs', 'ROI_{:02d}'.format(newindex))
        color = item_param.child(('Color')).value()

        self.add_roi_lineout_items(newindex, color)

        #todo chekc if still needed
        #self.data_integrated_plot[f"ROI_{newindex:02d}"] = np.zeros((2, 1))

        # if self.isdata['red']:
        #     item_param.child('use_channel').setValue('red')
        # elif self.isdata['green']:
        #     item_param.child('use_channel').setValue('green')
        # elif self.isdata['blue']:
        #     item_param.child('use_channel').setValue('blue')
        # elif self.isdata['spread']:
        #     item_param.child('use_channel').setValue('spread')

        self.roi_changed.emit(self.roi_manager.ROIs)

    def scale_axis(self, xaxis, yaxis):
        x_offset, x_scaling, y_offset, y_scaling = self.get_axis_scaling_offset()
        return xaxis * x_scaling + x_offset, yaxis * y_scaling + y_offset

    def unscale_axis(self, xaxis, yaxis):
        x_offset, x_scaling, y_offset, y_scaling = self.get_axis_scaling_offset()
        return (xaxis - x_offset) / x_scaling, (yaxis - y_offset) / y_scaling

    def get_axis_scaling_offset(self):
        x_offset = self.get_axis('top').offset
        x_scaling = self.get_axis('top').scaling
        y_offset = self.get_axis('right').offset
        y_scaling = self.get_axis('right').scaling
        return x_offset, x_scaling, y_offset, y_scaling

    def roi_clicked(self):
        roistate = self.is_action_checked('roi')
        self.roi_manager.roiwidget.setVisible(roistate)

        for k, roi in self.roi_manager.ROIs.items():
            roi.setVisible(roistate)
            for item in self.get_roi_curves()[k].values():
                item.setVisible(roistate)

        if self.is_action_checked('roi'):
            self.roi_changed.emit(self.roi_manager.ROIs)

        self.show_lineouts()

        # if len(self.ROIs) == 0 and roistate:
        #     self.roi_manager.settings.child(("ROIs")).addNew('RectROI')

class Viewer2D(QObject):

    crosshair_dragged = Signal(float, float)  # signal used to pass crosshair position to other modules in
    crosshair_clicked = Signal(bool)
    # scaled axes units
    sig_double_clicked = Signal(float, float)

    roi_changed_signal = Signal()
    roi_change_finished_signal = Signal()

    def __init__(self, parent=None):
        super().__init__()

        self.viewer_type = 'Data2D'  # by default
        self.title = ""
        self.isdata = dict([])

        if parent is None:
            parent = QtWidgets.QWidget()
            parent.show()

        self.view = View2D(parent)
        filter = Filter(self.view.roi_manager.settings, self.view._image_items['red'])
        self.model = Model2D(filter)

        self.model.data_to_show_signal[dict].connect(self.view.display_images)
        self.view.roi_changed.connect(self.model.roi_changed)
        self.model.roi_lineout_signal.connect(self.process_roi_lineouts)

        self.prepare_connect_ui()

    def show_data(self, data_red=None, data_green=None, data_blue=None, data_spread=None):
        """
        numpy arrays to be plotted and eventually filtered using ROI...
        Parameters
        ----------
        data_red: (ndarray)
        data_green: (ndarray)
        data_blue: (ndarray)
        data_spread: (ndarray)
        """
        self.data_to_export = OrderedDict(name=self.title, data0D=OrderedDict(), data1D=OrderedDict(),
                                          data2D=OrderedDict())

        self.raw_data = dict(blue=data_blue, green=data_green, red=data_red, spread=data_spread)

        self.isdata['red'] = data_red is not None
        self.isdata['green'] = data_green is not None
        self.isdata['blue'] = data_blue is not None
        self.isdata['spread'] = data_spread is not None

        self.set_visible_items()
        self.update_model_data()

    def update_model_data(self):
        self.model.set_data(*self.set_image_transform())

    def set_visible_items(self):
        for key in IMAGE_TYPES:
            if self.view.is_action_checked(key) and not self.isdata[key]:  # turn it off if it was on but there is no data
                self.view.set_action_checked(key, False)
                self.view.set_action_visible(key, False)

            elif self.isdata[key]:
                self.view.set_action_checked(key, True)
                self.view.set_action_visible(key, True)

            self.view._image_items[key].setVisible(self.view.is_action_checked(key))
            if self.view.is_action_checked('histo'):
                self.view.histograms[key].setVisible(self.view.is_action_checked(key))

    def transform_image(self, data):
        if data is not None:
            if len(data.shape) > 2:
                data = np.mean(data, axis=0)
            if self.view.is_action_checked('flip_ud'):
                data = np.flipud(data)
            if self.view.is_action_checked('flip_lr'):
                data = np.fliplr(data)
            if self.view.is_action_checked('rotate'):
                data = np.flipud(np.transpose(data))
        if data is not None:
            return data.copy()
        else:
            return None

    def set_image_transform(self):
        """
        Deactivate some tool buttons if data type is "spread" then apply transform_image
        """
        # deactiviate fliping and rotation as non sense for points defined data
        status = self.isdata["red"] is False and self.isdata["blue"] is False
        status = status and self.isdata["green"] is False and self.isdata["spread"] is True
        status = not status

        self.view.set_action_visible('flip_ud', status)
        self.view.set_action_visible('flip_lr', status)
        self.view.set_action_visible('rotate', status)

        data_red, data_blue, data_green = None, None, None
        if self.isdata["red"]:
            data_red = self.transform_image(self.raw_data['red'])
        if self.isdata["blue"]:
            data_blue = self.transform_image(self.raw_data['blue'])
        if self.isdata["green"]:
            data_green = self.transform_image(self.raw_data['green'])
        return data_red, data_green, data_blue, self.raw_data['spread']

    def prepare_connect_ui(self):
        # selection area checkbox
        self.view.set_action_visible('red', True)
        self.view.set_action_checked('red', True)
        self.view.set_action_visible('green', True)
        self.view.set_action_checked('green', True)
        self.view.set_action_visible('blue', True)
        self.view.set_action_checked('blue', True)
        self.view.set_action_visible('spread', True)
        self.view.set_action_checked('spread', True)

        #self.view.ROIselect.sigRegionChangeFinished.connect(self.selected_region_changed)

        self.view.connect_action('flip_ud', slot=self.update_model_data)
        self.view.connect_action('flip_lr', slot=self.update_model_data)
        self.view.connect_action('rotate', slot=self.update_model_data)
        self.view.connect_action('auto_levels', slot=self.update_model_data)

        self.view.connect_action('init_plot', self.model.ini_plot)

        self.view.get_crosshair_signal().connect(self.model.update_crosshair_data)

        self.view.get_double_clicked().connect(self.double_clicked)

    @Slot(float, float)
    def double_clicked(self, posx, posy):
        self.crosshair.set_crosshair_position(posx, posy)
        self.update_crosshair_data(posx, posy)
        self.sig_double_clicked.emit(posx, posy)

    @property
    def x_axis(self):
        return self.view.get_axis('top')

    @x_axis.setter
    def x_axis(self, axis):
        scaling, offset, label, units = extract_axis_info(axis)
        self.view.set_axis_scaling('top', scaling=scaling, offset=offset, label=label, units=units)

    @property
    def y_axis(self):
        return self.view.get_axis('right')

    @y_axis.setter
    def y_axis(self, axis):
        scaling, offset, label, units = extract_axis_info(axis)
        self.view.set_axis_scaling('top', scaling=scaling, offset=offset, label=label, units=units)

    @Slot(dict)
    def process_roi_lineouts(self, roi_dict):
        for roi_key in roi_dict:

            x_axis, y_axis = self.view.scale_axis(roi_dict[roi_key]['hor_axis'], roi_dict[roi_key]['ver_axis'])
            #TODO
            self.data_integrated_plot[image_source] = np.append(self.data_integrated_plot[image_source], np.array(
                [[self.data_integrated_plot[image_source][0, -1]], [0]]) + np.array([[1], [np.sum(data)]]), axis=1)

            if self.data_integrated_plot[image_source].shape[1] > self.max_size_integrated:
                self.data_integrated_plot[image_source] = \
                    self.data_integrated_plot[image_source][:, self.data_integrated_plot[image_source].shape[1] - 200:]

            self.RoiCurve_H[image_source].setData(y=data_H, x=data_H_axis)
            self.RoiCurve_V[image_source].setData(y=data_V_axis, x=data_V)

            self.RoiCurve_integrated[image_source].setData(y=self.data_integrated_plot[image_source][1, :],
                                                     x=self.data_integrated_plot[image_source][0, :])

            self.data_to_export['data2D'][self.title + '_{:s}'.format(image_source)] = \
                utils.DataToExport(name=self.title, data=data, source='roi',
                                   x_axis=utils.Axis(data=x_axis,
                                                     units=self.scaling_options['scaled_xaxis']['units'],
                                                     label=self.scaling_options['scaled_xaxis']['label']),
                                   y_axis=utils.Axis(data=y_axis,
                                                     units=self.scaling_options['scaled_yaxis']['units'],
                                                     label=self.scaling_options['scaled_yaxis']['label']))

            self.data_to_export['data1D'][self.title + '_Hlineout_{:s}'.format(image_source)] = \
                utils.DataToExport(name=self.title, data=data_H, source='roi',
                                   x_axis=utils.Axis(data=data_H_axis,
                                                     units=self.scaling_options['scaled_xaxis']['units'],
                                                     label=self.scaling_options['scaled_xaxis']['label']))
            self.data_to_export['data1D'][self.title + '_Vlineout_{:s}'.format(image_source)] = \
                utils.DataToExport(name=self.title, data=data_V, source='roi',
                                   x_axis=utils.Axis(data=data_V_axis,
                                                     units=self.scaling_options['scaled_yaxis']['units'],
                                                     label=self.scaling_options['scaled_yaxis']['label']))

            self.data_to_export['data0D'][self.title + '_Integrated_{:s}'.format(image_source)] = \
                utils.DataToExport(name=self.title, data=np.sum(data), source='roi', )

            self.measure_data_dict['Lineout {:s}:'.format(image_source)] = np.sum(data)

    self.roi_settings.child('measurements').setValue(self.measure_data_dict)

    self.data_to_export['acq_time_s'] = datetime.datetime.now().timestamp()
    self.data_to_export_signal.emit(self.data_to_export)
    self.roi_changed_signal.emit()



class Model2D(QObject):
    data_to_show_signal = Signal(dict)
    roi_lineout_signal = Signal(dict)

    data_to_export_signal = Signal(OrderedDict)  # OrderedDict(name=self.DAQ_type,data0D=None,data1D=None,data2D=None)
    crosshair_dragged = Signal(float, float)  # signal used to pass crosshair position to other modules in
    # scaled axes units
    sig_double_clicked = Signal(float, float)

    def __init__(self, filter):
        super().__init__()

        self.data = None
        self.isdata = dict([])

        self.filter = filter

        self.max_size_integrated = 200
        self._x_axis = None
        self._y_axis = None
        self.raw_data = None
        self.data_to_export = OrderedDict([])


    def crosshairChanged(self, posx=None, posy=None):
        if self.raw_data is None:
            return
        data_red, data_blue, data_green = self.set_image_transform()
        if posx is None or posy is None:
            (posx, posy) = self.get_crosshair_position()

        if self.isdata["red"]:
            indx, indy = self.mapfromview('red', posx, posy)
            x_axis_scaled, y_axis_scaled = \
                self.scale_axis(np.linspace(0, self._image_items['red'].width() - 1, self._image_items['red'].width()),
                                np.linspace(0, self._image_items['red'].height() - 1, self._image_items['red'].height()))
            data_H_indexes = slice(None, None, 1)
            data_V_indexes = slice(None, None, 1)
            H_indexes = (utils.rint(indy), data_H_indexes)
            V_indexes = (data_V_indexes, utils.rint(indx))

            if self.isdata["blue"]:
                self.crosshair_H_blue.setData(y=data_blue.__getitem__(H_indexes), x=x_axis_scaled)
            if self.isdata["green"]:
                self.crosshair_H_green.setData(y=data_green.__getitem__(H_indexes), x=x_axis_scaled)
            if self.isdata["red"]:
                self.crosshair_H_red.setData(y=data_red.__getitem__(H_indexes), x=x_axis_scaled)

            if self.isdata["blue"]:
                self.crosshair_V_blue.setData(y=y_axis_scaled, x=data_blue.__getitem__(V_indexes))
            if self.isdata["green"]:
                self.crosshair_V_green.setData(y=y_axis_scaled, x=data_green.__getitem__(V_indexes))
            if self.isdata["red"]:
                self.crosshair_V_red.setData(y=y_axis_scaled, x=data_red.__getitem__(V_indexes))

        if self.isdata["spread"]:
            data_H_indexes = slice(None, None, 1)
            data_V_indexes = slice(None, None, 1)

            posx_adpative, posy_adpative = self.mapfromview('spread', posx, posy)
            points, data = self.img_spread.get_points_at(axis='y', val=posy_adpative)
            x_sorted_indexes = np.argsort(points[:, 0])
            x = points[x_sorted_indexes, 0][data_H_indexes]
            xscaled, yscaled = self.scale_axis(x, x)
            self.crosshair_H_spread.setData(x=xscaled,
                                               y=data[x_sorted_indexes][data_H_indexes])
            points, data = self.img_spread.get_points_at(axis='x', val=posx_adpative)
            y_sorted_indexes = np.argsort(points[:, 1])
            y = points[y_sorted_indexes, 1][data_V_indexes]
            xscaled, yscaled = self.scale_axis(y, y)
            self.crosshair_V_spread.setData(y=yscaled,
                                               x=data[y_sorted_indexes][data_V_indexes])

    @Slot(bool)
    def show_hide_crosshair(self, show=True):
        range = self.image_widget.view.viewRange()
        self.set_crosshair_position(np.mean(np.array(range[0])), np.mean(np.array(range[0])))
        if show:
            for curve_key in IMAGE_TYPES:
                self.view.show_crosshair_curve(curve_key, self.isdata[curve_key])

            self.update_crosshair_data(*self.get_crosshair_position())

            QtWidgets.QApplication.processEvents()
            self.show_lineouts()


    def ini_plot(self):
        for k in self.data_integrated_plot.keys():
            self.data_integrated_plot[k] = np.zeros((2, 1))

    @Slot(dict)
    def roi_changed(self, ROIs):
        if self.data is None:
            return
        self.data_to_export['data0D'] = OrderedDict([])
        self.data_to_export['data1D'] = OrderedDict([])
        self.measure_data_dict = OrderedDict([])

        for roi_key in ROIs:
            roi_lineout_dict = dict(roi_key=self.filter.get_xydata_from_roi(roi_key, ROIs[roi_key]))

        self.roi_lineout_signal.emit(roi_lineout_dict)

    def set_data(self, data_red=None, data_green=None, data_blue=None, data_spread=None):
        try:
            self.isdata['red'] = data_red is not None
            self.isdata['green'] = data_green is not None
            self.isdata['blue'] = data_blue is not None
            self.isdata['spread'] = data_spread is not None

            self.data = dict(zip(IMAGE_TYPES, [data_red, data_green , data_blue, data_spread]))
            self.data_to_show_signal.emit(self.data)
            self.filter.data = self.data



            if data_red is not None:  # if only one data only the red canal should be set otherwise this would fail
                self._x_axis = np.linspace(0, data_red.shape[1] - 1, data_red.shape[1])
                self._y_axis = np.linspace(0, data_red.shape[0] - 1, data_red.shape[0])
                self.x_axis_scaled, self.y_axis_scaled = self.scale_axis(self._x_axis, self._y_axis)

            ind = 0
            symautolevel = self.auto_levels_action_sym.isChecked()
            if self.isdata['red']:
                self.data_to_export['data2D']['CH{:03d}'.format(ind)] = \
                    utils.DataToExport(data=data_red, source='raw',
                                       x_axis=utils.Axis(data=self.x_axis_scaled,
                                                         units=self.scaling_options['scaled_xaxis']['units'],
                                                         label=self.scaling_options['scaled_xaxis']['label']),
                                       y_axis=utils.Axis(data=self.y_axis_scaled,
                                                         units=self.scaling_options['scaled_yaxis']['units'],
                                                         label=self.scaling_options['scaled_yaxis']['label']))
                ind += 1

            if self.isdata['green']:
                self.data_to_export['data2D']['CH{:03d}'.format(ind)] = \
                    utils.DataToExport(data=data_green, source='raw',
                                       x_axis=utils.Axis(data=self.x_axis_scaled,
                                                         units=self.scaling_options['scaled_xaxis']['units'],
                                                         label=self.scaling_options['scaled_xaxis']['label']),
                                       y_axis=utils.Axis(data=self.y_axis_scaled,
                                                         units=self.scaling_options['scaled_yaxis']['units'],
                                                         label=self.scaling_options['scaled_yaxis']['label']))
                ind += 1

            if self.isdata['blue']:
                self.data_to_export['data2D']['CH{:03d}'.format(ind)] = \
                    utils.DataToExport(data=data_blue, source='raw',
                                       x_axis=utils.Axis(data=self.x_axis_scaled,
                                                         units=self.scaling_options['scaled_xaxis']['units'],
                                                         label=self.scaling_options['scaled_xaxis']['label']),
                                       y_axis=utils.Axis(data=self.y_axis_scaled,
                                                         units=self.scaling_options['scaled_yaxis']['units'],
                                                         label=self.scaling_options['scaled_yaxis']['label']))
                ind += 1
            if self.isdata['spread']:
                self.data_to_export['data2D']['CH{:03d}'.format(ind)] = \
                    utils.DataToExport(data=data_spread[:, 2], source='raw',
                                       x_axis=utils.Axis(data=data_spread[:, 0],
                                                         units=self.scaling_options['scaled_xaxis']['units'],
                                                         label=self.scaling_options['scaled_xaxis']['label']),
                                       y_axis=utils.Axis(data=data_spread[:, 1],
                                                         units=self.scaling_options['scaled_yaxis']['units'],
                                                         label=self.scaling_options['scaled_yaxis']['label']))
                ind += 1

            if self.roi_action.isChecked():
                self.roi_changed()
            else:
                self.data_to_export['acq_time_s'] = datetime.datetime.now().timestamp()
                self.data_to_export_signal.emit(self.data_to_export)

            if self.isocurve_action.isChecked() and self.isdata['red']:
                self._isocurve_item.setData(pg.gaussianFilter(data_red, (2, 2)))

            if self.crosshair_action.isChecked():
                self.crosshairChanged()

        except Exception as e:
            print(e)

    def show_data_temp(self, data_red=None, data_green=None, data_blue=None, data_spread=None):
        """
        to plot temporary data, for instance when all pixels are not yet populated...
        """

        self.raw_data = dict(blue=data_blue, green=data_green, red=data_red, spread=data_spread)
        red_flag = data_red is not None
        self.isdata['red'] = red_flag
        green_flag = data_green is not None
        self.isdata['green'] = green_flag
        blue_flag = data_blue is not None
        self.isdata['blue'] = blue_flag
        spread_flag = data_spread is not None
        self.isdata['spread'] = spread_flag

        data_red, data_blue, data_green = self.set_image_transform()
        self.set_visible_items()

        for key in IMAGE_TYPES:
            if self.isdata[key]:
                self.view.image_items[key].setImage(self.raw_data[key],
                                                    autoLevels=self.autolevels, symautolevel=self.autolevels_sym)

    def mapfromview(self, graphitem, x, y):
        """
        get item coordinates from view coordinates
        Parameters
        ----------
        graphitem: (str or GraphItem) either 'red', 'blue', 'green' or 'spread' referring to their corresponding
            graphitem (self._image_items['red'])...
        x: (float) x oordinate in the view reference frame
        y: (float) y coordinate in the view refernece frame

        Returns
        -------
        x: (float) coordinate in the item reference frame
        y: (float) coordinate in the item reference frame
        """
        if isinstance(graphitem, str):
            if graphitem not in ('red', 'blue', 'green', 'spread'):
                return None
            graphitem = getattr(self, f'img_{graphitem}')
        point = graphitem.mapFromView(QtCore.QPointF(x, y))
        return point.x(), point.y()

    def setObjectName(self, txt):
        self.parent.setObjectName(txt)

    def show_hide_histogram_with_data(self):
        for key in self.view.histograms:
            if self.isdata[key] and self.view.actions[key].isChecked():
                self.view.histograms[key].setVisible(self.view.actions['histo'].isChecked())
                self.view.histograms[key].setLevels(self.raw_data[key].min(), self.raw_data[key].max())
        QtWidgets.QApplication.processEvents()


    def update_image(self):
        self.setImageTemp(data_red=self.raw_data['red'], data_green=self.raw_data['green'],
                          data_blue=self.raw_data['blue'], data_spread=self.raw_data['spread'])



    def update_crosshair_data(self, posx, posy, name=""):
        try:
            (posx_scaled, posy_scaled) = self.scale_axis(posx, posy)
            self.crosshair_dragged.emit(posx_scaled, posy_scaled)

            # if self._x_axis is not None:
            #     x_axis_scaled, y_axis_scaled = self.scale_axis(self._x_axis, self._y_axis)
            #     indx = utils.find_index(self._x_axis, posx)[0][0]
            #     indy = utils.find_index(self._y_axis, posy)[0][0]
            # else: #case of spread data only
            #     indx, indy = (posx_scaled, posy_scaled)

            self.crosshairChanged(posx, posy)
            dat = f'({posx_scaled:.1e}, {posy_scaled:.1e})'
            if self.isdata['red']:
                indx, indy = self.mapfromview('red', posx, posy)
                z_red = self.transform_image(self.raw_data['red'])[utils.rint(indy), utils.rint(indx)]
                dat += f' r={z_red:.1e},'

            if self.isdata['green']:
                indx, indy = self.mapfromview('green', posx, posy)
                z_green = self.transform_image(self.raw_data['green'])[utils.rint(indy), utils.rint(indx)]
                dat += f' g={z_green:.1e},'

            if self.isdata['blue']:
                indx, indy = self.mapfromview('blue', posx, posy)
                z_blue = self.transform_image(self.raw_data['blue'])[utils.rint(indy), utils.rint(indx)]
                dat += f' b={z_blue:.1e},'

            if self.isdata['spread']:
                z_spread = self.img_spread.get_val_at(self.mapfromview('spread', posx, posy))
                dat += f' s={z_spread:.1e},'

            self._actions['position'].setText(dat)

        except Exception as e:
            print(e)

def main_controller():

    app = QtWidgets.QApplication(sys.argv)
    form = QtWidgets.QWidget()

    Nx = 100
    Ny = 200
    data_random = np.random.normal(size=(Ny, Nx))
    x = np.linspace(0, Nx - 1, Nx)
    y = np.linspace(0, Ny - 1, Ny)
    from pymodaq.daq_utils.daq_utils import gauss2D

    data_red = 3 * gauss2D(x, 0.2 * Nx, Nx / 5, y, 0.3 * Ny, Ny / 5, 1, 90) * np.sin(x/5)**2
    # data_red = pg.gaussianFilter(data_red, (2, 2))
    data_green = 3 * gauss2D(x, 0.2 * Nx, Nx / 5, y, 0.3 * Ny, Ny / 5, 1, 0)
    # data_green = pg.gaussianFilter(data_green, (2, 2))
    data_blue = data_random + 3 * gauss2D(x, 0.7 * Nx, Nx / 5, y, 0.2 * Ny, Ny / 5, 1)
    data_blue = pg.gaussianFilter(data_blue, (2, 2))

    prog = Viewer2D(form)
    # prog.set_axis_scaling(scaling_options=utils.ScalingOptions(
    #     scaled_xaxis=utils.ScaledAxis(label="eV", units=None, offset=100, scaling=0.1),
    #     scaled_yaxis=utils.ScaledAxis(label="time", units='s', offset=-20, scaling=2)))
    form.show()
    #prog.auto_levels_action_sym.trigger()
    #prog.view.actions['auto_levels'].trigger()

    # data = np.load('triangulation_data.npy')
    prog.show_data(data_red=data_red, data_blue=data_blue, )
    # prog.setImage(data_spread=data)
    #app.processEvents()

    sys.exit(app.exec_())



def main_view():
    app = QtWidgets.QApplication(sys.argv)
    form = QtWidgets.QWidget()
    prog = View2D(form)
    form.show()
    sys.exit(app.exec_())

if __name__ == '__main__':  # pragma: no cover

    #main_view()
    main_controller()
