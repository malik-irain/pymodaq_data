# -*- coding: utf-8 -*-
"""
Created the 28/10/2022

@author: Sebastien Weber
"""
import logging
import numpy as np
import pint.errors
from pint.errors import DimensionalityError
import pytest
from pytest import approx, mark
import time

import pymodaq_data
from pymodaq_utils import math_utils as mutils
from pymodaq_utils.units import nm2eV, eV2nm
from pymodaq_data import data as data_mod
from pymodaq_data.data import DataDim
from pymodaq_data.post_treatment.process_to_scalar import DataProcessorFactory

data_processors = DataProcessorFactory()

LABEL = 'A Label'
UNITS = 'units'
OFFSET = -20.4
SCALING = 0.22
SIZE = 20
DATA = OFFSET + SCALING * np.linspace(0, SIZE-1, SIZE)

DATA0D = np.array([2.7])
DATA1D = np.arange(0, 10)
DATA2D = np.arange(0, 5*6).reshape((5, 6))
XAXIS_GAUSSIAN = np.linspace(0, 128, endpoint=False)
X0 = 100
DX = 20
YAXIS_GAUSSIAN = np.linspace(0, 128, endpoint=False)
Y0 = 79
DY = 12
GAUSSIAN_2D = mutils.gauss2D(YAXIS_GAUSSIAN, Y0, DY,
                             XAXIS_GAUSSIAN, X0, DX)
DATAND = np.arange(0, 5 * 6 * 3).reshape((5, 6, 3))
REAL_UNITS = 'm'
Nn0 = 10
Nn1 = 5

DWA_RAW = data_mod.DataRaw('raw', units='s', data=[DATA1D, DATA1D])

def init_axis(data=None, index=0, label=LABEL) -> data_mod.Axis:
    if data is None:
        data = DATA
    return data_mod.Axis(label=label, units=UNITS, data=data, index=index)


def init_data(data=None, Ndata=1, axes=[], name='myData', source=data_mod.DataSource.raw,
              labels=None, units='') -> data_mod.DataWithAxes:
    if data is None:
        data = DATA2D
    return data_mod.DataWithAxes(name, units=units, source=source,
                                 data=[data for ind in range(Ndata)],
                                 axes=axes, labels=labels)

def init_dataND():
    N0 = 5
    N1 = 6
    N2 = 3
    DATAND = np.arange(0, N0 * N1 * N2).reshape((N0, N1, N2))

    axis0 = data_mod.Axis(label='myaxis0', data=np.linspace(0, N0 - 1, N0), index=0)
    axis1 = data_mod.Axis(label='myaxis1', data=np.linspace(0, N1 - 1, N1), index=1)
    axis2 = data_mod.Axis(label='myaxis2', data=np.linspace(0, N2 - 1, N2), index=2)
    return data_mod.DataWithAxes('mydata', source='raw', data=[DATAND], axes=[axis0, axis1, axis2],
                                 nav_indexes=(0, 1)), \
           (N0, N1, N2)


@pytest.fixture()
def init_data_spread():
    Nspread = 21
    sig_axis = data_mod.Axis(label='signal', index=1, data=np.linspace(0, DATA1D.size - 1, DATA1D.size))
    nav_axis_0 = data_mod.Axis(label='nav0', index=0, data=np.random.rand(Nspread), spread_order=0)
    nav_axis_1 = data_mod.Axis(label='nav1', index=0, data=np.random.rand(Nspread), spread_order=1)

    data_array = np.array([ind / Nspread * DATA1D for ind in range(Nspread)])
    data = data_mod.DataRaw('mydata', distribution='spread', data=[data_array], nav_indexes=(0,),
                            axes=[nav_axis_0, sig_axis, nav_axis_1])
    return data, data_array, sig_axis, nav_axis_0, nav_axis_1, Nspread


@pytest.fixture()
def init_spread_data_arrays():
    return ([np.random.randn(5), np.random.randn(5)],  # 0D spread arrays
            [np.random.randn(5, 4), np.random.randn(5, 4)],  # 1D spread arrays
            [np.random.randn(5, 2, 4),np.random.randn(5, 2, 4)],  # 2D spread arrays
            )


@pytest.fixture()
def init_data_uniform() -> data_mod.DataWithAxes:

    nav_axis_0 = data_mod.Axis(label='nav0', index=0, data=np.linspace(0, Nn0 - 1, Nn0))
    nav_axis_1 = data_mod.Axis(label='nav1', index=1, data=np.linspace(0, Nn1 - 1, Nn1))
    sig_axis_0 = data_mod.Axis(label='signal0', index=2,
                               data=np.linspace(0, DATA2D.shape[0] - 1, DATA2D.shape[0]))
    sig_axis_1 = data_mod.Axis(label='signal1', index=3,
                               data=np.linspace(0, DATA2D.shape[1] - 1, DATA2D.shape[1]))

    data_array = np.ones((Nn0, Nn1, DATA2D.shape[0], DATA2D.shape[1]))
    data = data_mod.DataRaw('mydata', data=[data_array], nav_indexes=(0, 1),
                            axes=[nav_axis_0, sig_axis_0, nav_axis_1, sig_axis_1])
    return data


@pytest.fixture()
def init_axis_fixt():
    return init_axis()


@pytest.fixture()
def init_data_fixt():
    return init_data()


@pytest.fixture()
def ini_data_to_export():
    dat1 = init_data(data=DATA2D, Ndata=2, name='data2D')
    dat2 = init_data(data=DATA1D, Ndata=3, name='data1D')
    data = data_mod.DataToExport(name='toexport', data=[dat1, dat2])
    return dat1, dat2, data


class TestDataDim:
    def test_index(self):
        dim = DataDim['Data0D']
        assert dim.dim_index == 0
        dim = DataDim['Data1D']
        assert dim.dim_index == 1
        dim = DataDim['Data2D']
        assert dim.dim_index == 2
        dim = DataDim['DataND']
        assert dim.dim_index == 3

    @pytest.mark.parametrize('dim1', ['Data0D', 'Data1D', 'Data2D', 'DataND'])
    @pytest.mark.parametrize('dim2', ['Data0D', 'Data1D', 'Data2D', 'DataND'])
    def test_comparison(self, dim1: str, dim2: str):
        assert (DataDim[dim1] < DataDim[dim2]) == (DataDim[dim1].dim_index < DataDim[dim2].dim_index)
        assert (DataDim[dim1] <= DataDim[dim2]) == (DataDim[dim1].dim_index <= DataDim[dim2].dim_index)
        assert (DataDim[dim1] > DataDim[dim2]) == (DataDim[dim1].dim_index > DataDim[dim2].dim_index)
        assert (DataDim[dim1] >= DataDim[dim2]) == (DataDim[dim1].dim_index >= DataDim[dim2].dim_index)


class TestAxis:

    def test_units(self):
        NOT_A_STRING = 45
        with pytest.raises(TypeError):
            axis = data_mod.Axis('axis', units=NOT_A_STRING, data=np.array([0, 1]))

        axis = data_mod.Axis('axis', units='unknown units', data=np.array([0, 1]))
        assert axis.units == ''

    def test_units_as(self):
        eVs = np.array([1, 1.5, 2])
        axis = data_mod.Axis('wavelength', units='eV', data=eVs)
        axis.units_as('nm', inplace=True, context='spectroscopy')
        assert axis.units == 'nm'
        assert np.allclose(axis.get_data(), eV2nm(eVs))

        axis_again = data_mod.Axis('wavelength', units='eV', data=eVs)
        new_axis = axis_again.units_as('nm', inplace=False, context='spectroscopy')
        assert new_axis.units == 'nm'
        assert np.allclose(new_axis.get_data(), eV2nm(eVs))
        assert axis_again == data_mod.Axis('wavelength', units='eV', data=eVs)

        axis_base = new_axis.to_base_units()
        assert axis_base.units == 'm'
        assert np.allclose(axis_base.get_data(), eV2nm(eVs) * 1e-9)

    def test_errors(self):
        with pytest.raises(TypeError):
            data_mod.Axis(label=24)
        with pytest.raises(TypeError):
            data_mod.Axis(units=42)
        with pytest.raises(TypeError):
            data_mod.Axis(index=1.)
        with pytest.raises(ValueError):
            data_mod.Axis(index=-2)

    def test_attributes(self, init_axis_fixt):
        ax = init_axis_fixt
        assert hasattr(ax, 'label')
        assert ax.label == LABEL
        assert hasattr(ax, 'units')
        assert ax.units == UNITS
        assert hasattr(ax, 'index')
        assert ax.index == 0
        assert hasattr(ax, 'data')
        assert ax.data is None
        assert ax.offset == pytest.approx(OFFSET)
        assert ax.scaling == pytest.approx(SCALING)

        data_tmp = np.array([0.1, 2, 23, 44, 21, 20])  # non linear axis
        ax = init_axis(data=data_tmp)
        assert np.all(ax.data == pytest.approx(data_tmp))
        assert ax.offset is None
        assert ax.scaling is None

    def test_operation(self, init_axis_fixt):
        scale = 2
        offset = 100
        ax = init_axis_fixt
        ax_scaled = ax * scale
        ax_offset = ax + offset
        assert isinstance(ax_scaled, data_mod.Axis)
        assert isinstance(ax_offset, data_mod.Axis)

        assert ax_scaled.scaling == approx(ax.scaling * scale)
        assert ax_offset.offset == approx(ax.offset + offset)

        data_tmp = np.array([0.1, 2, 23, 44, 21, 20])  # non linear axis
        ax = init_axis(data=data_tmp)
        ax_scaled = ax * scale
        ax_offset = ax + offset
        assert ax_scaled.data == approx(ax.data * scale)
        assert ax_offset.data == approx(ax.data + offset)

    def test_math(self, init_axis_fixt):
        ax = init_axis_fixt
        assert ax.mean() == pytest.approx(OFFSET + SIZE / 2 * SCALING)
        assert ax.min() == pytest.approx(OFFSET)
        assert ax.max() == pytest.approx(OFFSET + SIZE * SCALING)

        data_tmp = np.array([0.1, 2, 23, 44, 21, 20])  # non linear axis
        ax = init_axis(data=data_tmp)
        assert ax.mean() == pytest.approx(np.mean(data_tmp))
        assert ax.min() == pytest.approx(np.min(data_tmp))
        assert ax.max() == pytest.approx(np.max(data_tmp))

    def test_find_index(self, init_axis_fixt):
        ax = init_axis_fixt
        assert ax.find_index(0.01 + OFFSET + 4 * SCALING) == 4

        data_tmp = np.array([0.1, 2, 23, 44, 21, 20])  # non linear axis
        ax = init_axis(data=data_tmp)
        assert ax.find_index(5) == 1

    def test_slice_getter(self, init_axis_fixt):
        ax = init_axis_fixt

        ellipsis_axis = ax.iaxis[...]
        ellipsis_axis_value = ax.vaxis[...]
        assert isinstance(ellipsis_axis, data_mod.Axis)
        assert ellipsis_axis == ax
        assert ellipsis_axis_value == ax

        ind_start = 2
        ind_end = 10
        sliced_axis = ax.iaxis[ind_start:ind_end]
        assert isinstance(sliced_axis, data_mod.Axis)
        assert len(sliced_axis) == ind_end - ind_start
        assert np.allclose(sliced_axis.get_data(), ax.get_data()[ind_start:ind_end])

        sliced_axis_value = ax.vaxis[ax.get_data()[ind_start]:ax.get_data()[ind_end]]
        assert sliced_axis == sliced_axis_value

        ind_int = 3
        int_axis = ax.iaxis[ind_int]
        int_axis_value = ax.vaxis[ax.get_data()[ind_int]]

        assert isinstance(int_axis, data_mod.Axis)
        assert len(int_axis) == 1
        assert int_axis.get_data()[0] == ax.get_data()[ind_int]
        assert int_axis == int_axis_value

    def test_slice_setter(self, init_axis_fixt):
        ax = init_axis_fixt
        length = len(ax)
        ind_start = 2
        ind_end = 10
        axis_to_put_in = data_mod.Axis('replace', data=mutils.linspace_step(ind_start-10, ind_end-1-10, 1))

        ax.iaxis[ind_start:ind_end] = axis_to_put_in

        assert np.allclose(ax.data[ind_start:ind_end], axis_to_put_in.get_data())
        assert len(ax) == length

        ax.iaxis[ind_start:ind_end] = axis_to_put_in.get_data()

        assert np.allclose(ax.data[ind_start:ind_end], axis_to_put_in.get_data())
        assert len(ax) == length

    def test_get_data(self):
        # spread axis
        DATA = np.array([0, 1, 6, 8, 9])
        axis = init_axis(DATA)
        assert axis.data is not None
        assert axis.offset is None
        assert axis.scaling is None
        assert np.allclose(axis.get_data(), DATA)

        # linear axis
        DATA = np.array([1, 2, 3, 4])
        axis = init_axis(DATA)
        assert axis.data is None
        assert axis.offset == 1
        assert axis.scaling == 1
        assert np.allclose(axis.get_data(), DATA)

    def test_get_data_at(self):
        DATA = np.array([0, 1, 6, 8, 9])
        axis = init_axis(DATA)
        INDEX = 2
        INDEXES = (0, 3, 4)
        INDEXES_array = np.array(INDEXES)
        SLICE = slice(0, None, 2)
        assert np.allclose(axis.get_data_at(INDEX), DATA[INDEX])
        assert np.allclose(axis.get_data_at(INDEXES), DATA[np.array(INDEXES)])
        assert np.allclose(axis.get_data_at(INDEXES_array), DATA[INDEXES_array])
        assert np.allclose(axis.get_data_at(SLICE), DATA[SLICE])

class TestDataLowLevel:
    def test_init(self):
        data = data_mod.DataLowLevel('myData')
        assert hasattr(data, 'timestamp')
        assert hasattr(data, 'name')
        assert data.name == 'myData'

    def test_timestamp(self):
        t1 = time.time()
        data = data_mod.DataLowLevel('myData')
        t2 = time.time()
        assert t1 <= data.timestamp <= t2


class TestDataBase:
    def test_init(self):
        Ndata = 2
        data = data_mod.DataBase('myData', source=data_mod.DataSource(0), data=[DATA2D for ind in range(Ndata)])
        assert data.length == Ndata
        assert data.dim == data_mod.DataDim['Data2D']
        assert data.shape == DATA2D.shape
        assert data.size == DATA2D.size
        assert hasattr(data, 'timestamp')

    def test_labels(self):
        Ndata = 3
        labels = ['label0', 'label1']
        labels_auto = labels + ['CH02']
        data = data_mod.DataBase('myData', source=data_mod.DataSource(0), data=[DATA2D for ind in range(Ndata)])
        assert len(data.labels) == Ndata
        assert data.labels == [f'CH{ind:02d}' for ind in range(Ndata)]

        data = data_mod.DataBase('myData',
                                 source=data_mod.DataSource(0), data=[DATA2D for ind in range(Ndata)],
                                 labels=labels)
        assert len(data.labels) == Ndata
        assert data.labels == labels_auto

    def test_data_source(self):
        data = data_mod.DataBase('myData', source='calculated',  data=[DATA2D])
        assert data.source == data_mod.DataSource['calculated']

    @mark.parametrize("data_array, datadim", [(DATA0D, 'Data0D'), (DATA0D, data_mod.DataDim['Data0D']),
                                              (DATA0D, 'Data2D')])
    def test_get_dim(self, data_array, datadim):
        data = data_mod.DataBase('myData',
                                 source=data_mod.DataSource(0), data=[data_array], dim=datadim)
        assert data.dim == data_mod.DataDim['Data0D']  # force dim to reflect datashape

    def test_force_dim(self):
        data = data_mod.DataBase('myData',
                                 source=data_mod.DataSource(0), data=[DATA1D], dim='Data0D')
        assert data.dim.name == 'Data1D'

    def test_errors(self):
        with pytest.raises(TypeError):
            data_mod.DataBase()
        with pytest.raises(ValueError):
            data_mod.DataBase('myData')
        with pytest.raises(TypeError):
            data_mod.DataBase('myData', source=data_mod.DataSource(0))
        data = data_mod.DataBase('myData', source=data_mod.DataSource(0), data=DATA2D)  # only a ndarray
        assert len(data) == 1
        assert np.all(data.data[0] == approx(DATA2D))
        data = data_mod.DataBase('myData', source=data_mod.DataSource(0), data=12.4)  # only a numeric
        assert len(data) == 1
        assert isinstance(data.data[0], np.ndarray)
        assert data.data[0][0] == approx(12.4)

        with pytest.raises(data_mod.DataShapeError):
            data_mod.DataBase('myData', source=data_mod.DataSource(0), data=[DATA2D, DATA0D])  # list of different ndarray shape length
        with pytest.raises(TypeError):
            data_mod.DataBase('myData', source=data_mod.DataSource(0), data=['12', 5])  # list of non ndarray

    def test_dunder(self):
        LENGTH = 24
        data = init_data(DATA0D, Ndata=LENGTH)
        assert len(data) == LENGTH
        count = 0
        for dat in data:
            assert np.all(dat == pytest.approx(DATA0D))
            count += 1
        assert count == len(data)

        assert np.all(DATA0D == pytest.approx(data[-1]))

        with pytest.raises(IndexError):
            data['str']
        with pytest.raises(IndexError):
            data[len(data) + 1]

    def test_comparison_units(self):

        dwa_mm = data_mod.DataRaw('dwa', units='mm', data=[np.array([0, 1, 2])])
        dwa_um = data_mod.DataRaw('dwa', units='um', data=[np.array([0, 1, 2])])
        dwa_um_equal = data_mod.DataRaw('dwa', units='um', data=[np.array([0, 1, 2]) * 1000])
        dwa_unrelated = data_mod.DataRaw('dwa', units='', data=[np.array([0, 1, 2])])

        assert dwa_mm != dwa_um
        assert dwa_mm == dwa_um_equal
        assert dwa_mm != dwa_unrelated

    def test_maths(self):
        data = init_data(data=DATA2D, Ndata=2)
        data1 = init_data(data=DATA2D, Ndata=2)

        data_sum = data + data1
        for ind_data in range(len(data)):
            assert np.all(data_sum[ind_data] == pytest.approx(2 * DATA2D))
        data_diff = data - data1
        for ind_data in range(len(data)):
            assert np.all(data_diff[ind_data] == pytest.approx(0 * DATA2D))

        data_mult = data * 0.85
        for ind_data in range(len(data)):
            assert np.all(data_mult[ind_data] == pytest.approx(0.85 * DATA2D))

        data_div = data / 0.85
        for ind_data in range(len(data)):
            assert np.all(data_div[ind_data] == pytest.approx(DATA2D/.85))

    def test_multiply(self):
        units = 'm'
        data = init_data(data=DATA1D, Ndata=2, units=units)
        units1 = 'ms'
        data1 = init_data(data=DATA1D, Ndata=2, units=units1)
        number = 24.5
        array = DATA1D * 0.01

        data_number = data * number
        for ind in range(len(data_number)):
            assert np.allclose(data_number[ind], data[ind] * number)
        assert data_number.units == units

        data_array = data * array
        for ind in range(len(data_array)):
            assert np.allclose(data_array[ind], data[ind] * array)
        assert data_array.units == units

        data_data = data * data1
        for ind in range(len(data_data)):
            assert np.allclose(data_mod.Q_(data_data[ind], data_data.units),
                               data_mod.Q_(data[ind], data.units) *
                               data_mod.Q_(data1[ind], data1.units))
        assert data_mod.Unit(data_data.units).is_compatible_with('m * s')

    def test_abs(self):
        data_p = init_data(data=DATA2D, Ndata=2)
        data_m = init_data(data=-DATA2D, Ndata=2)

        assert data_p.abs() == data_p
        assert data_m.abs() == data_p

    def test_average(self):
        WEIGHT = 5
        FRAC = 0.23
        data = init_data(data=DATA2D, Ndata=2)
        data1 = init_data(data=-DATA2D, Ndata=2)

        assert data.average(data1, 1) == data * 0
        assert data.average(data, 1) == data
        assert data.average(data, 2) == data

    def test_append(self):
        Ndata = 2
        labels = [f'label{ind}' for ind in range(Ndata)]
        Ndatabis = 3
        label_bis = [f'labelbis{ind}' for ind in range(Ndatabis)]
        data = init_data(data=DATA1D, Ndata=Ndata, labels=labels)
        data_bis = init_data(data=DATA1D, Ndata=Ndatabis, labels=label_bis)

        data.append(data_bis)
        assert len(data) == Ndata + Ndatabis
        assert data.labels == labels + label_bis


class TestDataWithAxesUniform:
    def test_init(self):
        Ndata = 2
        data = data_mod.DataWithAxes('myData', source=data_mod.DataSource(0),
                                     data=[DATA2D for ind in range(Ndata)])
        assert data.axes == []

    def test_axis(self):
        with pytest.raises(TypeError):
            init_data(DATA2D, 2, axes=[np.linspace(0, 10)])

    @mark.parametrize('index', [-1, 0, 1, 3])
    def test_axis_index(self, index):
        with pytest.raises(ValueError):
            init_data(DATA2D, 2, axes=[init_axis(np.zeros((DATA2D.shape[1],)), -1)])
        index = 0
        data = init_data(DATA2D, 2, axes=[init_axis(np.zeros((10,)), index)])
        assert len(data.get_axis_from_index(0)[0]) == DATA2D.shape[index]

        index = 1
        data = init_data(DATA2D, 2, axes=[init_axis(np.zeros((DATA2D.shape[index],)), index)])
        assert len(data.get_axis_from_index(index)[0]) == DATA2D.shape[index]

        with pytest.raises(IndexError):
            init_data(DATA2D, 2, axes=[init_axis(np.zeros((DATA2D.shape[1],)), 2)])

    def test_get_shape_from_index(self):
        index = 1
        data = init_data(DATA2D, 2, axes=[init_axis(np.zeros((DATA2D.shape[index],)), index)])

        assert data.axes_manager.get_shape_from_index(0) == DATA2D.shape[0]
        assert data.axes_manager.get_shape_from_index(1) == DATA2D.shape[1]
        with pytest.raises(IndexError):
            data.axes_manager.get_shape_from_index(-1)
        with pytest.raises(IndexError):
            data.axes_manager.get_shape_from_index(2)

    def test_get_axis_from_dim(self):

        index1 = 1
        axis = init_axis(np.zeros((DATA2D.shape[index1],)), index1)
        data = init_data(DATA2D, 2, axes=[axis])

        assert data.get_axis_from_index(index1)[0] == axis
        index0 = 0
        axis = data.get_axis_from_index(index0)[0]
        assert axis is None
        axis = data.get_axis_from_index(index0, create=True)[0]
        assert len(axis) == data.axes_manager.get_shape_from_index(index0)

    def test_deep_copy_with_new_data(self):
        data, shape = init_dataND()
        #on nav_indexes
        IND_TO_REMOVE = 1
        new_data = data.deepcopy_with_new_data([np.squeeze(dat[:, 0, :]) for dat in data.data],
                                               remove_axes_index=IND_TO_REMOVE)

        assert new_data.nav_indexes == (0,)
        assert new_data.sig_indexes == (1,)
        assert new_data.shape == np.squeeze(data.data[0][:, 0, :]).shape
        assert len(new_data.get_axis_indexes()) == 2
        assert data.get_axis_from_index(IND_TO_REMOVE)[0] not in new_data.axes

        # on sig_indexes
        IND_TO_REMOVE = 2
        new_data = data.deepcopy_with_new_data([np.squeeze(dat[:, :, 0]) for dat in data.data],
                                               remove_axes_index=IND_TO_REMOVE)

        assert new_data.nav_indexes == (0, 1)
        assert new_data.sig_indexes == ()
        assert new_data.shape == np.squeeze(data.data[0][:, :, 0]).shape
        assert len(new_data.get_axis_indexes()) == 2
        assert data.get_axis_from_index(IND_TO_REMOVE)[0] not in new_data.axes


        # on nav_indexes both axes
        IND_TO_REMOVE = [0, 1]
        new_data = data.deepcopy_with_new_data([np.squeeze(dat[0, 0, :]) for dat in data.data],
                                               remove_axes_index=IND_TO_REMOVE)

        assert new_data.nav_indexes == ()
        assert new_data.sig_indexes == (0,)
        assert new_data.shape == np.squeeze(data.data[0][0, 0, :]).shape
        assert len(new_data.get_axis_indexes()) == 1
        for ind in IND_TO_REMOVE:
            assert data.get_axis_from_index(ind)[0] not in new_data.axes

        # on all axes
        IND_TO_REMOVE = [0, 1, 2]
        new_data = data.deepcopy_with_new_data([np.atleast_1d(np.mean(dat)) for dat in data.data],
                                               remove_axes_index=IND_TO_REMOVE)

        assert new_data.nav_indexes == ()
        assert new_data.sig_indexes == (0,)
        assert new_data.shape == (1,)
        assert len(new_data.get_axis_indexes()) == 0
        for ind in IND_TO_REMOVE:
            assert data.get_axis_from_index(ind)[0] not in new_data.axes

    @pytest.mark.parametrize("IND_MEAN", [0, 1, 2])
    def test_mean(self, IND_MEAN):
        data, shape = init_dataND()
        data_ini = data.deepcopy()
        new_data = data.mean(IND_MEAN)
        for ind, dat in enumerate(data):
            assert np.all(new_data[ind] == pytest.approx(np.mean(dat, IND_MEAN)))
        assert len(new_data.get_axis_indexes()) == 2
        assert data.shape == data_ini.shape

    def test_sorted_uniform_1D(self):
        data_arrays = [np.array([10, 11, 12, 13, 14]), np.array([20, 21, 22, 23, 24])]
        axis_array = np.array([5, 2, 3, 4, 1])
        sorted_index = np.argsort(axis_array)
        data = data_mod.DataRaw('mydata', distribution='uniform',
                                data=data_arrays[:],
                                axes=[data_mod.Axis('axis', data=axis_array)])

        assert data.sort_data(axis_index=1, inplace=False) == data
        AXIS_INDEX = 0
        sorted_data = data.sort_data(axis_index=AXIS_INDEX)
        for ind in range(len(data)):
            assert np.allclose(sorted_data.data[ind],  data_arrays[ind][sorted_index])
        assert np.allclose(sorted_data.get_axis_from_index(AXIS_INDEX)[0].get_data(), axis_array[sorted_index])

    def test_sorted_uniform_2D_not_inplace(self):
        data_arrays = [np.array([[10, 11, 12, 13, 14],
                                [20, 21, 22, 23, 24],
                                [30, 31, 32, 33, 34],
                                ])]
        axis_0 = data_mod.Axis('axis_0', data=np.array([-10, -15, -20]), index=0)
        axis_1 = data_mod.Axis('axis_1', data=np.array([-10, -20, -30, -40, -50]), index=1)

        sorted_index_0 = np.argsort(axis_0.get_data())
        sorted_index_1 = np.argsort(axis_1.get_data())

        data = data_mod.DataRaw('mydata', distribution='uniform',
                                data=data_arrays[:],
                                axes=[axis_0, axis_1])

        sorted_data_0 = data.sort_data(axis_index=0)
        for ind in range(len(data)):
            assert np.allclose(sorted_data_0.data[ind],  data_arrays[ind][sorted_index_0, :])
        assert np.allclose(sorted_data_0.get_axis_from_index(0)[0].get_data(), axis_0.get_data()[sorted_index_0])

        sorted_data_1 = data.sort_data(axis_index=1)
        for ind in range(len(data)):
            assert np.allclose(sorted_data_1.data[ind],  data_arrays[ind][:, sorted_index_1])
        assert np.allclose(sorted_data_1.get_axis_from_index(1)[0].get_data(), axis_1.get_data()[sorted_index_1])

    def test_sorted_uniform_2D_inplace(self):
        data_arrays = [np.array([[10, 11, 12, 13, 14],
                                 [20, 21, 22, 23, 24],
                                 [30, 31, 32, 33, 34],
                                 ])]
        axis_0 = data_mod.Axis('axis_0', data=np.array([-10, -15, -20]), index=0)
        axis_1 = data_mod.Axis('axis_1', data=np.array([-10, -20, -30, -40, -50]), index=1)

        sorted_index_0 = np.argsort(axis_0.get_data())
        sorted_index_1 = np.argsort(axis_1.get_data())

        data = data_mod.DataRaw('mydata', distribution='uniform',
                                data=data_arrays[:],
                                axes=[axis_0.copy(), axis_1.copy()])

        sorted_data_0 = data.sort_data(axis_index=0, inplace=True)

        for ind in range(len(data)):
            assert np.allclose(sorted_data_0.data[ind],  data_arrays[ind][sorted_index_0, :])
        assert np.allclose(sorted_data_0.get_axis_from_index(0)[0].get_data(), axis_0.get_data()[sorted_index_0])

        sorted_data_1 = data.sort_data(axis_index=1)
        for ind in range(len(data)):
            assert np.allclose(sorted_data_1.data[ind],  sorted_data_0.data[ind][:, sorted_index_1])
        assert np.allclose(sorted_data_1.get_axis_from_index(1)[0].get_data(), axis_1.get_data()[sorted_index_1])
        assert np.allclose(sorted_data_0.get_axis_from_index(0)[0].get_data(), axis_0.get_data()[sorted_index_0])

    def test_interp(self):
        data_arrays = [np.array([11, 12, 14, 15]), np.array([21, 22, 24, 25])]
        data_arrays_expected = [np.array([11, 12, 13, 14, 15]), np.array([21, 22, 23, 24, 25])]
        axis_array = np.array([1, 2, 4, 5])
        new_axis_array = np.array([1, 2, 3, 4, 5])

        dwa = data_mod.DataRaw('mydata', distribution='uniform',
                               data=data_arrays[:],
                               axes=[data_mod.Axis('axis', data=axis_array)])

        dwa_interp = dwa.interp(new_axis_array)
        for ind in range(len(data_arrays_expected)):
            assert np.allclose(dwa_interp.data[ind], data_arrays_expected[ind])

        dwa = init_data(DATA0D)
        with pytest.raises(ValueError):
            dwa.interp(new_axis_array)

        dwa = init_data(DATA0D)
        with pytest.raises(ValueError):
            dwa.interp(new_axis_array)

    def test_ft_ift(self):
        omega0 = 5
        time_axis = data_mod.Axis('time', 's', data=np.linspace(0, 10*2*np.pi, 2**10))
        dwa = data_mod.DataRaw('sinus', data=[np.sin(omega0 * time_axis.get_data())], labels=['sinus'],
                               axes=[time_axis])

        dwa_fft = dwa.ft(0)
        dwa_processed = data_processors.get('argmax').process(dwa_fft.abs())
        assert dwa_processed.abs().data[0][0] == pytest.approx(omega0, 0.1)

        dwa_ift = dwa_fft.ift(0)
        assert np.allclose(dwa, dwa_ift.real())

        assert data_mod.Unit(dwa_ift.axes[0].units) == data_mod.Unit(dwa.axes[0].units)

    def test_fit(self):
        OMEGA0 = 5
        OFFSET = -4
        AMPLITUDE = 2
        PHI = 2 * np.pi /3

        OMEGA02 = 5
        OFFSET2 = -4
        AMPLITUDE2 = 2
        PHI2 = 2 * np.pi /6

        time_axis = data_mod.Axis('time', 's', data=np.linspace(0, 2 * 2 * np.pi, 2 ** 8))
        dwa = data_mod.DataRaw('sinus',
                               data=[OFFSET + AMPLITUDE * np.sin(OMEGA0 * time_axis.get_data() +
                                                                 PHI),
                                     OFFSET2 + AMPLITUDE2 * np.sin(OMEGA02 * time_axis.get_data() +
                                                                 PHI2)
                                     ],
                               labels=['sinus1', 'sinus2'],
                               axes=[time_axis])
        def my_sinus(x, a, offset, omega0, phi):
            return offset + a * np.sin(omega0 * x +phi)

        dwa_fit = dwa.fit(my_sinus, initial_guess=(AMPLITUDE, OFFSET, OMEGA0, PHI))
        for ind in range(len(dwa)):
            assert np.allclose(dwa_fit[ind], dwa[ind])

        assert hasattr(dwa_fit, 'fit_coeffs')

    def test_find_peaks(self):
        OMEGA0 = 5
        OFFSET = -4
        AMPLITUDE = 2
        PHI = np.pi /3

        OMEGA02 = 20
        OFFSET2 = 2
        AMPLITUDE2 = 1
        PHI2 = np.pi /6

        time_axis = data_mod.Axis('time', 's', data=np.linspace(0, 10 * 2 * np.pi, 2 ** 10))
        dwa = data_mod.DataRaw('sinus',
                               data=[OFFSET + AMPLITUDE * np.sin(OMEGA0 * time_axis.get_data() +
                                                                 PHI),
                                     OFFSET2 + AMPLITUDE2 * np.sin(OMEGA02 * time_axis.get_data() +
                                                                 PHI2)
                                     ],
                               axes=[time_axis])
        dwa_ft = dwa.ft()

        dte_peak = dwa.ft().abs().find_peaks(height=50)

        assert np.allclose(dte_peak[0].axes[0].get_data(), np.array([-OMEGA0, 0, OMEGA0]), atol=0.1)
        assert np.allclose(dte_peak[1].axes[0].get_data(), np.array([-OMEGA02, 0, OMEGA02]),
                           atol=0.1)


class TestNavIndexes:

    def test_set_nav_indexes(self):
        data, shape = init_dataND()

        assert data.shape == shape
        assert data.sig_indexes == (2,)

        data.nav_indexes = (1,)
        assert data.shape == shape
        assert data.nav_indexes == (1,)
        assert data.axes_manager.nav_indexes == (1,)
        assert data.sig_indexes == (0, 2)

        data.nav_indexes = ()
        assert data.sig_indexes == (0, 1,  2)


class TestDataWithAxesSpread:
    def test_init_data(self, init_data_spread):
        data, data_array, sig_axis, nav_axis_0, nav_axis_1, Nspread = init_data_spread
        dwa = data_mod.DataWithAxes(name='spread', source=data_mod.DataSource['raw'],
                                    dim=data_mod.DataDim['Data1D'],
                                    distribution=data_mod.DataDistribution['spread'],
                                    data=[data_array],
                                    nav_indexes=(0,),
                                    axes=[sig_axis, nav_axis_0, nav_axis_1])

        assert dwa.distribution.name == 'spread'
        assert dwa.inav[10].distribution.name == 'uniform'  # because the remangin signal data has uniform axes
        assert dwa.isig[5].distribution.name == 'spread'

    def test_nav_index(self, init_data_spread):
        data, data_array, sig_axis, nav_axis_0, nav_axis_1, Nspread = init_data_spread
        with pytest.raises(ValueError):
            data_mod.DataWithAxes(name='spread', source=data_mod.DataSource['raw'],
                                  dim=data_mod.DataDim['Data1D'],
                                  distribution=data_mod.DataDistribution['spread'],
                                  data=[data_array],
                                  nav_indexes=(0, 1),
                                  axes=[sig_axis, nav_axis_0, nav_axis_1])

    def test_nav_axis_length(self, init_data_spread):
        data, data_array, sig_axis, nav_axis_0, nav_axis_1, Nspread = init_data_spread
        nav_axis_1.data = np.concatenate((nav_axis_1.data, np.array([0.1,])))
        with pytest.raises(data_mod.DataLengthError):
            data_mod.DataWithAxes(name='spread', source=data_mod.DataSource['raw'],
                                  dim=data_mod.DataDim['Data1D'],
                                  distribution=data_mod.DataDistribution['spread'],
                                  data=[data_array],
                                  nav_indexes=(0,),
                                  axes=[sig_axis, nav_axis_0, nav_axis_1])

    def test_compute_shape(self, init_data_spread):
        data, data_array, sig_axis, nav_axis_0, nav_axis_1, Nspread = init_data_spread
        assert data.axes_manager.compute_shape_from_axes() == data_array.shape

    def test_repr(self, init_data_spread):
        data, data_array, sig_axis, nav_axis_0, nav_axis_1, Nspread = init_data_spread
        assert data.axes_manager._get_dimension_str() == f'({nav_axis_1.size}|{sig_axis.size})'

    def test_deep_copy_with_new_data(self, init_data_spread):
        data, data_array, sig_axis, nav_axis_0, nav_axis_1, Nspread = init_data_spread

        #on nav_indexes
        IND_TO_REMOVE = 0
        new_data = data.deepcopy_with_new_data([np.squeeze(dat[0, :]) for dat in data.data],
                                               remove_axes_index=IND_TO_REMOVE)

        assert new_data.nav_indexes == ()
        assert new_data.sig_indexes == (0,)
        assert new_data.shape == np.squeeze(data.data[0][0, :]).shape
        assert len(new_data.get_axis_indexes()) == 1
        assert data.get_axis_from_index(IND_TO_REMOVE)[0] not in new_data.axes

        # on sig_indexes
        IND_TO_REMOVE = 1
        new_data = data.deepcopy_with_new_data([np.squeeze(dat[:, 0]) for dat in data.data],
                                               remove_axes_index=IND_TO_REMOVE)

        assert new_data.nav_indexes == (0,)
        assert new_data.sig_indexes == ()
        assert new_data.shape == np.squeeze(data.data[0][:, 0]).shape
        assert len(new_data.get_axis_indexes()) == 1
        assert data.get_axis_from_index(IND_TO_REMOVE)[0] not in new_data.axes

    @pytest.mark.parametrize("IND_MEAN", [0, 1])
    def test_mean(self, IND_MEAN, init_data_spread):
        data, data_array, sig_axis, nav_axis_0, nav_axis_1, Nspread = init_data_spread

        new_data = data.mean(IND_MEAN)
        for ind, dat in enumerate(data):
            assert np.all(new_data[ind] == pytest.approx(np.mean(dat, IND_MEAN)))
        assert len(new_data.get_axis_indexes()) == 1
        if IND_MEAN in data.nav_indexes:
            assert len(new_data.axes) == 1
        else:
            assert len(new_data.axes) == 2

    def test_sorted_spread_signal(self, init_spread_data_arrays):

        for data_arrays in init_spread_data_arrays:
            axis_0_0 = data_mod.Axis('axis_0_0', data=np.array([0, 4, 2, 45, -10]), index=0, spread_order=0)
            axis_0_1 = data_mod.Axis('axis_0_1', data=np.array([-10, -20, -30, -40, -50]), index=0, spread_order=1)

            axes = [axis_0_0.copy(), axis_0_1.copy()]

            sorted_index_0_0 = np.argsort(axis_0_0.get_data())

            data = data_mod.DataRaw('mydata', distribution='spread',
                                    data=data_arrays[:],
                                    axes=[axis_0_0.copy(), axis_0_1.copy()],
                                    nav_indexes=(0,))
            data.create_missing_axes()
            data.inav[3]

            assert data.sort_data(axis_index=3, spread_index=0) == data
            sorted_data = data.sort_data(axis_index=0, spread_index=0)
            for ind in range(len(data)):
                assert np.allclose(sorted_data.data[ind],  data_arrays[ind][sorted_index_0_0, ...])
            for nav_index in sorted_data.nav_indexes:
                for axis in sorted_data.get_axis_from_index(nav_index):
                    assert np.allclose(axis.get_data(),
                                       data._am.get_axis_from_index_spread(axis.index, axis.spread_order)
                                       .get_data()[sorted_index_0_0])
            for sig_index in sorted_data.sig_indexes:
                for axis in sorted_data.get_axis_from_index(sig_index):
                    assert np.allclose(axis.get_data(),
                                       data._am.get_axis_from_index_spread(axis.index, axis.spread_order)
                                       .get_data())

    def test_interp(self):
        data_arrays = [np.array([11, 12, 14, 15]), np.array([21, 22, 24, 25])]
        data_arrays_expected = [np.array([11, 12, 13, 14, 15]), np.array([21, 22, 23, 24, 25])]
        axis_array = np.array([1, 2, 4, 5])
        new_axis_array = np.array([1, 2, 3, 4, 5])

        dwa = data_mod.DataRaw('mydata', distribution='spread',
                               data=data_arrays[:],
                               axes=[data_mod.Axis('axis', data=axis_array)])

        dwa_interp = dwa.interp(new_axis_array)
        for ind in range(len(data_arrays_expected)):
            assert np.allclose(dwa_interp.data[ind], data_arrays_expected[ind])

        dwa = init_data(DATA0D)
        with pytest.raises(ValueError):
            dwa.interp(new_axis_array)

        dwa = init_data(DATA0D)
        with pytest.raises(ValueError):
            dwa.interp(new_axis_array)


class TestSlicingUniform:
    def test_slice_navigation(self, init_data_uniform):
        data_raw = init_data_uniform
        assert data_raw.shape == (Nn0, Nn1, DATA2D.shape[0], DATA2D.shape[1])

        data_00: data_mod.DataWithAxes = data_raw.inav[0, :]
        data_OO_value: data_mod.DataWithAxes = data_raw.vnav[0., :]
        assert data_OO_value == data_00
        assert data_00.shape == (Nn1, DATA2D.shape[0], DATA2D.shape[1])
        assert len(data_00.axes) == 3
        assert data_00.nav_indexes == (0, )
        assert data_00.get_axis_from_index(0)[0].label == 'nav1'

        data_01 = data_raw.inav[:, 2]
        axis = data_raw.get_axis_from_index(data_raw.nav_indexes[1])[0]
        data_01_value = data_raw.vnav[:, axis.get_data()[2]]
        assert data_01 == data_01_value
        assert data_01.shape == (Nn0, DATA2D.shape[0], DATA2D.shape[1])
        assert len(data_01.axes) == 3
        assert data_01.nav_indexes == (0, )
        assert data_01.get_axis_from_index(0)[0].label == 'nav0'

        data_1 = data_raw.inav[0, 2]
        axis_0 = axis = data_raw.get_axis_from_index(data_raw.nav_indexes[0])[0]
        axis_1 = axis = data_raw.get_axis_from_index(data_raw.nav_indexes[1])[0]
        data_1_value = data_raw.vnav[axis_0.get_data()[0],
            axis_1.get_data()[2]]
        assert data_1 == data_1_value
        assert data_1.shape == (DATA2D.shape[0], DATA2D.shape[1])
        assert len(data_1.axes) == 2
        assert data_1.nav_indexes == ()

        data_2: data_mod.DataWithAxes = data_raw.inav[0:3, 2:4]
        data_2_value = data_raw.vnav[axis_0.get_data()[0]:axis_0.get_data()[3],
            axis_1.get_data()[2]:axis_1.get_data()[4]]
        assert data_2 == data_2_value
        assert data_2.shape == (3, 2, DATA2D.shape[0], DATA2D.shape[1])
        assert data_2.get_axis_from_index(0)[0].size == 3
        assert data_2.get_axis_from_index(1)[0].size == 2

    def test_slice_ellipsis(self, init_data_uniform):
        data_raw = init_data_uniform
        assert data_raw.shape == (Nn0, Nn1, DATA2D.shape[0], DATA2D.shape[1])

        data_sliced = data_raw.inav[0, ...]
        data_sliced_value = data_raw.vnav[0, ...]
        assert data_sliced == data_sliced_value
        assert data_sliced.nav_indexes == (0,)
        assert data_sliced.shape == (Nn1, DATA2D.shape[0], DATA2D.shape[1])

    def test_slice_miscellanous(self, init_data_uniform):
        data_raw = init_data_uniform
        assert data_raw.shape == (Nn0, Nn1, DATA2D.shape[0], DATA2D.shape[1])

        data_sliced = data_raw.inav[0:1, ...]
        assert data_sliced.shape == (Nn1, DATA2D.shape[0], DATA2D.shape[1])
        assert data_sliced.nav_indexes == (0,)

    def test_slice_signal(self, init_data_uniform):
        data_raw = init_data_uniform
        assert data_raw.shape == (Nn0, Nn1, DATA2D.shape[0], DATA2D.shape[1])

        data_00: data_mod.DataWithAxes = data_raw.isig[0, :]
        assert data_00.shape == (Nn0, Nn1, DATA2D.shape[1])
        assert len(data_00.axes) == 3
        assert data_00.nav_indexes == (0, 1)
        assert data_00.get_axis_from_index(2)[0].label == 'signal1'

        data_01 = data_raw.isig[:, 2]
        assert data_01.shape == (Nn0, Nn1, DATA2D.shape[0])
        assert len(data_01.axes) == 3
        assert data_01.nav_indexes == (0, 1)
        assert data_01.get_axis_from_index(2)[0].label == 'signal0'

        data_1 = data_raw.isig[0, 2]
        assert data_1.shape == (Nn0, Nn1)
        assert len(data_1.axes) == 2
        assert data_1.nav_indexes == (0, 1)

        data_2: data_mod.DataWithAxes = data_raw.isig[0:3, 2:4]
        assert data_2.shape == (Nn0, Nn1, 3, 2)
        assert data_2.get_axis_from_index(2)[0].size == 3
        assert data_2.get_axis_from_index(3)[0].size == 2

    def test_slicing_setter(self):
        data_raw, shape = init_dataND()
        assert data_raw.shape == (5, 6, 3)
        data_raw.nav_indexes = (0, 1)

        data_nav = data_mod.DataRaw('to replace', data=[np.ones((5, 6))])

        data_raw.isig[0] = data_nav

        assert np.allclose(data_raw.isig[0][0], data_nav[0])


class TestSlicingSpread:
    def test_slice_navigation(self, init_data_spread):
        data, data_array, sig_axis, nav_axis_0, nav_axis_1, Nspread = init_data_spread

        data_0: data_mod.DataWithAxes = data.inav[3]
        assert data_0.shape == DATA1D.shape
        assert len(data_0.axes) == 1
        assert data_0.nav_indexes == ()
        assert data_0.axes[0].index == 0

        data_1: data_mod.DataWithAxes = data.inav[3:6]
        assert data_1.shape == (3, DATA1D.shape[0])
        assert len(data_1.axes) == 3
        assert data_1.nav_indexes == (0, )

    def test_slice_signal(self, init_data_spread):
        data, data_array, sig_axis, nav_axis_0, nav_axis_1, Nspread = init_data_spread

        data_0: data_mod.DataWithAxes = data.isig[3]
        assert data_0.shape == (Nspread, )
        assert len(data_0.axes) == 2
        assert data_0.nav_indexes == (0, )

        data_1: data_mod.DataWithAxes = data.isig[3:6]
        assert data_1.shape == (Nspread, 3)
        assert len(data_1.axes) == 3
        assert data_1.nav_indexes == (0, )


class TestDataSource:

    def test_data_raw(self):
        Ndata = 2
        data = data_mod.DataRaw('myData', data=[DATA2D for ind in range(Ndata)])
        assert isinstance(data, data_mod.DataWithAxes)
        assert data.source == data_mod.DataSource['raw']

    def test_data_calculated(self):
        Ndata = 2
        data = data_mod.DataCalculated('myData', data=[DATA2D for ind in range(Ndata)])
        assert isinstance(data, data_mod.DataWithAxes)
        assert data.source == data_mod.DataSource['calculated']

    def test_data_from_roi(self):
        Ndata = 2
        data = data_mod.DataFromRoi('myData', data=[DATA2D for ind in range(Ndata)])
        assert isinstance(data, data_mod.DataWithAxes)
        assert data.source == data_mod.DataSource['calculated']


class TestDataToExport:
    def test_init(self, ini_data_to_export):
        dat1, dat2, data = ini_data_to_export

        assert data.name == 'toexport'
        assert data.data == [dat1, dat2]
        assert len(data) == 2

    def test_data_type(self):
        with pytest.raises(TypeError):
            data_mod.DataToExport(name='toexport', data=[np.zeros((10,))])
        with pytest.raises(TypeError):
            data_mod.DataToExport(name='toexport', data=init_data())

        data_mod.DataToExport(name='toexport', data=[data_mod.DataFromRoi('mydata', data=[np.zeros((10,))])])

    def test_append(self, ini_data_to_export):
        dat1, dat2, data = ini_data_to_export

        data.append([dat1, dat2])
        assert data.data == [dat1, dat2]
        assert len(data) == 2

        dat3 = init_data(data=DATA0D, Ndata=1, name='data0D')
        data.append(dat3)
        assert len(data) == 3
        assert data.data == [dat1, dat2, dat3]

    def test_getitem(self):
        dat0D = init_data(DATA0D, 2, name='my0DData', source='raw')
        dat1D_calculated = init_data(DATA1D, 2, name='my1DDatacalculated', source='calculated')
        dat1D_raw = init_data(DATA1D, 2, name='my1DDataraw', source='raw')

        data = data_mod.DataToExport(name='toexport', data=[dat0D, dat1D_calculated, dat1D_raw])

        assert isinstance(data[0], data_mod.DataWithAxes)

        assert data[0] == dat0D
        assert data[1] == dat1D_calculated
        assert data[2] == dat1D_raw

        index_slice = 1
        sliced_data = data[index_slice:]
        assert isinstance(sliced_data, data_mod.DataToExport)
        assert len(sliced_data) == len(data) - index_slice

        sliced_data = data[0:2]
        assert len(sliced_data) == 2
        assert sliced_data[0] == dat0D
        assert sliced_data[1] == dat1D_calculated

        sliced_data = data[0::2]
        assert len(sliced_data) == 2
        assert sliced_data[0] == dat0D
        assert sliced_data[1] == dat1D_raw

    def test_get_data_from_source(self):
        dat0D = init_data(DATA0D, 2, name='my0DData', source='raw')
        dat1D_calculated = init_data(DATA1D, 2, name='my1DDatacalculated', source='calculated')
        dat1D_raw = init_data(DATA1D, 2, name='my1DDataraw', source='raw')

        data = data_mod.DataToExport(name='toexport', data=[dat0D, dat1D_calculated, dat1D_raw])

        assert len(data.get_data_from_source('calculated')) == 1
        assert data.get_data_from_source('calculated').data == [dat1D_calculated]

        assert len(data.get_data_from_source('raw')) == 2
        assert data.get_data_from_source('raw').get_data_from_dim('Data0D').data == [dat0D]
        assert data.get_data_from_source('raw').get_data_from_dim('Data1D').data == [dat1D_raw]

    def test_get_data_from_attribute(self):
        dat0D = init_data(DATA0D, 2, name='my0DData', source='raw')
        dat1D_calculated = init_data(DATA1D, 2, name='my1DDatacalculated', source='calculated')
        dat1D_raw = init_data(DATA1D, 2, name='my1DDataraw', source='raw')

        data = data_mod.DataToExport(name='toexport', data=[dat0D, dat1D_calculated, dat1D_raw])

        assert data.get_data_from_attribute('name', 'my1DDataraw')[0] == dat1D_raw
        assert len(data.get_data_from_attribute('dim', 'Data1D')) == 2
        assert data.get_data_from_attribute('dim', 'Data0D')[0] == dat0D

    def test_get_data_from_missing_attribute(self):
        dat0D = init_data(DATA0D, 2, name='my0DData', source='raw')
        dat1D_calculated = init_data(DATA1D, 2, name='my1DDatacalculated', source='calculated')
        dat1D_raw = init_data(DATA1D, 2, name='my1DDataraw', source='raw')

        EXTRA_ATTRIBUE = 'aweirdattribute'
        data = data_mod.DataToExport(name='toexport', data=[dat0D, dat1D_calculated, dat1D_raw])

        assert len(data.get_data_from_missing_attribute(EXTRA_ATTRIBUE)) == 3
        dat1D_calculated.add_extra_attribute(**{EXTRA_ATTRIBUE: 'blabla'})
        assert len(data.get_data_from_missing_attribute(EXTRA_ATTRIBUE)) == 2

    def test_get_data_by_dim(self, ini_data_to_export):
        dat1, dat2, data = ini_data_to_export
        assert len(data.get_data_from_dim(data_mod.DataDim['Data0D'])) == 0

        dat3 = init_data(data=DATA0D, Ndata=1, name='data0D')
        data.append(dat3)
        assert isinstance(data.get_data_from_dim('Data0D'), data_mod.DataToExport)
        assert data.get_data_from_dim('Data0D').data == [dat3]
        assert data.get_data_from_dim(data_mod.DataDim['Data0D']).data == [dat3]

        assert data.get_data_from_dim(data_mod.DataDim['Data1D']).data == [dat2]
        assert data.get_data_from_dim(data_mod.DataDim['Data2D']).data == [dat1]

        dat4 = init_data(data=DATA2D, Ndata=1, name='data2Dbis')
        data.append(dat4)
        assert data.get_data_from_dim(data_mod.DataDim['Data2D']).data == [dat1, dat4]

    def test_get_data_from_name(self, ini_data_to_export):
        dat1, dat2, data = ini_data_to_export
        # without origin first
        assert data.get_data_from_name_origin('data2D') == dat1
        assert data.get_data_from_name_origin('data1D') == dat2
        dat3 = init_data(data=DATA2D, Ndata=1, name='data2Dbis')
        data.append(dat3)
        assert data.get_data_from_name_origin('data2Dbis') == dat3
        dat4 = init_data(data=DATA2D, Ndata=1, name='data2D')
        data.append(dat4)
        assert data.get_data_from_name_origin('data2D') == dat4
        assert dat1 not in data.data

    def test_get_full_names(self, ini_data_to_export):
        dat1, dat2, data = ini_data_to_export
        ORGIN1 = 'origin1'
        ORIGIN2 = 'origin2'
        dat1.origin = ORGIN1
        dat2.origin = ORIGIN2

        assert data.get_full_names() == [dat1.get_full_name(), dat2.get_full_name()]

    def test_get_origins(self, ini_data_to_export):
        dat1, dat2, data = ini_data_to_export
        ORIGIN1 = 'origin1'
        ORIGIN2 = 'origin2'
        dat1.origin = ORIGIN1
        dat2.origin = ORIGIN2

        for origin in data.get_origins():
            assert origin in [ORIGIN1, ORIGIN2]

        dat2.origin = ORIGIN1

        assert data.get_origins()  == [ORIGIN1]

    def test_get_data_from_name_origin(self, ini_data_to_export):
        dat1, dat2, data = ini_data_to_export
        # with origin
        assert data.get_data_from_name_origin('dataxxD', 'toexport') is None

        assert data.get_data_from_name_origin('data2D', 'toexport') == dat1
        assert data.get_data_from_name_origin('data2D', 'toexport2') is None
        assert data.get_data_from_name_origin('data1D', 'toexport') == dat2

    def test_get_data_from_full_names(self, ini_data_to_export):
        dat1, dat2, data = ini_data_to_export

        assert data.get_data_from_full_names(['toexport/data2D'])[0] == dat1
        assert data.get_data_from_full_names(['toexport/data1D'])[0] == dat2
        assert data.get_data_from_full_names(['toexport/data2D', 'toexport/data1D'])[0] == dat1
        assert data.get_data_from_full_names(['toexport/data2D', 'toexport/data1D'])[1] == dat2

    def test_index(self, ini_data_to_export):
        dat1, dat2, dte = ini_data_to_export
        dat3 = init_data(data=DATA2D, Ndata=2, name='data2D3')
        dat4 = init_data(data=0.1*dat1.data[0], Ndata=2, name=dat1.name)
        dat3.origin = 'anorigin'
        dte.append(dat3)
        dat4.origin = 'anotherorigin'
        dte.append(dat4)
        assert dte.index(dat1) == 0
        assert dte.index(dat1) == dte.data.index(dat1)
        assert dte.index(dat2) == dte.data.index(dat2)
        assert dte.index(dat3) != 0  # same data but not the same name
        assert dte.index(dat4) != 0  # same name but not the same data

    def test_index_from_name_origin(self, ini_data_to_export):
        dat1, dat2, data = ini_data_to_export
        assert data.index_from_name_origin(dat1.name, dat1.origin) == data.index(dat1)
        assert data.index_from_name_origin(dat2.name, dat2.origin) == data.index(dat2)

    def test_pop(self, ini_data_to_export):
        dat1, dat2, data = ini_data_to_export
        assert len(data) == 2
        dat = data.pop(1)
        assert dat is dat2
        assert len(data) == 1
        with pytest.raises(ValueError):
            data.index(dat2)

    def test_remove(self, ini_data_to_export):
        dat1, dat2, data = ini_data_to_export

        assert len(data) == 2
        dat2bis = data.remove(dat2)
        assert dat2 is dat2bis

    def test_get_names(self, ini_data_to_export):
        dat1, dat2, data = ini_data_to_export
        assert data.get_names() == ['data2D', 'data1D']
        assert data.get_names('Data1D') == ['data1D']
        assert data.get_names('Data2D') == ['data2D']

        dat3 = init_data(data=DATA2D, Ndata=1, name='data2Dbis')
        data.append(dat3)
        assert data.get_names('data2d') == ['data2D', 'data2Dbis']

    def test_math(self):
        dat1 = init_data(data=DATA2D, Ndata=2, name='data2D1')
        dat2 = init_data(data=0.2 * DATA2D, Ndata=2, name='data2D2')
        dat2bis = init_data(data=0.2 * DATA2D.reshape(DATA2D.shape[-1::-1]), Ndata=2, name='data2D2bis')
        dat3 = init_data(data=-0.7 * DATA2D, Ndata=2, name='data2D3')

        data1 = data_mod.DataToExport(name='toexport', data=[dat1, dat2])
        data2 = data_mod.DataToExport(name='toexport', data=[dat3, dat2])
        data3 = data_mod.DataToExport(name='toexport', data=[dat3, dat2bis])
        data4 = data_mod.DataToExport(name='toexport', data=[dat3, dat2, dat1])

        data_sum = data1 + data2
        data_diff = data1 - data2

        MUL_COEFF = 0.24
        DIV_COEFF = 12.7

        data_mul = data1 * MUL_COEFF
        data_div = data2 / DIV_COEFF

        assert data_sum[0] == dat1 + dat3
        assert data_sum[1] == dat2 + dat2

        assert data_diff[0] == dat1 - dat3
        assert data_diff[1] == dat2 - dat2

        with pytest.raises(ValueError):
            data1 + data3

        with pytest.raises(TypeError):
            data1 + data4

        assert data_mul[0] == dat1 * MUL_COEFF
        assert data_mul[1] == dat2 * MUL_COEFF

        assert data_div[0] == dat3 / DIV_COEFF
        assert data_div[1] == dat2 / DIV_COEFF

    def test_average(self):
        dat1 = init_data(data=DATA2D, Ndata=2, name='data2D1')
        dat2 = init_data(data=0.2 * DATA2D, Ndata=2, name='data2D2')
        dat3 = init_data(data=-0.7 * DATA2D, Ndata=2, name='data2D3')

        data1 = data_mod.DataToExport(name='toexport', data=[dat1, dat2])
        data2 = data_mod.DataToExport(name='toexport', data=[dat3, dat2])

        WEIGHT = 6

        data1 = data1.average(data2, WEIGHT)

        assert data1[0] == dat1.average(dat3, WEIGHT)
        assert data1[1] == dat2.average(dat2, WEIGHT)

    def test_merge(self):

        dat1 = init_data(data=DATA1D, Ndata=1, name='data1D1')
        dat2 = init_data(data=0.2 * DATA1D, Ndata=2, name='data1D2')
        dat3 = init_data(data=-0.7 * DATA1D, Ndata=3, name='data1D3')

        dte = data_mod.DataToExport('merging', data=[dat1, dat2, dat3])

        dwa = dte.merge_as_dwa('Data1D')
        assert dwa.name == dte.name
        assert len(dwa) == 6
        assert np.all(dwa[0] == pytest.approx(DATA1D))
        assert np.all(dwa[1] == pytest.approx(0.2*DATA1D))
        assert np.all(dwa[3] == pytest.approx(-0.7*DATA1D))


        assert dwa.labels == dat1.labels + dat2.labels + dat3.labels


class TestUnits:

    def test_unit_in_registry(self):

        dwa = data_mod.DataRaw('data', units='unknown_unit', data=[np.array([0, 1, 2])])
        assert dwa.units == ''  # transforms to dimensionless

        dwa = data_mod.DataRaw('data', units='ms', data=[np.array([0, 1, 2])])
        assert dwa.units == 'ms'

    def test_add_with_units(self):
        array_1 = np.array([0, 1, 2])
        dwa_1 = data_mod.DataRaw('data', units='s', data=[array_1])
        assert dwa_1.units == 's'

        dwa_2 = data_mod.DataRaw('data', units='ms', data=[array_1])
        assert dwa_2.units == 'ms'

        dwa_sum_s = dwa_1 + dwa_2
        assert np.allclose(dwa_sum_s[0], array_1 + array_1 / 1000)
        assert dwa_sum_s.units == 's'

        dwa_sum_ms = dwa_2 + dwa_1
        assert np.allclose(dwa_sum_ms[0], array_1 + array_1 * 1000)
        assert dwa_sum_ms.units == 'ms'

    def test_units_as(self):
        array = np.array([0, 1, 2])
        dwa_s = data_mod.DataRaw('data', units='s', data=[array])
        assert dwa_s.units == 's'

        dwa_ms = dwa_s.units_as('ms')
        assert dwa_ms.units == 'ms'
        assert np.allclose(dwa_ms[0], array * 1000)
        assert dwa_ms is dwa_s

        dwa_s = data_mod.DataRaw('data', units='s', data=[array])
        assert dwa_s.units == 's'
        dwa_ms = dwa_s.units_as('ms', inplace=False)
        assert dwa_ms.units == 'ms'
        assert np.allclose(dwa_ms[0], array * 1000)
        assert dwa_s.units == 's'
        assert np.allclose(dwa_s[0], array)

        wavelength = np.array([400, 600, 800])
        dwa = data_mod.DataRaw('data', units='nm', data=[wavelength])
        dwa_fs = dwa.units_as('eV', inplace=False, context='spectroscopy')

        assert np.allclose(dwa_fs.data[0], nm2eV(wavelength))

        dwa.units_as('eV', context='spectroscopy')
        assert np.allclose(dwa[0], nm2eV(wavelength))

class TestNumpyUfunc:

    units = 'm'

    @pytest.mark.parametrize(
        ('elt1', 'elt2', 'unit_error'), (
                [np.random.rand(),
                 init_data(DATA1D, Ndata=2, name='raw', source=data_mod.DataSource.raw,
                           units=REAL_UNITS),
                 True
                 ],
                [np.random.rand(),
                 init_data(DATA1D, Ndata=2, name='raw', source=data_mod.DataSource.raw,
                           units=''),
                 False
                 ],
                [init_data(DATA1D, Ndata=2, name='raw', source=data_mod.DataSource.raw,
                           units=REAL_UNITS),
                 data_mod.Q_(np.random.rand(), REAL_UNITS),
                 False
                 ],
                [init_data(DATA1D, Ndata=2, name='raw', source=data_mod.DataSource.raw,
                           units=REAL_UNITS),
                 data_mod.Q_(np.random.rand(len(DATA1D)), REAL_UNITS),
                 False
                 ],
                [init_data(DATA1D, Ndata=2, name='raw', source=data_mod.DataSource.raw,
                      units=REAL_UNITS),
                 init_data(DATA1D * 10, Ndata=2, name='raw', source=data_mod.DataSource.raw,
                           units=REAL_UNITS),
                 False
                 ],
        ))
    def test_add(self, elt1, elt2, unit_error):
        if unit_error:
            with pytest.raises(pint.errors.DimensionalityError):
                dwa_add = np.add(elt1, elt2)
                return
        else:
            dwa_add = np.add(elt1, elt2)

            for ind, data_array in enumerate(dwa_add.data):
                if isinstance(elt1, data_mod.DataBase):
                    elt1bis = elt1.data[ind]
                elif isinstance(elt1, data_mod.Q_):
                    elt1bis = elt1.m_as(REAL_UNITS)
                else:
                    elt1bis = elt1
                if isinstance(elt2, data_mod.DataBase):
                    elt2bis = elt2.data[ind]
                elif isinstance(elt2, data_mod.Q_):
                    elt2bis = elt2.m_as(REAL_UNITS)
                else:
                    elt2bis = elt2
                assert np.allclose(data_array, np.add(elt1bis, elt2bis))

    @pytest.mark.parametrize('operator', (np.add, np.subtract))
    def test_add_subtract_operator(self, operator):
        dwa_s = data_mod.DataRaw('raw', units='s', data=[DATA1D, DATA1D])
        dwa_ms = data_mod.DataRaw('raw', units='ms', data=[DATA1D, DATA1D])
        dwa_m = data_mod.DataRaw('raw', units='m', data=[DATA1D, DATA1D])

        dwa_s_ms = operator(dwa_s, dwa_ms)

        assert dwa_s_ms.units == 's'
        assert np.allclose(dwa_s_ms.data[0], operator(DATA1D, DATA1D / 1000))

        with pytest.raises(pint.errors.DimensionalityError):
            dwa_s + dwa_m

    def test_mult_operator(self):
        dwa_s = data_mod.DataRaw('raw', units='s', data=[DATA1D, DATA1D])
        dwa_ms = data_mod.DataRaw('raw', units='ms', data=[DATA1D, DATA1D])
        dwa_m = data_mod.DataRaw('raw', units='m', data=[DATA1D, DATA1D])

        dwa_s_ms = np.multiply(dwa_s, dwa_ms)

        assert np.allclose(dwa_s_ms.quantities[0],
                           data_mod.Q_(DATA1D, 's') * data_mod.Q_(DATA1D, 'ms'))

        assert dwa_s_ms.quantities[0].is_compatible_with('s*s')

    def test_quantity_dwa_mult(self):
        q = data_mod.Q_(3.5, 's')
        dwa_s = data_mod.DataRaw('raw', units='Hz', data=[DATA1D, DATA1D])

        dwa_ufunc = q * dwa_s
        assert isinstance(dwa_ufunc, type(dwa_s))
        assert np.allclose(dwa_ufunc[0], q.magnitude * DATA1D)


class TestFuncNumpy:
    def test_all(self):
        dwa_bool = data_mod.DataRaw('raw', units='', data=[DATA1D == DATA1D])
        assert np.all(dwa_bool)

    def test_func_remove_axis(self):

        dwa = data_mod.DataRaw('raw', units='', data=[GAUSSIAN_2D],
                               axes=[init_axis(YAXIS_GAUSSIAN, 0, label='Yaxis'),
                                     init_axis(XAXIS_GAUSSIAN, 1, label='Xaxis')])

        dwa_std = np.std(dwa, axis=0)
        dwa_mean = np.mean(dwa, axis=0)

        dwa_max = np.max(dwa)
        assert dwa_max.dim == data_mod.DataDim.Data0D
        assert np.allclose(dwa_max[0], np.max(GAUSSIAN_2D))
        assert len(dwa_max.axes) == 0

        dwa_min = np.min(dwa)
        assert dwa_min.dim == data_mod.DataDim.Data0D
        assert np.allclose(dwa_min[0], np.min(GAUSSIAN_2D))

        dwa_min = np.min(dwa, axis=1)
        assert dwa_min.dim == data_mod.DataDim.Data1D
        assert np.allclose(dwa_min[0], np.min(GAUSSIAN_2D, axis=1))

        dwa_min = np.min(dwa, axis=0)
        assert dwa_min.dim == data_mod.DataDim.Data1D
        assert np.allclose(dwa_min[0], np.min(GAUSSIAN_2D, axis=0))

        dwa_min = np.min(dwa, axis=-1)
        assert dwa_min.dim == data_mod.DataDim.Data1D
        assert np.allclose(dwa_min[0], np.min(GAUSSIAN_2D, axis=-1))

        dwa_min = np.min(dwa, axis=(0, 1))
        assert dwa_min.dim == data_mod.DataDim.Data0D
        assert np.allclose(dwa_min[0], np.min(GAUSSIAN_2D, axis=(0, 1)))

    def test_booleans(self):
        dwa_bool = data_mod.DataRaw('raw', units='', data=[
            np.array([[True, False], [True, True]])],)

        dwa_all = np.all(dwa_bool)
        assert not dwa_all[0]
        assert dwa_all.dim == data_mod.DataDim.Data0D

        dwa_all = np.all(dwa_bool, axis=0)
        assert dwa_all[0][0] == True
        assert dwa_all[0][1] == False
        assert dwa_all.dim == data_mod.DataDim.Data1D

        dwa_all = np.all(dwa_bool, axis=1)
        assert dwa_all[0][0] == False
        assert dwa_all[0][1] == True
        assert dwa_all.dim == data_mod.DataDim.Data1D

        dwa_any = np.any(dwa_bool, axis=0)
        assert np.any(dwa_any[0])
        assert dwa_any.dim == data_mod.DataDim.Data1D

    def test_func_on_complex(self):
        ANGLE = np.pi / 3
        dwa = data_mod.DataRaw('raw', units='', data=[np.exp(1j * ANGLE * np.ones((10,)))])

        dwa_angle_rad = np.angle(dwa)
        assert np.allclose(dwa_angle_rad.data[0], ANGLE)

        dwa_angle_deg = np.angle(dwa, deg=True)
        assert np.allclose(dwa_angle_deg[0], np.rad2deg(ANGLE))

        assert np.allclose(np.real(dwa)[0], np.cos(ANGLE))
        assert np.allclose(np.imag(dwa)[0], np.sin(ANGLE))
        assert np.allclose(np.abs(dwa)[0], 1)
        assert np.allclose(np.absolute(dwa)[0], 1)

    def test_db(self):
        dwa = data_mod.DataRaw('raw', units='s', data=[GAUSSIAN_2D],)

        dwa_db = dwa.to_dB()
        assert dwa_db.units == ''

    def test_all_close(self):
        dwa_a = data_mod.DataRaw('raw', units='', data=[DATA1D, DATA1D])
        dwa_b = data_mod.DataRaw('raw', units='', data=[DATA1D, DATA1D])

        assert np.allclose(dwa_a, dwa_b)

        assert not np.allclose(dwa_a, dwa_b * 0.1)

    def test_flip_transpose_rotate(self):
        dwa = data_mod.DataRaw('raw', units='', data=[DATA2D])

        dwa_transform = np.flipud(dwa)
        for ind in range(len(dwa)):
            assert np.allclose(np.flipud(dwa[ind]), dwa_transform[ind])

        dwa_transform = np.fliplr(dwa)
        for ind in range(len(dwa)):
            assert np.allclose(np.fliplr(dwa[ind]), dwa_transform[ind])

        dwa_transform = np.transpose(dwa)
        for ind in range(len(dwa)):
            assert np.allclose(np.transpose(dwa[ind]), dwa_transform[ind])


