import numpy as np
import pytest

from pymodaq_data.serialize.factory import SerializableFactory
from pymodaq_data.serialize.serializer import (StringSerializeDeserialize as SSD,
                                               BytesSerializeDeserialize as BSD,
                                               ScalarSerializeDeserialize as ScSD,
                                               NdArraySerializeDeserialize as NdSD,
                                               ListSerializeDeserialize as LSD,)


from pymodaq_data import data as data_mod


ser_factory = SerializableFactory()


LABEL = 'A Label'
UNITS = 'mm'
OFFSET = -20.4
SCALING = 0.22
SIZE = 20
DATA = OFFSET + SCALING * np.linspace(0, SIZE-1, SIZE)

DATA0D = np.array([2.7])
DATA1D = np.arange(0, 10)
DATA2D = np.arange(0, 5*6).reshape((5, 6))
DATAND = np.arange(0, 5 * 6 * 3).reshape((5, 6, 3))
Nn0 = 10
Nn1 = 5


class DataFromPlugins(data_mod.DataRaw):
    """ Inheriting class for test purpose"""
    pass


def init_axis(data=None, index=0):
    if data is None:
        data = DATA
    return data_mod.Axis(label=LABEL, units=UNITS, data=data, index=index)


def init_data(data=None, Ndata=1, axes=[], name='myData', source=data_mod.DataSource['raw'],
              labels=None, klass=data_mod.DataWithAxes,
              errors=True, units='ms') -> data_mod.DataWithAxes:
    if data is None:
        data = DATA2D
    if errors:
        errors = [np.random.random_sample(data.shape) for _ in range(Ndata)]
    else:
        errors = None
    return klass(name, source=source, data=[data for _ in range(Ndata)],
                 axes=axes, labels=labels, errors=errors,
                 extra1=True, extra2=[1, 2, 3], units=units)


@pytest.fixture()
def get_data():
    dat0D = init_data(DATA0D, 2, name='my0DData', source='raw', errors=True)
    dat1D_calculated = init_data(DATA1D, 2, name='my1DDatacalculated',
                                 klass=data_mod.DataCalculated, errors=True)
    dat1D_plugin = init_data(DATA1D, 2, name='my1DDataraw', klass=DataFromPlugins,
                          errors=False)
    dte = data_mod.DataToExport(name='toexport', data=[dat0D, dat1D_calculated, dat1D_plugin])
    return dte


def test_string_serialization():
    s = 'ert'
    obj_type = 'str'

    assert SSD.serialize(s) == b'\x00\x00\x00' + chr(len(s)).encode() + s.encode()

    assert ser_factory.get_serializer(type(s))(s) == \
           b'\x00\x00\x00' + chr(len(obj_type)).encode() + obj_type.encode() + SSD.serialize(s)

    assert ser_factory.get_serializer(type(s))(s) == \
           ser_factory.get_apply_serializer(s)

    assert SSD.deserialize(SSD.serialize(s)) == (s, b'')

    assert (ser_factory.get_apply_deserializer(ser_factory.get_apply_serializer(s))
            == (s, b''))


def test_bytes_serialization():
    b = b'kjlksjdf'
    obj_type = 'bytes'
    assert BSD.serialize(b) == b'\x00\x00\x00' + chr(len(b)).encode() + b
    assert (ser_factory.get_serializer(type(b))(b) ==
           b'\x00\x00\x00' + chr(len(obj_type)).encode() + obj_type.encode() +
            BSD.serialize(b))
    assert ser_factory.get_serializer(type(b))(b) == \
           ser_factory.get_apply_serializer(b)

    assert BSD.deserialize(BSD.serialize(b)) == (b, b'')

    assert (ser_factory.get_apply_deserializer(ser_factory.get_apply_serializer(b))
            == (b, b''))


def test_scalar_serialization():
    s = 23
    obj_type = 'int'
    assert (ser_factory.get_serializer(type(s))(s) ==
           b'\x00\x00\x00' + chr(len(obj_type)).encode() + obj_type.encode() +
            ScSD.serialize(s))
    assert (ser_factory.get_serializer(type(s))(s) ==
            ser_factory.get_apply_serializer(s))

    assert ScSD.deserialize(ScSD.serialize(s)) == (s, b'')

    assert (ser_factory.get_apply_deserializer(ser_factory.get_apply_serializer(s))
            == (s, b''))

    s = -3.8
    obj_type = 'float'
    assert (ser_factory.get_serializer(type(s))(s) ==
           b'\x00\x00\x00' + chr(len(obj_type)).encode() + obj_type.encode() +
            ScSD.serialize(s))
    assert (ser_factory.get_serializer(type(s))(s) ==
            ser_factory.get_apply_serializer(s))

    assert ScSD.deserialize(ScSD.serialize(s)) == (s, b'')

    assert (ser_factory.get_apply_deserializer(ser_factory.get_apply_serializer(s))
            == (s, b''))

    s = 4 - 2.5j
    obj_type = 'complex'
    assert (ser_factory.get_serializer(type(s))(s) ==
           b'\x00\x00\x00' + chr(len(obj_type)).encode() + obj_type.encode() +
            ScSD.serialize(s))
    assert (ser_factory.get_serializer(type(s))(s) ==
            ser_factory.get_apply_serializer(s))

    assert ScSD.deserialize(ScSD.serialize(s)) == (s, b'')

    assert (ser_factory.get_apply_deserializer(ser_factory.get_apply_serializer(s))
            == (s, b''))


def test_bool_serialization():
    s = True
    obj_type = 'bool'
    assert (ser_factory.get_serializer(type(s))(s) ==
           b'\x00\x00\x00' + chr(len(obj_type)).encode() + obj_type.encode() +
            ScSD.serialize(s))
    assert (ser_factory.get_serializer(type(s))(s) ==
            ser_factory.get_apply_serializer(s))

    assert ScSD.deserialize(ScSD.serialize(s)) == (s, b'')

    assert (ser_factory.get_apply_deserializer(ser_factory.get_apply_serializer(s))
            == (s, b''))

    s = False
    obj_type = 'bool'
    assert (ser_factory.get_serializer(type(s))(s) ==
           b'\x00\x00\x00' + chr(len(obj_type)).encode() + obj_type.encode() +
            ScSD.serialize(s))
    assert (ser_factory.get_serializer(type(s))(s) ==
            ser_factory.get_apply_serializer(s))
    assert ScSD.deserialize(ScSD.serialize(s)) == (s, b'')

    assert (ser_factory.get_apply_deserializer(ser_factory.get_apply_serializer(s))
            == (s, b''))


def test_ndarray_serialization_deserialization():

    ndarrays = [np.array([12, 56, 78,]),
                np.array([-12.8, 56, 78, ]),
                np.array([12]),
                np.array([True, False]),
                np.array([[12+6j, 56, 78, ],
                          [12, 56, 78, ],
                          [12, 56, 78, ]])]

    for ndarray in ndarrays:
        ser = NdSD.serialize(ndarray)
        assert isinstance(ser, bytes)
        assert np.allclose(NdSD.deserialize(NdSD.serialize(ndarray))[0], ndarray)

        assert np.allclose(
            ser_factory.get_apply_deserializer(
                ser_factory.get_apply_serializer(ndarray))[0], ndarray)


@pytest.mark.parametrize('obj_list', (['hjk', 'jkgjg', 'lkhlkhl'],  # homogeneous string
                                      [21, 34, -56, 56.7, 1+1j*99],  # homogeneous numbers
                                      [np.array([45, 67, 87654]),
                                       np.array([[45, 67, 87654], [-45, -67, -87654]])],  # homogeneous ndarrays
                                      ['hjk', 23, 34.7, np.array([1, 2, 3])],  # inhomogeneous list
                                    ))
def test_list_serialization_deserialization(obj_list):
    ser = LSD.serialize(obj_list)
    assert isinstance(ser, bytes)

    list_back = LSD.deserialize(ser)[0]
    assert isinstance(list_back, list)
    for ind in range(len(obj_list)):
        if isinstance(obj_list[ind], np.ndarray):
            assert np.allclose(obj_list[ind], list_back[ind])
        else:
            assert obj_list[ind] == list_back[ind]

    for ind, obj in enumerate(
            ser_factory.get_apply_deserializer(ser_factory.get_apply_serializer(obj_list))[0]):
        if isinstance(obj, np.ndarray):
            assert np.allclose(obj_list[ind], obj)
        else:
            assert obj_list[ind] == obj


def test_axis_serialization_deserialization():

    axis = init_axis()

    ser = axis.serialize(axis)
    assert isinstance(ser, bytes)

    axis_deser = axis.deserialize(ser)[0]
    assert axis_deser == axis

    axis_back = ser_factory.get_apply_deserializer(ser_factory.get_apply_serializer(axis))[0]
    assert axis_back == axis


def test_dwa_serialization_deserialization(get_data):
    dte = get_data

    for dwa in dte:
        dwa.extra_attributes = ['extra1', 'extra2']
        dwa.extra1 = True
        dwa.extra2 = 12.4
        ser = ser_factory.get_apply_serializer(dwa)
        assert isinstance(ser, bytes)
        dwa_back = ser_factory.get_apply_deserializer(ser)[0]

        assert dwa_back == dwa
        assert dwa_back.__class__.__name__ == dwa.__class__.__name__
        assert dwa == dwa_back
        assert dwa.extra_attributes == dwa_back.extra_attributes
        for attr in dwa.extra_attributes:
            assert getattr(dwa, attr) == getattr(dwa_back, attr)

    for dwa in dte:
        assert ser_factory.get_apply_deserializer(ser_factory.get_apply_serializer(dwa))[0] == dwa


def test_dte_serialization(get_data):
    dte = get_data

    ser = ser_factory.get_apply_serializer(dte)
    assert isinstance(ser, bytes)
    dte_back = ser_factory.get_apply_deserializer(ser)[0]

    assert dte_back.name == dte.name
    assert dte_back.timestamp == dte.timestamp
    for dwa in dte_back:
        assert dwa == dte.get_data_from_full_name(dwa.get_full_name())

    dte_back_factory = ser_factory.get_apply_deserializer(ser_factory.get_apply_serializer(dte))[0]
    assert dte_back_factory.name == dte.name
    assert dte_back_factory.timestamp == dte.timestamp
    for dwa in dte_back_factory:
        assert dwa == dte.get_data_from_full_name(dwa.get_full_name())

