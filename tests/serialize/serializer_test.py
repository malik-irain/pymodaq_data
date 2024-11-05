import numpy as np
import pytest

from pymodaq_data.serialize.serializer import (StringSerializeDeserialize as SSD,
                                               BytesSerializeDeserialize as BSD,
                                               ScalarSerializeDeserialize as ScSD,
                                               NdArraySerializeDeserialize as NdSD,
                                               ListSerializeDeserialize as LSD,)

from pymodaq_data.serialize.factory import SerializableFactory

ser_factory = SerializableFactory()


def test_string_serialization():
    s = 'ert'
    obj_type = 'str'

    assert SSD.string_serialize(s) == b'\x00\x00\x00' + chr(len(s)).encode() + s.encode()

    assert ser_factory.get_serializer(type(s))(s) == \
           b'\x00\x00\x00' + chr(len(obj_type)).encode() + obj_type.encode() + SSD.string_serialize(s)

    assert ser_factory.get_serializer(type(s))(s) == \
           ser_factory.get_apply_serializer(s)

    assert SSD.string_deserialize(SSD.string_serialize(s)) == (s, b'')

    assert (ser_factory.get_apply_deserializer(ser_factory.get_apply_serializer(s))
            == (s, b''))


def test_bytes_serialization():
    b = b'kjlksjdf'
    obj_type = 'bytes'
    assert BSD.bytes_serialize(b) == b'\x00\x00\x00' + chr(len(b)).encode() + b
    assert (ser_factory.get_serializer(type(b))(b) ==
           b'\x00\x00\x00' + chr(len(obj_type)).encode() + obj_type.encode() +
            BSD.bytes_serialize(b))
    assert ser_factory.get_serializer(type(b))(b) == \
           ser_factory.get_apply_serializer(b)

    assert BSD.bytes_deserialize(BSD.bytes_serialize(b)) == (b, b'')

    assert (ser_factory.get_apply_deserializer(ser_factory.get_apply_serializer(b))
            == (b, b''))


def test_scalar_serialization():
    s = 23
    obj_type = 'int'
    assert ScSD.scalar_serialize(s) == b'\x00\x00\x00\x03<i4\x00\x00\x00\x04\x17\x00\x00\x00'
    assert (ser_factory.get_serializer(type(s))(s) ==
           b'\x00\x00\x00' + chr(len(obj_type)).encode() + obj_type.encode() +
            ScSD.scalar_serialize(s))
    assert (ser_factory.get_serializer(type(s))(s) ==
            ser_factory.get_apply_serializer(s))

    assert ScSD.scalar_deserialize(ScSD.scalar_serialize(s)) == (s, b'')

    assert (ser_factory.get_apply_deserializer(ser_factory.get_apply_serializer(s))
            == (s, b''))

    s = -3.8
    obj_type = 'float'
    assert ScSD.scalar_serialize(s) == b'\x00\x00\x00\x03<f8\x00\x00\x00\x08ffffff\x0e\xc0'
    assert (ser_factory.get_serializer(type(s))(s) ==
           b'\x00\x00\x00' + chr(len(obj_type)).encode() + obj_type.encode() +
            ScSD.scalar_serialize(s))
    assert (ser_factory.get_serializer(type(s))(s) ==
            ser_factory.get_apply_serializer(s))

    assert ScSD.scalar_deserialize(ScSD.scalar_serialize(s)) == (s, b'')

    assert (ser_factory.get_apply_deserializer(ser_factory.get_apply_serializer(s))
            == (s, b''))

    s = 4 - 2.5j
    obj_type = 'complex'
    assert ScSD.scalar_serialize(s) == \
           (b'\x00\x00\x00\x04<c16\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x10@\x00\x00'
            b'\x00\x00\x00\x00\x04\xc0')
    assert (ser_factory.get_serializer(type(s))(s) ==
           b'\x00\x00\x00' + chr(len(obj_type)).encode() + obj_type.encode() +
            ScSD.scalar_serialize(s))
    assert (ser_factory.get_serializer(type(s))(s) ==
            ser_factory.get_apply_serializer(s))

    assert ScSD.scalar_deserialize(ScSD.scalar_serialize(s)) == (s, b'')

    assert (ser_factory.get_apply_deserializer(ser_factory.get_apply_serializer(s))
            == (s, b''))


def test_bool_serialization():
    s = True
    obj_type = 'bool'
    assert ScSD.scalar_serialize(s) == b'\x00\x00\x00\x03|b1\x00\x00\x00\x01\x01'
    assert (ser_factory.get_serializer(type(s))(s) ==
           b'\x00\x00\x00' + chr(len(obj_type)).encode() + obj_type.encode() +
            ScSD.scalar_serialize(s))
    assert (ser_factory.get_serializer(type(s))(s) ==
            ser_factory.get_apply_serializer(s))

    assert ScSD.scalar_deserialize(ScSD.scalar_serialize(s)) == (s, b'')

    assert (ser_factory.get_apply_deserializer(ser_factory.get_apply_serializer(s))
            == (s, b''))

    s = False
    obj_type = 'bool'
    assert ScSD.scalar_serialize(s) == b'\x00\x00\x00\x03|b1\x00\x00\x00\x01\x00'
    assert (ser_factory.get_serializer(type(s))(s) ==
           b'\x00\x00\x00' + chr(len(obj_type)).encode() + obj_type.encode() +
            ScSD.scalar_serialize(s))
    assert (ser_factory.get_serializer(type(s))(s) ==
            ser_factory.get_apply_serializer(s))
    assert ScSD.scalar_deserialize(ScSD.scalar_serialize(s)) == (s, b'')

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
        ser = NdSD.ndarray_serialize(ndarray)
        assert isinstance(ser, bytes)
        assert np.allclose(NdSD.ndarray_deserialize(NdSD.ndarray_serialize(ndarray))[0], ndarray)

        assert np.allclose(
            ser_factory.get_apply_deserializer(
                ser_factory.get_apply_serializer(ndarray))[0], ndarray)


@pytest.mark.parametrize('obj_list', (['hjk', 'jkgjg', 'lkhlkhl'],  # homogeneous string
                                      [21, 34, -56, 56.7, 1+1j*99],  # homogeneous numbers
                                      [np.array([45, 67, 87654]),
                                       np.array([[45, 67, 87654], [-45, -67, -87654]])],  # homogeneous ndarrays
                                    ))
def test_list_serialization_deserialization(obj_list):
    ser = LSD.list_serialize(obj_list)
    list_back = DeSerializer(ser.to_bytes()).list_deserialization()
    assert isinstance(list_back, list)
    for ind in range(len(obj_list)):
        if isinstance(obj_list[ind], np.ndarray):
            assert np.allclose(obj_list[ind], list_back[ind])
        else:
            assert obj_list[ind] == list_back[ind]