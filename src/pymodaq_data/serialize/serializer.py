# -*- coding: utf-8 -*-
"""
Created the 20/10/2023

@author: Sebastien Weber
"""
from base64 import b64encode, b64decode
from enum import Enum
import numbers
from typing import Optional, Tuple, List, Union, TYPE_CHECKING, Any

import numpy as np
from pymodaq.utils import data as data_mod
from pymodaq_data.data import DataWithAxes, DataToExport, Axis, DwaType

from pymodaq_gui.parameter import Parameter, utils as putils, ioxml

import pymodaq.utils.data as data_mod_pymodaq
from . import utils
from ..serialize.factory import SerializableFactory

if TYPE_CHECKING:
    from pymodaq.utils.tcp_ip.mysocket import Socket


ser_factory = SerializableFactory()


class StringSerializeDeserialize:

    @staticmethod
    def serialize(string: str) -> bytes:
        """ Convert a string into a bytes message together with the info to convert it back

        Parameters
        ----------
        string: str

        Returns
        -------
        bytes: the total bytes message to serialize the string
        """
        bytes_string = b''
        cmd_bytes, cmd_length_bytes = utils.str_len_to_bytes(string)
        bytes_string += cmd_length_bytes
        bytes_string += cmd_bytes
        return bytes_string

    @staticmethod
    def deserialize(bytes_str) -> Tuple[str, bytes]:
        """Convert bytes into a str object

        Convert first the fourth first bytes into an int encoding the length of the string to decode

        Returns
        -------
        str: the decoded string
        bytes: the remaining bytes string if any
        """
        string_len, remaining_bytes = utils.get_int_from_bytes(bytes_str)
        str_bytes,  remaining_bytes = utils.split_nbytes(remaining_bytes, string_len)
        str_obj = utils.bytes_to_string(str_bytes)
        return str_obj, remaining_bytes


class BytesSerializeDeserialize:
    @staticmethod
    def serialize(some_bytes: bytes) -> bytes:
        bytes_string = b''
        bytes_string += utils.int_to_bytes(len(some_bytes))
        bytes_string += some_bytes
        return bytes_string

    @staticmethod
    def deserialize(bytes_str: bytes) -> Tuple[bytes, bytes]:
        bytes_len, remaining_bytes = utils.get_int_from_bytes(bytes_str)
        bytes_str, remaining_bytes = utils.split_nbytes(remaining_bytes, bytes_len)
        return bytes_str, remaining_bytes


class ScalarSerializeDeserialize:
    @staticmethod
    def serialize(scalar: complex) -> bytes:
        """ Convert a scalar into a bytes message together with the info to convert it back

        Parameters
        ----------
        scalar: A python number (complex or subtypes like float and int)

        Returns
        -------
        bytes: the total bytes message to serialize the scalar
        """
        if not isinstance(scalar, numbers.Number):
            # type hint is complex, instance comparison Number
            raise TypeError(f'{scalar} should be an integer or a float, not a {type(scalar)}')
        scalar_array = np.array([scalar])
        data_type = scalar_array.dtype.descr[0][1]
        data_bytes = scalar_array.tobytes()

        bytes_string = b''
        bytes_string += StringSerializeDeserialize.string_serialize(data_type)
        bytes_string += utils.int_to_bytes(len(data_bytes))
        bytes_string += data_bytes
        return bytes_string

    @staticmethod
    def deserialize(bytes_str: bytes) -> Tuple[complex, bytes]:
        """Convert bytes into a python object of type (float, int, complex or boolean)

        Get first the data type from a string deserialization, then the data length and finally convert this
        length of bytes into an object of type (float, int, complex or boolean)

        Returns
        -------
        numbers.Number: the decoded number
        bytes: the remaining bytes string if any
        """
        data_type, remaining_bytes = StringSerializeDeserialize.string_deserialize(bytes_str)
        data_len, remaining_bytes = utils.get_int_from_bytes(remaining_bytes)
        number_bytes, remaining_bytes = utils.split_nbytes(remaining_bytes, data_len)
        number = np.frombuffer(number_bytes, dtype=data_type)[0]
        if 'f' in data_type:
            number = float(number)  # because one get numpy float type
        elif 'i' in data_type:
            number = int(number)  # because one get numpy int type
        elif 'c' in data_type:
            number = complex(number)  # because one get numpy complex type
        elif 'b' in data_type:
            number = bool(number)  # because one get numpy complex type
        return number, remaining_bytes


class NdArraySerializeDeserialize:

    @staticmethod
    def serialize(array: np.ndarray) -> bytes:
        """ Convert a ndarray into a bytes message together with the info to convert it back

        Parameters
        ----------
        array: np.ndarray

        Returns
        -------
        bytes: the total bytes message to serialize the scalar

        Notes
        -----

        The bytes sequence is constructed as:

        * get data type as a string
        * reshape array as 1D array and get the array dimensionality (len of array's shape)
        * convert Data array as bytes
        * serialize data type
        * serialize data length
        * serialize data shape length
        * serialize all values of the shape as integers converted to bytes
        * serialize array as bytes
        """
        if not isinstance(array, np.ndarray):
            raise TypeError(f'{array} should be an numpy array, not a {type(array)}')
        array_type = array.dtype.descr[0][1]
        array_shape = array.shape

        array = array.reshape(array.size)
        array_bytes = array.tobytes()
        bytes_string = b''
        bytes_string += StringSerializeDeserialize.string_serialize(array_type)
        bytes_string += utils.int_to_bytes(len(array_bytes))
        bytes_string += utils.int_to_bytes(len(array_shape))
        for shape_elt in array_shape:
            bytes_string += utils.int_to_bytes(shape_elt)
        bytes_string += array_bytes
        return bytes_string

    @staticmethod
    def deserialize(bytes_str: bytes) -> Tuple[np.ndarray, bytes]:
        """Convert bytes into a numpy ndarray object

        Convert the first bytes into a ndarray reading first information about the array's data

        Returns
        -------
        ndarray: the decoded numpy array
        bytes: the remaining bytes string if any
        """
        ndarray_type, remaining_bytes = StringSerializeDeserialize.string_deserialize(bytes_str)
        ndarray_len, remaining_bytes = utils.get_int_from_bytes(remaining_bytes)
        shape_len, remaining_bytes = utils.get_int_from_bytes(remaining_bytes)
        shape = []
        for ind in range(shape_len):
            shape_elt, remaining_bytes = utils.get_int_from_bytes(remaining_bytes)
            shape.append(shape_elt)

        ndarray_bytes, remaining_bytes = utils.split_nbytes(remaining_bytes, ndarray_len)
        ndarray = np.frombuffer(ndarray_bytes, dtype=ndarray_type)
        ndarray = ndarray.reshape(tuple(shape))
        ndarray = np.atleast_1d(ndarray)  # remove singleton dimensions
        return ndarray, remaining_bytes


class ListSerializeDeserialize:
    @staticmethod
    def serialize(list_object: List) -> bytes:
        """ Convert a list of objects into a bytes message together with the info to convert it back

        Parameters
        ----------
        list_object: list
            the list could contain whatever objects are registered in the SerializableFactory

        Returns
        -------
        bytes: the total bytes message to serialize the list of objects

        Notes
        -----

        The bytes sequence is constructed as:
        * the length of the list

        Then for each object:
        * use the serialization method adapted to each object in the list
        """
        if not isinstance(list_object, list):
            raise TypeError(f'{list_object} should be a list, not a {type(list_object)}')

        bytes_string = b''
        bytes_string += utils.int_to_bytes(len(list_object))
        for obj in list_object:
            bytes_string += ser_factory.get_apply_serializer(obj)
        return bytes_string

    @staticmethod
    def deserialize(bytes_str: bytes) -> Tuple[List[Any], bytes]:
        """Convert bytes into a list of objects

        Convert the first bytes into a list reading first information about the list elt types, length ...

        Returns
        -------
        list: the decoded list
        bytes: the remaining bytes string if any
        """
        list_obj = []
        list_len, remaining_bytes = utils.get_int_from_bytes(bytes_str)

        for ind in range(list_len):
            obj, remaining_bytes = ser_factory.get_apply_deserializer(remaining_bytes)
            list_obj.append(obj)
        return list_obj, remaining_bytes


ser_factory.register_from_type(bytes,
                                       BytesSerializeDeserialize.serialize,
                                       BytesSerializeDeserialize.deserialize)
ser_factory.register_from_type(str, StringSerializeDeserialize.serialize,
                                       StringSerializeDeserialize.deserialize)
ser_factory.register_from_type(int, ScalarSerializeDeserialize.serialize,
                                       ScalarSerializeDeserialize.deserialize)
ser_factory.register_from_type(float, ScalarSerializeDeserialize.serialize,
                                       ScalarSerializeDeserialize.deserialize)
ser_factory.register_from_obj(1 + 1j, ScalarSerializeDeserialize.serialize,
                                      ScalarSerializeDeserialize.deserialize)
ser_factory.register_from_type(bool, ScalarSerializeDeserialize.serialize,
                                       ScalarSerializeDeserialize.deserialize)
ser_factory.register_from_obj(np.array([0, 1]),
                                      NdArraySerializeDeserialize.serialize,
                                      NdArraySerializeDeserialize.deserialize)
ser_factory.register_from_type(list,
                                       ListSerializeDeserialize.serialize,
                                       ListSerializeDeserialize.deserialize)

SERIALIZABLE = Union[
    # native
    bool,
    bytes,
    str,
    complex,  # float and int are subtypes for type hinting
    float,
    int,
    list,
    # numpy
    np.ndarray,
    # pymodaq
    Axis,
    DataWithAxes,
    DataToExport,
    putils.ParameterWithPath,
]


class SerializableTypes(Enum):
    """Type names of serializable types"""
    BOOL = "bool"
    BYTES = "bytes"
    STRING = "string"
    SCALAR = "scalar"
    LIST = "list"
    ARRAY = "array"
    AXIS = "axis"
    DATA_WITH_AXES = "dwa"
    DATA_TO_EXPORT = "dte"
    PARAMETER = "parameter"


class SocketString:
    """Mimic the Socket object but actually using a bytes string not a socket connection

    Implements a minimal interface of two methods

    Parameters
    ----------
    bytes_string: bytes

    See Also
    --------
    :class:`~pymodaq.utils.tcp_ip.mysocket.Socket`
    """
    def __init__(self, bytes_string: bytes):
        self._bytes_string = bytes_string

    def check_received_length(self, length: int) -> bytes:
        """
        Make sure all bytes (length) that should be received are received through the socket.

        Here just read the content of the underlying bytes string

        Parameters
        ----------
        length: int
            The number of bytes to be read from the socket

        Returns
        -------
        bytes
        """
        data = self._bytes_string[0:length]
        self._bytes_string = self._bytes_string[length:]
        return data

    def get_first_nbytes(self, length: int) -> bytes:
        """ Read the first N bytes from the socket

        Parameters
        ----------
        length: int
            The number of bytes to be read from the socket

        Returns
        -------
        bytes
            the read bytes string
        """
        return self.check_received_length(length)


class Serializer:
    """Used to Serialize to bytes python objects, numpy arrays and PyMoDAQ DataWithAxes and
    DataToExport objects"""

    def __init__(self, obj: Optional[SERIALIZABLE] = None) -> None:
        self._bytes_string = b''
        self._obj = obj

    def to_bytes(self):
        """ Generic method to obtain the bytes string from various objects

        Compatible objects are:

        * :class:`bytes`
        * :class:`numbers.Number`
        * :class:`str`
        * :class:`numpy.ndarray`
        * :class:`~pymodaq.utils.data.Axis`
        * :class:`~pymodaq.utils.data.DataWithAxes` and sub-flavours
        * :class:`~pymodaq.utils.data.DataToExport`
        * :class:`list` of any objects above

        """
        return ser_factory.get_apply_serializer(self._obj)

    def to_b64_string(self) -> str:
        b = self.to_bytes()
        return b64encode(b).decode()


class DeSerializer:
    """Used to DeSerialize bytes to python objects, numpy arrays and PyMoDAQ Axis, DataWithAxes and DataToExport
    objects

    Parameters
    ----------
    bytes_string: bytes or Socket
        the bytes string to deserialize into an object: int, float, string, arrays, list, Axis, DataWithAxes...
        Could also be a Socket object reading bytes from the network having a `get_first_nbytes` method

    See Also
    --------
    :py:class:`~pymodaq.utils.tcp_ip.serializer.SocketString`
    :py:class:`~pymodaq.utils.tcp_ip.mysocket.Socket`
    """

    def __init__(self, bytes_string:  Union[bytes, 'Socket'] = None) -> None:
        if isinstance(bytes_string, bytes):
            bytes_string = SocketString(bytes_string)
        self._bytes_string = bytes_string


    def _int_deserialization(self) -> int:
        """Convert the fourth first bytes into an unsigned integer to be used internally. For integer serialization
        use scal_serialization"""
        int_obj = self.bytes_to_int(self._bytes_string.get_first_nbytes(4))
        return int_obj

    @classmethod
    def from_b64_string(cls, b64_string: Union[bytes, str]) -> "DeSerializer":
        return cls(b64decode(b64_string))



    def axis_deserialization(self) -> Axis:
        """Convert bytes into an Axis object

        Convert the first bytes into an Axis reading first information about the Axis

        Returns
        -------
        Axis: the decoded Axis
        """

        class_name = self.string_deserialization()
        if class_name != Axis.__name__:
            raise TypeError(f'Attempting to deserialize an Axis but got the bytes for a {class_name}')
        axis_label = self.string_deserialization()
        axis_units = self.string_deserialization()
        axis_array = self.ndarray_deserialization()
        axis_index = self.scalar_deserialization()
        axis_spread_order = self.scalar_deserialization()

        axis = Axis(axis_label, axis_units, data=axis_array, index=axis_index,
                    spread_order=axis_spread_order)
        return axis

    def dwa_deserialization(self) -> DataWithAxes:
        """Convert bytes into a DataWithAxes object

        Convert the first bytes into a DataWithAxes reading first information about the underlying data

        Returns
        -------
        DataWithAxes: the decoded DataWithAxes
        """
        class_name = self.string_deserialization()
        if class_name not in DwaType.names():
            raise TypeError(f'Attempting to deserialize a DataWithAxes flavor but got the bytes for a {class_name}')
        timestamp = self.scalar_deserialization()
        try:
            dwa_class = getattr(data_mod, class_name)
        except AttributeError:  # in case it is a particular data object defined only in pymodaq
            dwa_class = getattr(data_mod_pymodaq, class_name)
        dwa = dwa_class(self.string_deserialization(),
                        source=self.string_deserialization(),
                        dim=self.string_deserialization(),
                        distribution=self.string_deserialization(),
                        data=self.list_deserialization(),
                        units=self.string_deserialization(),
                        labels=self.list_deserialization(),
                        origin=self.string_deserialization(),
                        nav_indexes=tuple(self.list_deserialization()),
                        axes=self.list_deserialization(),
                        )
        errors = self.list_deserialization()
        if len(errors) != 0:
            dwa.errors = errors
        dwa.extra_attributes = self.list_deserialization()
        for attribute in dwa.extra_attributes:
            setattr(dwa, attribute, self.type_and_object_deserialization())

        dwa.timestamp = timestamp
        return dwa

    def dte_deserialization(self) -> DataToExport:
        """Convert bytes into a DataToExport object

        Convert the first bytes into a DataToExport reading first information about the underlying data

        Returns
        -------
        DataToExport: the decoded DataToExport
        """
        class_name = self.string_deserialization()
        if class_name != DataToExport.__name__:
            raise TypeError(f'Attempting to deserialize a DataToExport but got the bytes for a {class_name}')
        timestamp = self.scalar_deserialization()
        dte = DataToExport(self.string_deserialization(),
                           data=self.list_deserialization(),
                           )
        dte.timestamp = timestamp
        return dte
