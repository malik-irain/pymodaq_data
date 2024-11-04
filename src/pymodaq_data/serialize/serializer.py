# -*- coding: utf-8 -*-
"""
Created the 20/10/2023

@author: Sebastien Weber
"""
from base64 import b64encode, b64decode
from enum import Enum
import numbers
from typing import Optional, Tuple, List, Union, TYPE_CHECKING

import numpy as np
from pymodaq.utils import data as data_mod
from pymodaq_data.data import DataWithAxes, DataToExport, Axis, DwaType

from pymodaq_gui.parameter import Parameter, utils as putils, ioxml

import pymodaq.utils.data as data_mod_pymodaq
from ..serialize.factory import SerializableFactory, SerializableBase

if TYPE_CHECKING:
    from pymodaq.utils.tcp_ip.mysocket import Socket


class BaseSerializer:
    """ Basic functionalities to serialize objects"""

    def __init__(self):
        self._bytes_string = b''

    @staticmethod
    def int_to_bytes(an_integer: int) -> bytes:
        """Convert an unsigned integer into a byte array of length 4 in big endian

        Parameters
        ----------
        an_integer: int

        Returns
        -------
        bytearray
        """
        if not isinstance(an_integer, int):
            raise TypeError(f'{an_integer} should be an integer, not a {type(an_integer)}')
        elif an_integer < 0:
            raise ValueError('Can only serialize unsigned integer using this method')
        return an_integer.to_bytes(4, 'big')

    @staticmethod
    def str_to_bytes(message: str) -> bytes:
        if not isinstance(message, str):
            raise TypeError('Can only serialize str object using this method')
        return message.encode()

    @classmethod
    def str_len_to_bytes(cls, message: Union[str, bytes]) -> Tuple[bytes, bytes]:
        """ Convert a string and its length to two bytes
        Parameters
        ----------
        message: str
            the message to convert

        Returns
        -------
        bytes: message converted as a byte array
        bytes: length of the message byte array, itself as a byte array of length 4
        """

        if not isinstance(message, str) and not isinstance(message, bytes):
            message = str(message)
        if not isinstance(message, bytes):
            message = cls.str_to_bytes(message)
        return message, cls.int_to_bytes(len(message))


def bytes_serialize(some_bytes: bytes) -> bytes:
    bytes_string = b''
    bytes_string += BaseSerializer.int_to_bytes(len(some_bytes))
    bytes_string += some_bytes
    return bytes_string


def string_serialize(string: str) -> bytes:
    """ Convert a string into a bytes message together with the info to convert it back

    Parameters
    ----------
    string: str

    Returns
    -------
    bytes: the total bytes message to serialize the string
    """
    bytes_string = b''
    cmd_bytes, cmd_length_bytes = BaseSerializer.str_len_to_bytes(string)
    bytes_string += cmd_length_bytes
    bytes_string += cmd_bytes
    return bytes_string


def scalar_serialize(scalar: complex) -> bytes:
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
    bytes_string += string_serialize(data_type)
    bytes_string += BaseSerializer.int_to_bytes(len(data_bytes))
    bytes_string += data_bytes
    return bytes_string


def ndarray_serialize(array: np.ndarray) -> bytes:
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
    bytes_string += string_serialize(array_type)
    bytes_string += BaseSerializer.int_to_bytes(len(array_bytes))
    bytes_string += BaseSerializer.int_to_bytes(len(array_shape))
    for shape_elt in array_shape:
        bytes_string += BaseSerializer.int_to_bytes(shape_elt)
    bytes_string += array_bytes
    return bytes_string


def list_serialize(self, list_object: List) -> bytes:
    """ Convert a list of objects into a bytes message together with the info to convert it back

    Parameters
    ----------
    list_object: list
        the list could contains either scalars, strings or ndarrays or Axis objects or DataWithAxis objects
        module

    Returns
    -------
    bytes: the total bytes message to serialize the list of objects

    Notes
    -----

    The bytes sequence is constructed as:
    * the length of the list

    Then for each object:

    * get data type as a string
    * use the serialization method adapted to each object in the list
    """
    if not isinstance(list_object, list):
        raise TypeError(f'{list_object} should be a list, not a {type(list_object)}')

    bytes_string = b''

    bytes_string += BaseSerializer.int_to_bytes(len(list_object))
    for obj in list_object:
        bytes_string += string_serialize(obj.__class__.__name__)
        bytes_string += SerializableFactory.get_apply_serializer(obj)
    self._bytes_string += bytes_string
    return bytes_string


SerializableFactory.register_from_type(bytes, bytes_serialize)
SerializableFactory.register_from_type(str, string_serialize)
SerializableFactory.register_from_type(int, scalar_serialize)
SerializableFactory.register_from_type(float, scalar_serialize)
SerializableFactory.register_from_obj(1 + 1j, scalar_serialize)
SerializableFactory.register_from_type(bool, scalar_serialize)
SerializableFactory.register_from_type(np.ndarray, ndarray_serialize)

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


class Serializer(BaseSerializer):
    """Used to Serialize to bytes python objects, numpy arrays and PyMoDAQ DataWithAxes and
    DataToExport objects"""

    def __init__(self, obj: Optional[SERIALIZABLE] = None) -> None:
        super().__init__()
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
        return SerializableFactory.get_apply_serializer(self._obj)

    def to_b64_string(self) -> str:
        b = self.to_bytes()
        return b64encode(b).decode()

    def type_and_object_serialization(self, obj: Optional[SERIALIZABLE] = None) -> bytes:
        """Serialize an object with its type, such that it can be retrieved by
        `DeSerializer.type_and_object_deserialization`.
        """

        if obj is None and self._obj is not None:
            obj = self._obj

        bytes_string = b''
        bytes_string += SerializableFactory.get_apply_serializer(obj.__class__.__name__)
        bytes_string += SerializableFactory.get_apply_serializer(obj)
        return bytes_string

    def dte_serialization(self, dte: DataToExport) -> bytes:
        """ Convert a DataToExport into a bytes string

        Parameters
        ----------
        dte: DataToExport

        Returns
        -------
        bytes: the total bytes message to serialize the DataToExport

        Notes
        -----
        The bytes sequence is constructed as:

        * serialize the string type: 'DataToExport'
        * serialize the timestamp: float
        * serialize the name
        * serialize the list of DataWithAxes
        """
        if not isinstance(dte, DataToExport):
            raise TypeError(f'{dte} should be a DataToExport, not a {type(dte)}')

        bytes_string = b''
        bytes_string += self.object_type_serialization(dte)
        bytes_string += self.scalar_serialization(dte.timestamp)
        bytes_string += self.string_serialization(dte.name)
        bytes_string += self.list_serialization(dte.data)
        self._bytes_string += bytes_string
        return bytes_string


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

    @classmethod
    def from_b64_string(cls, b64_string: Union[bytes, str]) -> "DeSerializer":
        return cls(b64decode(b64_string))

    @staticmethod
    def bytes_to_string(message: bytes) -> str:
        return message.decode()

    @staticmethod
    def bytes_to_int(bytes_string: bytes) -> int:
        """Convert a bytes of length 4 into an integer"""
        if not isinstance(bytes_string, bytes):
            raise TypeError(f'{bytes_string} should be an bytes string, not a {type(bytes_string)}')
        assert len(bytes_string) == 4
        return int.from_bytes(bytes_string, 'big')

    @staticmethod
    def bytes_to_scalar(data: bytes, dtype: np.dtype) -> complex:
        """Convert bytes to a scalar given a certain numpy dtype

        Parameters
        ----------
        data: bytes
        dtype:np.dtype

        Returns
        -------
        numbers.Number
        """
        return np.frombuffer(data, dtype=dtype)[0]

    @staticmethod
    def bytes_to_nd_array(data: bytes, dtype: np.dtype, shape: Tuple[int]) -> np.ndarray:
        """Convert bytes to a ndarray given a certain numpy dtype and shape

        Parameters
        ----------
        data: bytes
        dtype: np.dtype
        shape: tuple of int

        Returns
        -------
        np.ndarray
        """
        array = np.frombuffer(data, dtype=dtype)
        array = array.reshape(tuple(shape))
        array = np.atleast_1d(array)  # remove singleton dimensions but keeping ndarrays
        return array

    def _int_deserialization(self) -> int:
        """Convert the fourth first bytes into an unsigned integer to be used internally. For integer serialization
        use scal_serialization"""
        int_obj = self.bytes_to_int(self._bytes_string.get_first_nbytes(4))
        return int_obj

    def bytes_deserialization(self) -> bytes:
        bstring_len = self._int_deserialization()
        bstr = self._bytes_string.get_first_nbytes(bstring_len)
        return bstr

    def string_deserialization(self) -> str:
        """Convert bytes into a str object

        Convert first the fourth first bytes into an int encoding the length of the string to decode

        Returns
        -------
        str: the decoded string
        """
        string_len = self._int_deserialization()
        str_obj = self._bytes_string.get_first_nbytes(string_len).decode()
        return str_obj

    def scalar_deserialization(self) -> complex:
        """Convert bytes into a python number object

        Get first the data type from a string deserialization, then the data length and finally convert this
        length of bytes into a number (float, int)

        Returns
        -------
        numbers.Number: the decoded number
        """
        data_type = self.string_deserialization()
        data_len = self._int_deserialization()
        number = np.frombuffer(self._bytes_string.get_first_nbytes(data_len), dtype=data_type)[0]
        if 'f' in data_type:
            number = float(number)  # because one get numpy float type
        elif 'i' in data_type:
            number = int(number)  # because one get numpy int type
        elif 'c' in data_type:
            number = complex(number)  # because one get numpy complex type
        return number

    def boolean_deserialization(self) -> bool:
        """Convert bytes into a boolean object

        Get first the data type from a string deserialization, then the data length and finally
        convert this length of bytes into a number (float, int) and cast it to a bool

        Returns
        -------
        bool: the decoded boolean
        """
        number = self.scalar_deserialization()
        return bool(number)

    def ndarray_deserialization(self) -> np.ndarray:
        """Convert bytes into a numpy ndarray object

        Convert the first bytes into a ndarray reading first information about the array's data

        Returns
        -------
        ndarray: the decoded numpy array
        """
        ndarray_type = self.string_deserialization()
        ndarray_len = self._int_deserialization()
        shape_len = self._int_deserialization()
        shape = []
        for ind in range(shape_len):
            shape_elt = self._int_deserialization()
            shape.append(shape_elt)

        ndarray = np.frombuffer(self._bytes_string.get_first_nbytes(ndarray_len), dtype=ndarray_type)
        ndarray = ndarray.reshape(tuple(shape))
        ndarray = np.atleast_1d(ndarray)  # remove singleton dimensions
        return ndarray

    def type_and_object_deserialization(self) -> SERIALIZABLE:
        """ Deserialize specific objects from their binary representation (inverse of `Serializer.type_and_object_serialization`).

        See Also
        --------
        Serializer.dwa_serialization, Serializer.dte_serialization

        """
        obj_type = self.string_deserialization()
        elt = None
        if obj_type == SerializableTypes.SCALAR.value:
            elt = self.scalar_deserialization()
        elif obj_type == SerializableTypes.STRING.value:
            elt = self.string_deserialization()
        elif obj_type == SerializableTypes.BYTES.value:
            elt = self.bytes_deserialization()
        elif obj_type == SerializableTypes.ARRAY.value:
            elt = self.ndarray_deserialization()
        elif obj_type == SerializableTypes.DATA_WITH_AXES.value:
            elt = self.dwa_deserialization()
        elif obj_type == SerializableTypes.DATA_TO_EXPORT.value:
            elt = self.dte_deserialization()
        elif obj_type == SerializableTypes.AXIS.value:
            elt = self.axis_deserialization()
        elif obj_type == SerializableTypes.BOOL.value:
            elt = self.boolean_deserialization()
        elif obj_type == SerializableTypes.LIST.value:
            elt = self.list_deserialization()
        elif obj_type == SerializableTypes.PARAMETER.value:
            elt = self.parameter_deserialization()
        else:
            print(f'invalid object type {obj_type}')
            elt = None  # desired behavior?
        return elt

    def list_deserialization(self) -> list:
        """Convert bytes into a list of homogeneous objects

        Convert the first bytes into a list reading first information about the list elt types, length ...

        Returns
        -------
        list: the decoded list
        """
        list_obj = []
        list_len = self._int_deserialization()

        for ind in range(list_len):
            list_obj.append(self.type_and_object_deserialization())
        return list_obj

    def parameter_deserialization(self) -> putils.ParameterWithPath:
        path = self.list_deserialization()
        param_as_xml = self.string_deserialization()
        param_dict = ioxml.XML_string_to_parameter(param_as_xml)
        param_obj = Parameter(**param_dict[0])
        return putils.ParameterWithPath(param_obj, path)

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
