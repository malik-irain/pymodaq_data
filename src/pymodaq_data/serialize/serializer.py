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
from pymodaq_data import data as data_mod

from . import utils
from .mysocket import SocketString
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
        bytes_string += StringSerializeDeserialize.serialize(data_type)
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
        data_type, remaining_bytes = StringSerializeDeserialize.deserialize(bytes_str)
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
        bytes_string += StringSerializeDeserialize.serialize(array_type)
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
        ndarray_type, remaining_bytes = StringSerializeDeserialize.deserialize(bytes_str)
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


class Serializer:
    """Used to Serialize to bytes python objects, numpy arrays and PyMoDAQ DataWithAxes and
    DataToExport objects

    Deprecated in PyMoDAQ >= 5 use the SerializerFactory object
    """

    def __init__(self, obj: Optional[Union[ser_factory.get_serialazables()]] = None) -> None:
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

    def bytes_serialization(self, bytes_string_in: bytes) -> bytes:
        """ Convert a bytes string into a bytes message together with the info to convert it back"""
        return ser_factory.get_apply_serializer(bytes_string_in)

    def string_serialization(self, string: str) -> bytes:
        """ Convert a string into a bytes message together with the info to convert it back

        Parameters
        ----------
        string: str

        Returns
        -------
        bytes: the total bytes message to serialize the string
        """
        return ser_factory.get_apply_serializer(string)

    def scalar_serialization(self, scalar: complex) -> bytes:
        """ Convert a scalar into a bytes message together with the info to convert it back

        Parameters
        ----------
        scalar: A python number (complex or subtypes like float and int)

        Returns
        -------
        bytes: the total bytes message to serialize the scalar
        """
        return ser_factory.get_apply_serializer(scalar)

    def ndarray_serialization(self, array: np.ndarray) -> bytes:
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
        return ser_factory.get_apply_serializer(array)

    def object_type_serialization(self, obj: Any) -> bytes:
        """ Convert an object type into a bytes message as a string together with the info to
        convert it back

        """
        return ser_factory.get_apply_serializer(obj.__class__.__name__)

    def axis_serialization(self, axis: data_mod.Axis) -> bytes:
        """ Convert an Axis object into a bytes message together with the info to convert it back

        Parameters
        ----------
        axis: Axis

        Returns
        -------
        bytes: the total bytes message to serialize the Axis

        Notes
        -----

        The bytes sequence is constructed as:

        * serialize the type: 'Axis'
        * serialize the axis label
        * serialize the axis units
        * serialize the axis array
        * serialize the axis
        * serialize the axis spread_order
        """
        return ser_factory.get_apply_serializer(axis)

    def list_serialization(self, list_object: List) -> bytes:
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
        return ser_factory.get_apply_serializer(list_object)

    def dwa_serialization(self, dwa: data_mod.DataWithAxes) -> bytes:
        """ Convert a DataWithAxes into a bytes string

        Parameters
        ----------
        dwa: DataWithAxes

        Returns
        -------
        bytes: the total bytes message to serialize the DataWithAxes

        Notes
        -----
        The bytes sequence is constructed as:

        * serialize the string type: 'DataWithAxes'
        * serialize the timestamp: float
        * serialize the name
        * serialize the source enum as a string
        * serialize the dim enum as a string
        * serialize the distribution enum as a string
        * serialize the list of numpy arrays
        * serialize the list of labels
        * serialize the origin
        * serialize the nav_index tuple as a list of int
        * serialize the list of axis
        * serialize the errors attributes (None or list(np.ndarray))
        * serialize the list of names of extra attributes
        * serialize the extra attributes
        """
        return ser_factory.get_apply_serializer(dwa)

    def dte_serialization(self, dte: data_mod.DataToExport) -> bytes:
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
        return ser_factory.get_apply_serializer(dte)


class DeSerializer:
    """Used to DeSerialize bytes to python objects, numpy arrays and PyMoDAQ Axis,
     DataWithAxes and DataToExport
    objects

    Parameters
    ----------
    bytes_string: bytes or Socket
        the bytes string to deserialize into an object: int, float, string, arrays, list, Axis, DataWithAxes...
        Could also be a Socket object reading bytes from the network having a `get_first_nbytes` method

    See Also
    --------
    :py:class:`~pymodaq_data.serialize.mysocket.SocketString`
    :py:class:`~pymodaq_data.serialize.mysocket.Socket`
    """

    def __init__(self, bytes_string:  Union[bytes, 'Socket'] = None) -> None:
        if isinstance(bytes_string, bytes):
            bytes_string = SocketString(bytes_string)
        self._bytes_string = bytes_string

    @classmethod
    def from_b64_string(cls, b64_string: Union[bytes, str]) -> "DeSerializer":
        return cls(b64decode(b64_string))

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





