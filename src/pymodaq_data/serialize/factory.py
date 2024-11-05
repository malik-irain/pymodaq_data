from abc import ABCMeta, abstractmethod
from typing import Callable, List, Any, Tuple
from . import utils


class SerializableBase(metaclass=ABCMeta):
    """Base class for a Serializer. """

    @classmethod
    def name(cls):
        """str: the object class name"""
        return cls.__class__.__name__

    @classmethod
    def type(cls):
        """object: the type of the object"""
        raise cls.__class__

    @staticmethod
    @abstractmethod
    def serialize(obj: any) -> bytes:
        "implementation of the serialization into bytes"
        ...

    @staticmethod
    @abstractmethod
    def deserialize(bytes_str: bytes) -> Tuple[Any, bytes]:
        "implementation of the deserialization from bytes"
        ...


class SerializableFactory:
    """The factory class for creating executors"""

    serializable_registry = {}

    @classmethod
    @property
    def name(cls) -> str:
        """str: object identifier, if not specified will be the name of the class"""
        ...

    @classmethod
    def add_type_to_serialize(cls, serialize_method: Callable) -> Callable:
        def wrap(obj: Any):
            bytes_str = b''
            type_as_bytes, len_as_bytes = utils.str_len_to_bytes(obj.__class__.__name__)
            bytes_str += len_as_bytes
            bytes_str += type_as_bytes
            bytes_str += serialize_method(obj)
            return bytes_str
        return wrap

    @classmethod
    def register_from_obj(cls, obj: Any, serialize_method: Callable,
                          deserialize_method: Callable = None):
        """Method to register a serializable object class to the internal registry.

        """
        obj_type = obj.__class__

        if obj_type not in cls.serializable_registry:
            cls.serializable_registry[obj_type] = dict(
                serializer=cls.add_type_to_serialize(serialize_method),
                deserializer=deserialize_method)

    @classmethod
    def register_decorator(cls) -> Callable:
        """Class decorator method to register exporter class to the internal registry. Must be used as
        decorator above the definition of an H5Exporter class. H5Exporter must implement specific class
        attributes and methods, see definition: h5node_exporter.H5Exporter
        See h5node_exporter.H5txtExporter and h5node_exporter.H5txtExporter for usage examples.
        returns:
            the exporter class
        """

        def inner_wrapper(wrapped_class: type(SerializableBase)) -> Callable:
            cls.register_from_type(wrapped_class,
                                   wrapped_class.serialize,
                                   wrapped_class.deserialize)

            # Return wrapped_class
            return wrapped_class
        return inner_wrapper

    @classmethod
    def register_from_type(cls, obj_type: type, serialize_method: Callable,
                           deserialize_method: Callable = None):
        """Method to register a serializable object class to the internal registry.

        """
        if obj_type not in cls.serializable_registry:
            cls.serializable_registry[obj_type] = dict(
                serializer=cls.add_type_to_serialize(serialize_method),
                deserializer=deserialize_method)

    def get_type_from_str(self, obj_type_str) -> type:
        for k in self.serializable_registry:
            if obj_type_str in str(k):
                return k

    def get_serialazables(self) -> List[type]:
        return list(self.serializable_registry.keys())

    def get_serializer(self, obj_type: type) -> Callable:
        entry_dict = self.serializable_registry.get(obj_type, None)
        if entry_dict is not None:
            return entry_dict['serializer']
        else:
            raise NotImplementedError(f'There is no known method to serialize {obj_type}')

    def get_apply_serializer(self, obj: Any) -> bytes:
        entry_dict = self.serializable_registry.get(obj.__class__, None)
        if entry_dict is not None:
            return entry_dict['serializer'](obj)
        else:
            raise NotImplementedError(f'There is no known method to serialize {obj}')

    def get_deserializer(self, obj_type: type) -> Callable:
        entry_dict = self.serializable_registry.get(obj_type, None)
        if entry_dict is not None:
            return entry_dict['deserializer']
        else:
            raise NotImplementedError(f'There is no known method to deserialize an {obj_type} type')

    def get_apply_deserializer(self, bytes_str: bytes) -> Tuple[Any, bytes]:
        obj_type_str, remaining_bytes = self.get_deserializer(str)(bytes_str)

        obj_type = self.get_type_from_str(obj_type_str)
        if obj_type is None:
            raise NotImplementedError(f'There is no known method to deserialize an {obj_type_str} '
                                      f'type')

        entry_dict = self.serializable_registry.get(obj_type, None)
        if entry_dict is not None:
            return entry_dict['deserializer'](remaining_bytes)
        else:
            raise NotImplementedError(f'There is no known method to deserialize an {obj_type_str} '
                                      f'type')