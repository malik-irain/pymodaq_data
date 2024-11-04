from abc import ABCMeta, abstractmethod
from typing import Callable, List, Any


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


class SerializableFactory:
    """The factory class for creating executors"""

    serializable_registry = {}

    @classmethod
    @property
    def name(cls) -> str:
        """str: object identifier, if not specified will be the name of the class"""
        ...

    @classmethod
    def register_from_obj(cls, obj: Any, serialize_method: Callable):
        """Method to register a serializable object class to the internal registry.

        """
        obj_type = obj.__class__

        if obj_type not in cls.serializable_registry:
            cls.serializable_registry[obj_type] = serialize_method

    @classmethod
    def register_from_type(cls, obj_type: type, serialize_method: Callable):
        """Method to register a serializable object class to the internal registry.

        """
        if obj_type not in cls.serializable_registry:
            cls.serializable_registry[obj_type] = serialize_method

    @classmethod
    def get_serialazables(cls) -> List[type]:
        return list(cls.serializable_registry.keys())

    @classmethod
    def get_serializer(cls, obj: Any) -> Callable:
        return cls.serializable_registry.get(obj.__class__, None)

    @classmethod
    def get_apply_serializer(cls, obj: Any) -> bytes:
        serializer = cls.serializable_registry.get(obj.__class__, None)
        if serializer is not None:
            return serializer(obj)
        else:
            raise NotImplementedError(f'There is no known method to serialize {obj}')