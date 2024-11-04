from pymodaq_data.serialize import serializer
from pymodaq_data.serialize.factory import SerializableFactory

class TestBaseSerialize:

    def test_bytes_serialization(self):
        b = b'kjlksjdf'
        assert serializer.bytes_serialize(b) == b'\x00\x00\x00\x08' + b
        assert serializer.bytes_serialize(b) == SerializableFactory.get_serializer(b)(b)
        assert serializer.bytes_serialize(b) == SerializableFactory.get_apply_serializer(b)

    def test_string_serialization(self):
        s = 'ert'
        assert serializer.SerializableFactory.get_serializer(s)(s)
        assert serializer.string_serialize(s) ==  b'\x00\x00\x00\x03' + s.encode()
        assert serializer.string_serialize(s) == SerializableFactory.get_serializer(s)(s)
        assert serializer.string_serialize(s) == SerializableFactory.get_apply_serializer(s)

    def test_scalar_serialization(self):
        s = 23
        assert serializer.scalar_serialize(s) == b'\x00\x00\x00\x03<i4\x00\x00\x00\x04\x17\x00\x00\x00'
        assert serializer.scalar_serialize(s) == SerializableFactory.get_serializer(s)(s)
        assert serializer.scalar_serialize(s) == SerializableFactory.get_apply_serializer(s)

        s = -3.8
        assert serializer.scalar_serialize(s) == b'\x00\x00\x00\x03<f8\x00\x00\x00\x08ffffff\x0e\xc0'
        assert serializer.scalar_serialize(s) == SerializableFactory.get_serializer(s)(s)
        assert serializer.scalar_serialize(s) == SerializableFactory.get_apply_serializer(s)

        s = 4 -2.5j
        assert serializer.scalar_serialize(s) == b'\x00\x00\x00\x04<c16\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x10@\x00\x00\x00\x00\x00\x00\x04\xc0'
        assert serializer.scalar_serialize(s) == SerializableFactory.get_serializer(s)(s)
        assert serializer.scalar_serialize(s) == SerializableFactory.get_apply_serializer(s)

        s = True
        assert serializer.scalar_serialize(s) == b'\x00\x00\x00\x03|b1\x00\x00\x00\x01\x01'
        assert serializer.scalar_serialize(s) == SerializableFactory.get_serializer(s)(s)
        assert serializer.scalar_serialize(s) == SerializableFactory.get_apply_serializer(s)

        s = False
        assert serializer.scalar_serialize(s) == b'\x00\x00\x00\x03|b1\x00\x00\x00\x01\x00'
        assert serializer.scalar_serialize(s) == SerializableFactory.get_serializer(s)(s)
        assert serializer.scalar_serialize(s) == SerializableFactory.get_apply_serializer(s)