from typing import Union

from pymodaq_data.serialize.serializer import Serializer
from pymodaq_utils.mysocket import Socket

from .factory import SerializableFactory

ser_factory = SerializableFactory()


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


class Socket(Socket):
    """Custom Socket wrapping the built-in one and added functionalities to
    make sure message have been sent and received entirely"""

    def check_sended_with_serializer(self,
                                     obj: Union[ser_factory.get_serialazables()]):
        """ Convenience function to convert permitted objects to bytes and then use
        the check_sended method

        For a list of allowed objects, see :meth:`Serializer.to_bytes`
        """
        self.check_sended(Serializer(obj).to_bytes())
