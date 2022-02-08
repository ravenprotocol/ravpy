import socketio

from .config import SOCKET_SERVER_HOST, SOCKET_SERVER_PORT, CID
from .utils import Singleton


def get_client():
    client = socketio.Client(logger=True)
    client.connect(url="http://{}:{}?cid={}".format(SOCKET_SERVER_HOST, SOCKET_SERVER_PORT, CID),
                   namespaces=['/client'])
    return client


@Singleton
class Globals(object):
    def __init__(self):
        self._client = get_client()

    @property
    def client(self):
        return self._client


g = Globals.Instance()
