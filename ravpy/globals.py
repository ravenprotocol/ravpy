import socketio

from .config import SOCKET_SERVER_HOST, SOCKET_SERVER_PORT, TYPE
from .utils import Singleton


def get_client(cid):
    client = socketio.Client(logger=False)
    try:
        client.connect(url="http://{}:{}?cid={}&type={}".format(SOCKET_SERVER_HOST, SOCKET_SERVER_PORT, cid, TYPE),
                       namespaces=['/client'])
        return client
    except Exception as e:
        print("Exception:{}".format(e))
        print("Unable to connect to ravsock. Make sure you are using the right hostname and port")
        exit()


@Singleton
class Globals(object):
    def __init__(self):
        self._client = None
        self._cid = None
        self._timeoutId = None
        self._ops = {}
        self._opTimeout = 2000
        self._initialTimeout = 1000
        self._outputs = {}

    @property
    def cid(self):
        return self._cid

    @property
    def timeoutId(self):
        return self._timeoutId

    @property
    def ops(self):
        return self._ops

    @property
    def opTimeout(self):
        return self._opTimeout

    @property
    def initialTimeout(self):
        return self._initialTimeout

    @property
    def outputs(self):
        return self._outputs

    @cid.setter
    def cid(self, cid):
        self._cid = cid

    @property
    def client(self):
        if self._cid is None:
            print("Set cid first")
            exit()

        if self._client is not None:
            return self._client

        if self._client is None and self._cid is not None:
            self._client = get_client(self._cid)
            return self._client


g = Globals.Instance()