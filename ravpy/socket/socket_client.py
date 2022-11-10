import socketio
from socketio.exceptions import ConnectionError
from ravpy.config import RAVENVERSE_URL, TYPE, RAVENVERSE_FTP_URL
from ravpy.globals import g


class SocketNamespace(socketio.ClientNamespace):
    def on_connect(self):
        g.logger.debug('Connected to Ravenverse successfully!')

    def on_disconnect(self):
        g.logger.debug('Disconnected from the server')

    def on_message(self, data):
        g.logger.debug('Message received:', data)

    def on_result(self, data):
        g.logger.debug(data)

    def on_connect_error(self, e):
        g.logger.debug("Error:{}".format(str(e)))


class SocketClient(object):
    def __init__(self):
        self._client = socketio.Client(logger=False, request_timeout=100, engineio_logger=False)
        self._client.register_namespace(SocketNamespace('/client'))

    def connect(self, token):
        g.logger.debug("Connecting to Ravenverse...")
        g.logger.debug("Ravenverse url: {}?type={}".format(RAVENVERSE_URL, TYPE))
        g.logger.debug("Ravenverse FTP host: {}".format(RAVENVERSE_FTP_URL))
        auth_headers = {"token": token}

        try:
            self._client.connect(url="{}?type={}".format(RAVENVERSE_URL, TYPE),
                                 auth=auth_headers,
                                 transports=['websocket'],
                                 namespaces=['/client'], wait_timeout=100)
        except ConnectionError as e:
            g.logger.error("Error: Unable to connect to Ravenverse. "
                           "Make sure you are using the right hostname and port. \n{}".format(e))
            self._client.disconnect()
        except Exception as e:
            g.logger.error("Error: Unable to connect to Ravenverse. "
                           "Make sure you are using the right hostname and port. \n{}".format(e))
            self._client.disconnect()

    @property
    def client(self):
        return self._client

    def disconnect(self):
        self._client.disconnect()
