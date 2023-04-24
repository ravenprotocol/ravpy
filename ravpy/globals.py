import os
import socketio
from .config import RAVENVERSE_URL, TYPE, RAVENVERSE_FTP_URL, FTP_TEMP_FILES_FOLDER
from .logger import get_logger
from .singleton_utils import Singleton

async def exit_handler():
    g.logger.debug('Application is Closing!')
    if g.client is not None:
        g.logger.debug("Disconnecting...")
        if g.client.connected:
            g.client.emit("disconnect", namespace="/client")

    dir = FTP_TEMP_FILES_FOLDER
    if os.path.exists(dir):
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))

def get_client():
    """
    Connect to Ravebverse and return socket client instance
    :param ravenverse_token: authentication token
    :return: socket client
    """
    g.logger.debug("Connecting to Ravenverse...")

    client = socketio.AsyncClient(logger=False, request_timeout=100, engineio_logger=False)

    @client.on('error', namespace='/client')
    async def check_error(d):
        g.logger.error("Connection error:a{}".format(d))
        await client.disconnect()
        os._exit(1)

    @client.on('invalid_graph', namespace='/client')
    async def invalid_graph(d):
        g.logger.error("Invalid Graph error:{}".format(d))
        await exit_handler()
        os._exit(1)

    return client

@Singleton
class Globals(object):
    def __init__(self):
        self._client = None
        self._timeoutId = None
        self._ops = {}
        self._opTimeout = 300#5000
        self._initialTimeout = 100
        self._param_queue = {}
        self._ftp_client = None
        self._delete_files_list = []
        self._has_subgraph = False
        self._is_downloading = False
        self._is_uploading = False
        self._noop_counter = 0
        self._ftp_upload_blocksize = 8192
        self._upload_speed = 0
        self._ftp_download_blocksize = 8192
        self._download_speed = 0
        self._ping_timeout_counter = 0
        self._error = False
        self._ravenverse_token = None
        self._logger = get_logger()
        self._dashboard_data = [['Subgraph ID', 'Graph ID', 'Status']]

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
    def param_queue(self):
        return self._param_queue
    
    @param_queue.setter
    def param_queue(self, param_queue):
        self._param_queue = param_queue

    @property
    def ftp_upload_blocksize(self):
        return self._ftp_upload_blocksize

    @ftp_upload_blocksize.setter
    def ftp_upload_blocksize(self, ftp_upload_blocksize):
        self._ftp_upload_blocksize = ftp_upload_blocksize

    @property
    def ftp_download_blocksize(self):
        return self._ftp_download_blocksize

    @ftp_download_blocksize.setter
    def ftp_download_blocksize(self, ftp_download_blocksize):
        self._ftp_download_blocksize = ftp_download_blocksize

    @property
    def upload_speed(self):
        return self._upload_speed

    @upload_speed.setter
    def upload_speed(self, upload_speed):
        self._upload_speed = upload_speed

    @property
    def download_speed(self):
        return self._download_speed

    @download_speed.setter
    def download_speed(self, download_speed):
        self._download_speed = download_speed

    @ops.setter
    def ops(self, ops):
        self._ops = ops

    @property
    def client(self):
        if self._client is not None:
            return self._client
        if self._client is None:
            self._client = get_client()
            return self._client

    async def connect(self):
        try:
            g.logger.debug("Ravenverse url: {}?type={}".format(RAVENVERSE_URL, TYPE))
            g.logger.debug("Ravenverse FTP host: {}".format(RAVENVERSE_FTP_URL))
            auth_headers = {"token": self._ravenverse_token}
            await self._client.connect(url="{}?type={}".format(RAVENVERSE_URL, TYPE),
                        auth=auth_headers,
                        namespaces=['/client','/comm'],
                        transports=['websocket'], wait_timeout=100)
            g.logger.debug("Successfully connected to Ravenverse")
        except Exception as e:
            g.logger.error("Error: Unable to connect to Ravenverse. "
                        "Make sure you are using the right hostname and port. \n{}".format(e))
            await self._client.disconnect()
            os._exit(1)

    @property
    def ftp_client(self):
        return self._ftp_client

    @ftp_client.setter
    def ftp_client(self, ftp_client):
        self._ftp_client = ftp_client

    @property
    def delete_files_list(self):
        return self._delete_files_list

    @delete_files_list.setter
    def delete_files_list(self, delete_files_list):
        self._delete_files_list = delete_files_list

    @property
    def has_subgraph(self):
        return self._has_subgraph

    @has_subgraph.setter
    def has_subgraph(self, has_subgraph):
        self._has_subgraph = has_subgraph

    @property
    def is_downloading(self):
        return self._is_downloading

    @is_downloading.setter
    def is_downloading(self, is_downloading):
        self._is_downloading = is_downloading

    @property
    def is_uploading(self):
        return self._is_uploading

    @is_uploading.setter
    def is_uploading(self, is_uploading):
        self._is_uploading = is_uploading

    @property
    def noop_counter(self):
        return self._noop_counter

    @noop_counter.setter
    def noop_counter(self, noop_counter):
        self._noop_counter = noop_counter

    @property
    def ping_timeout_counter(self):
        return self._ping_timeout_counter

    @ping_timeout_counter.setter
    def ping_timeout_counter(self, ping_timeout_counter):
        self._ping_timeout_counter = ping_timeout_counter

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, error):
        self._error = error

    @property
    def ravenverse_token(self):
        return self._ravenverse_token

    @ravenverse_token.setter
    def ravenverse_token(self, ravenverse_token):
        self._ravenverse_token = ravenverse_token

    @property
    def logger(self):
        return self._logger

    @property
    def dashboard_data(self):
        return self._dashboard_data

    @dashboard_data.setter
    def dashboard_data(self, dashboard_data):
        self._dashboard_data = dashboard_data


g = Globals.Instance()
