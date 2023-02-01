import os
import socketio
from .config import RAVENVERSE_URL, TYPE, RAVENVERSE_FTP_URL
from .logger import get_logger
from .singleton_utils import Singleton


def get_client(ravenverse_token):
    """
    Connect to Ravebverse and return socket client instance
    :param ravenverse_token: authentication token
    :return: socket client
    """
    g.logger.debug("Connecting to Ravenverse...")

    auth_headers = {"token": ravenverse_token}
    client = socketio.Client(logger=False, request_timeout=100, engineio_logger=False)

    @client.on('error', namespace='/client')
    def check_error(d):
        g.logger.error("Connection error:a{}".format(d))
        client.disconnect()
        os._exit(1)

    try:
        g.logger.debug("Ravenverse url: {}?type={}".format(RAVENVERSE_URL, TYPE))
        g.logger.debug("Ravenverse FTP host: {}".format(RAVENVERSE_FTP_URL))
        client.connect(url="{}?type={}".format(RAVENVERSE_URL, TYPE),
                       auth=auth_headers,
                       transports=['websocket'],
                       namespaces=['/client'], wait_timeout=100)
        g.logger.debug("Successfully connected to Ravenverse")
        return client
    except Exception as e:
        g.logger.error("Error: Unable to connect to Ravenverse. "
                       "Make sure you are using the right hostname and port. \n{}".format(e))
        client.disconnect()
        os._exit(1)


@Singleton
class Globals(object):
    def __init__(self):
        self._client = None
        self._timeoutId = None
        self._ops = {}
        self._opTimeout = 300#5000
        self._initialTimeout = 100
        self._outputs = {}
        self._ftp_client = None
        self._delete_files_list = []
        self._has_subgraph = False
        self._is_downloading = False
        self._is_uploading = False
        self._noop_counter = 0
        self._ftp_upload_blocksize = 8192
        self._ftp_download_blocksize = 8192
        self._ping_timeout_counter = 0
        self._error = False
        self._ravenverse_token = None
        self._logger = get_logger()
        self._dashboard_data = [['Subgraph ID', 'Graph ID', 'Status']]
        self._forward_computations = {}
        self._grad_tensors = {}

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

    @ops.setter
    def ops(self, ops):
        self._ops = ops

    @property
    def client(self):
        if self._client is not None:
            return self._client
        if self._client is None:
            self._client = get_client(self._ravenverse_token)
            return self._client

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

    @outputs.setter
    def outputs(self, outputs):
        self._outputs = outputs

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

    @property
    def forward_computations(self):
        return self._forward_computations

    @forward_computations.setter
    def forward_computations(self, forward_computations):
        self._forward_computations = forward_computations

    @property
    def grad_tensors(self):
        return self._grad_tensors

    @grad_tensors.setter
    def grad_tensors(self, grad_tensors):
        self._grad_tensors = grad_tensors


g = Globals.Instance()
