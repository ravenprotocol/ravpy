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
        self._opTimeout = 50
        self._initialTimeout = 100
        self._outputs = {}
        self._ftp_client = None
        self._delete_files_list = []
        self._has_subgraph = False
        self._ftp_upload_blocksize = 8192
        self._ftp_download_blocksize = 8192
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


g = Globals.Instance()
