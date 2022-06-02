import socketio
import os
from .config import RAVENVERSE_HOST, RAVENVERSE_PORT, TYPE
from .singleton_utils import Singleton


def get_client(ravenverse_token):
    auth_headers = {"token": ravenverse_token}
    client = socketio.Client(logger=False, request_timeout=60)

    @client.on('error',namespace='/client')
    def check_error(d):
        print("\n======= Error: {} =======".format(d))
        client.disconnect()
        os._exit(1)

    try:
        client.connect(url="http://{}:{}?type={}".format(RAVENVERSE_HOST, RAVENVERSE_PORT, TYPE),
                       auth=auth_headers,
                       namespaces=['/client'], wait_timeout=10)
        return client
    except Exception as e:
        print("Exception:{}, Unable to connect to ravsock. Make sure you are using the right hostname and port".format(e))
        exit()


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


g = Globals.Instance()
