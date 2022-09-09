import os

from ftplib import FTP

from ..config import RAVENVERSE_FTP_URL
from ..globals import g


class FTPClient:
    def __init__(self, host, user, passwd):
        self.ftp = FTP(host)
        # self.ftp.set_debuglevel(2)
        self.ftp.login(user, passwd)
        self.ftp.set_pasv(True)

    def download(self, filename, path):
        with open(filename, 'wb') as f:
            self.ftp.retrbinary('RETR ' + path, f.write, blocksize=g.ftp_download_blocksize)

    def upload(self, filename, path):
        with open(filename, 'rb') as f:
            self.ftp.storbinary('STOR ' + path, f, blocksize=g.ftp_upload_blocksize)

    def list_server_files(self):
        self.ftp.retrlines('LIST')

    def delete_file(self, path):
        self.ftp.delete(path)

    def close(self):
        self.ftp.quit()


def get_client(username, password):
    """
    Create FTP client
    :param username: FTP username
    :param password: FTP password
    :return: FTP client
    """
    try:
        g.logger.debug("Creating FTP client...")
        client = FTPClient(host=RAVENVERSE_FTP_URL, user=username, passwd=password)
        g.logger.debug("FTP client created successfully")
        return client
    except Exception as e:
        g.logger.debug("Unable to create FTP client")
        os._exit(1)


def check_credentials(username, password):
    try:
        FTPClient(host=RAVENVERSE_FTP_URL, user=username, passwd=password)
        return True
    except Exception as e:
        print("Error:{}".format(str(e)))
        return False
