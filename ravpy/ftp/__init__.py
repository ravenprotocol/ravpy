from ftplib import FTP

from ..config import RAVENVERSE_FTP_HOST
from ..globals import g


class FTPClient:
    def __init__(self, host, user, passwd):
        self.ftp = FTP(host)
        # self.ftp.set_debuglevel(2)
        self.ftp.login(user, passwd)
        self.ftp.set_pasv(True)

    def download(self, filename, path):
        print('Downloading')
        with open(filename, 'wb') as f:
            self.ftp.retrbinary('RETR ' + path, f.write, blocksize=g.ftp_download_blocksize)
        print("Downloaded")

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
    print("FTP User credentials:", RAVENVERSE_FTP_HOST, username, password)
    return FTPClient(host=RAVENVERSE_FTP_HOST, user=username, passwd=password)


def check_credentials(username, password):
    try:
        FTPClient(host=RAVENVERSE_FTP_HOST, user=username, passwd=password)
        return True
    except Exception as e:
        print("Error:{}".format(str(e)))
        return False
