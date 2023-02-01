import os
import socket
from ftplib import FTP

from ..config import RAVENVERSE_FTP_URL, FTP_TEMP_FILES_FOLDER
from ..globals import g

try:
    import ssl
except ImportError:
    _SSLSocket = None
else:
    _SSLSocket = ssl.SSLSocket

class FTPClient:
    def __init__(self, host, user, passwd):
        self.ftp = FTP(host)
        # self.ftp.set_debuglevel(2)
        self.ftp.login(user, passwd)
        self.ftp.set_pasv(True)

    def download(self, filename, path):
        try:
            sock = socket.create_connection(('8.8.8.8',53))
            sock.close()
        except Exception as e:
            print('\n ----------- Device offline -----------')
            os._exit(1)
        g.is_downloading = True
                
        try:
            with open(filename, 'wb') as f:
                self.ftp.retrbinary('RETR ' + path, f.write, blocksize=g.ftp_download_blocksize)
        except Exception as e:
            exit_handler()
            os._exit(1)

        g.is_downloading = False


    def upload(self, filename, path):
        try:
            sock = socket.create_connection(('8.8.8.8',53))
            sock.close()
        except Exception as e:
            print('\n ----------- Device offline -----------')
            os._exit(1)
        g.is_uploading = True

        try:
            with open(filename, 'rb') as f:
                # self.ftp.storbinary('STOR ' + path, f, blocksize=g.ftp_upload_blocksize)
                self.storbinary('STOR ' + path, f, blocksize=g.ftp_upload_blocksize)
        except Exception as e:
            exit_handler()
            os._exit(1)

        g.is_uploading = False

    def list_server_files(self):
        self.ftp.retrlines('LIST')

    def delete_file(self, path):
        self.ftp.delete(path)

    def close(self):
        self.ftp.quit()

    def storbinary(self, cmd, fp, blocksize=8192, callback=None, rest=None):
        """Store a file in binary mode.  A new port is created for you.

        Args:
          cmd: A STOR command.
          fp: A file-like object with a read(num_bytes) method.
          blocksize: The maximum data size to read from fp and send over
                     the connection at once.  [default: 8192]
          callback: An optional single parameter callable that is called on
                    each block of data after it is sent.  [default: None]
          rest: Passed to transfercmd().  [default: None]

        Returns:
          The response code.
        """
        self.ftp.voidcmd('TYPE I')
        with self.ftp.transfercmd(cmd, rest) as conn:
            while 1:
                buf = fp.read(blocksize)
                if not buf:
                    break
                conn.sendall(buf)
                if callback:
                    callback(buf)
            # shutdown ssl layer
            if _SSLSocket is not None and isinstance(conn, _SSLSocket):
                # conn.unwrap()
                pass
        return self.ftp.voidresp()


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

def exit_handler():
    g.logger.debug('Application is Closing!')
    if g.client is not None:
        g.logger.debug("Disconnecting...")
        if g.client.connected:
            g.client.emit("disconnect", namespace="/client")

    dir = FTP_TEMP_FILES_FOLDER
    if os.path.exists(dir):
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))