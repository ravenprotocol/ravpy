from ftplib import FTP

class FTPClient:
    def __init__(self, host, user, passwd):
        self.ftp = FTP(host)
        self.ftp.login(user, passwd)

    def download(self, filename, path):
        self.ftp.retrbinary('RETR ' + path, open(filename, 'wb').write)

    def upload(self, filename, path):
        self.ftp.storbinary('STOR ' + path, open(filename, 'rb'))

    def list_server_files(self):
        self.ftp.retrlines('LIST')

    def close(self):
        self.ftp.quit()

# ftp_client = FTPClient('0.0.0.0', 'user', 'password')
# ftp_client.list_server_files()
# ftp_client.upload('model/local.h5','local2.h5')
# # ftp_client.download('model/global.h5','global.h5')
# ftp_client.close()