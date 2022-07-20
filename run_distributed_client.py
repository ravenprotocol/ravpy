import os

os.environ['RAVENVERSE_URL'] = "http://0.0.0.0:8081"
os.environ['RAVENVERSE_FTP_URL'] = "0.0.0.0"

from ravpy.distributed.participate import participate
from ravpy.initialize import initialize


if __name__ == '__main__':
    client = initialize(
        "<token>")
    participate()
