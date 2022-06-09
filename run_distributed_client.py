import os

os.environ['RAVENVERSE_HOST'] = "0.0.0.0"
os.environ['RAVENVERSE_PORT'] = "9999"
os.environ['RAVENVERSE_FTP_HOST'] = "0.0.0.0"

from ravpy.distributed.participate import participate
from ravpy.initialize import initialize

if __name__ == '__main__':
    initialize("<token>")
    participate()
