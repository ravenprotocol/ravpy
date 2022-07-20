import os

os.environ['RAVENVERSE_URL'] = "http://0.0.0.0:8081"
os.environ['RAVENVERSE_FTP_URL'] = "0.0.0.0"

from ravpy.distributed.participate import participate
from ravpy.initialize import initialize
from ravpy.globals import g
import atexit

client = None


def exit_handler():
    g.logger.debug('My application is ending!')
    if client is not None:
        g.logger.debug("disconnecting")
        if client.connected:
            client.emit("disconnect", namespace="/client")
    else:
        g.logger.debug("client is none")


atexit.register(exit_handler)

if __name__ == '__main__':
    client = initialize(
        "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNjU4NDI1NjQzLCJpYXQiOjE2NTgyMDk2NDMsImp0aSI6IjVmYmRjZGMyZTU1ZjRiMmFiNzIzYTBiNzM0NDkxOGQ1IiwidXNlcl9pZCI6IjAwMjY3MDgwNTgifQ.N4V_z4lRbewD8XjjazyTa6Z9BEzF2i5dMpsPO-yT88M")

    participate()
