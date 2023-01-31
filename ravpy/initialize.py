import atexit
import os
import sys

from .globals import g

from .utils import isLatestVersion
from .config import FTP_TEMP_FILES_FOLDER


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


atexit.register(exit_handler)


def initialize(ravenverse_token):
    g.logger.debug("Checking Version of Ravpy...")

    if not isLatestVersion('ravpy'):
        g.logger.debug("Please update ravpy to latest version...")
        os._exit(1)

    g.logger.debug("Initializing...")
    g.ravenverse_token = ravenverse_token

    client = g.client
    if client is None:
        g.client.disconnect()
        g.logger.error("Unable to connect to ravsock. Make sure you are using the right hostname and port")
        os._exit(1)
    else:
        g.logger.debug("Initialized successfully\n")
        return client
