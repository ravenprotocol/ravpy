import atexit
import os

from .globals import g
from .utils import isLatestVersion, initialize_ftp_client
from .config import FTP_TEMP_FILES_FOLDER, FTP_DOWNLOAD_FILES_FOLDER


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

async def initialize(ravenverse_token, graph_id=None):
    dir = FTP_TEMP_FILES_FOLDER
    if os.path.exists(dir):
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))
    dir = FTP_DOWNLOAD_FILES_FOLDER
    if os.path.exists(dir):
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))

    dir = os.getcwd()
    if os.path.exists(dir):
        for f in os.listdir(dir):
            if f.endswith(".zip") and "local_" in f:
                os.remove(os.path.join(dir, f))

    g.logger.debug("Checking Version of Ravpy...")

    if not isLatestVersion('ravpy'):
        g.logger.debug("Please update ravpy to latest version...")
        os._exit(1)

    g.logger.debug("Initializing...")
    g.ravenverse_token = ravenverse_token

    client = g.client
    await g.connect()
    if client is None:
        await g.client.disconnect()
        g.logger.error("Unable to connect to ravsock. Make sure you are using the right hostname and port")
        os._exit(1)
    else:
        g.logger.debug("Initialized successfully\n")

    # Initialize and create FTP client
    res = initialize_ftp_client()
    if res is None:
        os._exit(1)

    from .distributed.benchmarking import benchmark_model
    await benchmark_model(seed=123, graph_id=graph_id)

    g.logger.debug("")
    g.logger.debug("Ravpy is waiting for ops and subgraphs...")
    g.logger.debug("Warning: Do not close this terminal if you like to "
                   "keep participating and keep earning Raven tokens\n")
    await g.client.wait()