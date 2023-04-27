import atexit
import os
import speedtest

from .globals import g
from .utils import isLatestVersion, initialize_ftp_client
from .config import FTP_TEMP_FILES_FOLDER, FTP_DOWNLOAD_FILES_FOLDER, RAVENVERSE_FTP_URL

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

def internet_speedtest():
    try:
        g.logger.debug("")
        g.logger.debug("Testing network speed...")
        if RAVENVERSE_FTP_URL != 'localhost' and RAVENVERSE_FTP_URL != '0.0.0.0':
            wifi = speedtest.Speedtest()
            upload_speed = int(wifi.upload())
            download_speed = int(wifi.download())
            upload_speed = upload_speed / 8
            download_speed = download_speed / 8
            g.upload_speed = upload_speed
            g.download_speed = download_speed

            if upload_speed <= 3000000:
                upload_multiplier = 1
            elif upload_speed < 80000000:
                upload_multiplier = int((upload_speed / 80000000) * 1000)
            else:
                upload_multiplier = 1000

            if download_speed <= 3000000:
                download_multiplier = 1
            elif download_speed < 80000000:
                download_multiplier = int((download_speed / 80000000) * 1000)
            else:
                download_multiplier = 1000

            g.ftp_upload_blocksize = 8192 * upload_multiplier
            g.ftp_download_blocksize = 8192 * download_multiplier

        else:
            g.upload_speed = 100000000
            g.download_speed = 100000000
            g.ftp_upload_blocksize = 8192 * 1000
            g.ftp_download_blocksize = 8192 * 1000
        g.logger.debug("Upload Speed: {} Mbps".format(g.upload_speed / 1000000))
        g.logger.debug("Download Speed: {} Mbps".format(g.download_speed / 1000000))

    except Exception as e:
        g.ftp_upload_blocksize = 8192 * 1000
        g.ftp_download_blocksize = 8192 * 1000
        

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

    internet_speedtest()

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