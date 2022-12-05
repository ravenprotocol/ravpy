import eel
import logging
import os
import psutil
import shutil
from hurry.filesize import size

os.environ['RAVENVERSE_URL'] = "http://server.ravenverse.ai"
os.environ['RAVENVERSE_FTP_HOST'] = "server.ravenverse.ai"
os.environ['RAVENVERSE_FTP_URL'] = "server.ravenverse.ai"
os.environ['RAVENAUTH_URL'] = "https://auth.ravenverse.ai"

eel.init('web')


@eel.expose
def disconnect():
    if g.client.connected:
        g.logger.debug("Disconnecting...")
        if g.client.connected:
            g.client.emit("disconnect", namespace="/client")
            g.logger.debug("Disconnected")
            g.logger.debug("")

    return True


def close_callback(a, b):
    disconnect()


from ravpy.globals import g


class CustomHandler(logging.Handler):
    def __init__(self):
        logging.Handler.__init__(self)

    def emit(self, record):
        print("Custom", record)
        eel.getLog({"asctime": record.asctime, "threadName": record.threadName, "levelname": record.levelname,
                    "message": record.message})
        return record


log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
my_handler = CustomHandler()
my_handler.setLevel(logging.DEBUG)
my_handler.setFormatter(log_formatter)
g.logger.addHandler(my_handler)


@eel.expose
def verify_access_token(access_token):
    from ravpy.utils import verify_token
    if verify_token(access_token):
        return [access_token, "success", ""]
    else:
        return [access_token, "failure", "Invalid access token!"]


@eel.expose
def get_system_config():
    ram_total = str(size(psutil.virtual_memory().total))
    ram_available = str(size(psutil.virtual_memory().available))
    cpu_count = psutil.cpu_count(logical=False)
    cpu_percent = psutil.cpu_percent()
    total, used, free = shutil.disk_usage("/")
    storage_total = total // (2 ** 30)
    storage_available = free // (2 ** 30)

    return {"ram_total": ram_total, "ram_available": ram_available,
            "cpu_count": cpu_count, "cpu_percent": cpu_percent, "storage_total": storage_total,
            "storage_available": storage_available}


@eel.expose
def get_logs(skip, limit):
    with open("debug.log", "r") as f:
        logs = f.readlines()[skip:]
        print(logs)


@eel.expose
def participate(token):
    from ravpy.distributed.benchmarking import benchmark
    from ravpy.utils import initialize_ftp_client
    from ravpy.initialize import initialize

    # Initialize
    socket_client = initialize(ravenverse_token=token)

    if socket_client is None:
        disconnect()
        eel.clientDisconnected()
        return False
    else:
        eel.clientConnected()

    # get ftp client
    ftp_client = initialize_ftp_client()

    if ftp_client is None:
        disconnect()
        eel.clientDisconnected()
        return False

    # do benchmark
    benchmark()

    g.logger.debug("")
    g.logger.debug("Ravpy is waiting for graphs/subgraphs/ops...")
    g.logger.debug("Warning: Do not close Ravpy if you like to "
                   "keep participating and keep earning Raven tokens\n")

    return True


@eel.expose
def get_subgraphs():
    subgraphs = g.ravdb.get_subgraphs()
    subgraphs = [sg.as_dict() for sg in subgraphs]
    return subgraphs


@eel.expose
def delete_subgraphs():
    g.ravdb.delete_subgraphs()
    return True


g.ravdb.create_database()
g.ravdb.create_tables()


eel.start('main.html', close_callback=close_callback)
