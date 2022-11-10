import logging

import eel
import psutil
import shutil
from dotenv import load_dotenv
from hurry.filesize import size

load_dotenv()

from ravpy.utils import verify_token
from ravpy.distributed.participate import participate
from ravpy.initialize import initialize
from ravpy.globals import g

eel.init('web')


class CustomHandler(logging.Handler):
    def __init__(self):
        logging.Handler.__init__(self)

    def emit(self, record):
        print("Custom", record)
        eel.getLog("{} [{}] {}".format(record.asctime, record.levelname, record.message))


log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
my_handler = CustomHandler()
my_handler.setLevel(logging.DEBUG)
my_handler.setFormatter(log_formatter)
g.logger.addHandler(my_handler)


@eel.expose
def verify_access_token(access_token):
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
def initialize_exposed(token):
    client = initialize(token)
    if client is None:
        return False
    else:
        return True


@eel.expose
def participate_exposed():
    participate()


@eel.expose
def get_logs(skip, limit):
    with open("debug.log", "r") as f:
        logs = f.readlines()[skip:]
        print(logs)


@eel.expose
def disconnect():
    if g.client.connected:
        g.logger.debug("Disconnecting...")
        if g.client.connected:
            g.client.emit("disconnect", namespace="/client")
            g.logger.debug("Disconnected")

    return True


def close_callback(a, b):
    print("closing", a, b)
    disconnect()


eel.start('main.html', close_callback=close_callback)
