import os
from pathlib import Path

BASE_DIR = os.path.join(str(Path.home()), "rdf/ravpy")

FTP_SERVER_URL = "localhost"

CONTEXT_FOLDER = os.path.join(BASE_DIR, "contexts")
PARAMS_DIR = os.path.join(BASE_DIR, "params")

os.makedirs(CONTEXT_FOLDER, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)

SOCKET_SERVER_HOST = "localhost"
SOCKET_SERVER_PORT = "9999"

SOCKET_SERVER_URL = "http://{}:{}".format(SOCKET_SERVER_HOST, SOCKET_SERVER_PORT)

BENCHMARK_FILE_NAME = "distributed/benchmark.json"
CID = "111"
TYPE = "client"

ENCRYPTION = False