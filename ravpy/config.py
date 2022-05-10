import os
from pathlib import Path

BASE_DIR = os.path.join(str(Path.home()), "rdf/ravpy")

FTP_SERVER_URL = "0.0.0.0" # Change to IP Address of GCP Instance

CONTEXT_FOLDER = os.path.join(BASE_DIR, "contexts")
PARAMS_DIR = os.path.join(BASE_DIR, "params")

os.makedirs(CONTEXT_FOLDER, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)

SOCKET_SERVER_HOST = "0.0.0.0" # Change to IP Address of GCP Instance
SOCKET_SERVER_PORT = "9999"

SOCKET_SERVER_URL = "http://{}:{}".format(SOCKET_SERVER_HOST, SOCKET_SERVER_PORT)

BENCHMARK_FILE_NAME = "ravpy/distributed/benchmark.json"
TYPE = "client"

ENCRYPTION = False

FTP_TEMP_FILES_FOLDER = os.path.join(os.getcwd(),"ravpy/distributed/temp_files")

FTP_DOWNLOAD_FILES_FOLDER = os.path.join(os.getcwd(),"ravpy/distributed/downloads")