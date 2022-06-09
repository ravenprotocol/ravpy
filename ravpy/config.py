import os
from pathlib import Path

BASE_DIR = os.path.join(str(Path.home()), "ravenverse/ravpy")

CONTEXT_FOLDER = os.path.join(BASE_DIR, "contexts")
PARAMS_DIR = os.path.join(BASE_DIR, "params")

os.makedirs(CONTEXT_FOLDER, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)

RAVENVERSE_HOST = os.environ.get("RAVENVERSE_HOST")
RAVENVERSE_PORT = os.environ.get("RAVENVERSE_PORT")
RAVENVERSE_URL = "http://{}:{}".format(RAVENVERSE_HOST, RAVENVERSE_PORT)
RAVENVERSE_FTP_HOST = os.environ.get("RAVENVERSE_FTP_HOST")

BENCHMARK_FILE_NAME = "ravpy/distributed/benchmark.json"
TYPE = "client"

ENCRYPTION = False

FTP_TEMP_FILES_FOLDER = os.path.join(os.getcwd(), "ravpy/distributed/temp_files")
FTP_DOWNLOAD_FILES_FOLDER = os.path.join(os.getcwd(), "ravpy/distributed/downloads")
