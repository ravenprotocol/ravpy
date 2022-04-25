import json
import time
import ast

from ravop import functions

from ..config import SOCKET_SERVER_URL, BENCHMARK_FILE_NAME
from ..utils import download_file, get_key, setTimeout, get_ftp_credentials
from ..globals import g
from .compute import compute_locally_bm
from .evaluate import waitInterval
from ..ftp import check_credentials
from ..ftp import get_client as get_ftp_client


def initialize():
    time.sleep(5)
    creds = ast.literal_eval(get_ftp_credentials(g.cid)['ftp_credentials'])
    print("Ftp credentials: ", creds)
    time.sleep(2)

    g.ftp_client = get_ftp_client(creds['username'], creds['password'])
    print("Check creds:", check_credentials(creds['username'], creds['password']))
    g.ftp_client.list_server_files()


def benchmark():
    initialize()
    client = g.client
    initialTimeout = g.initialTimeout

    # Receive benchmarking ops
    download_file(url="{}/ravenjs/get/benchmark/".format(SOCKET_SERVER_URL), file_name=BENCHMARK_FILE_NAME)

    # Load benchmark file and execute ops
    with open(BENCHMARK_FILE_NAME, "rb") as f:
        benchmark_ops = json.load(f)
        benchmark_results = {}

        for benchmark_op in benchmark_ops:
            # print("BM OP inside enumerate: ",benchmark_op)
            operator = get_key(benchmark_op['operator'], functions)
            t1 = time.time()
            print(compute_locally_bm(*benchmark_op['values'], op_type=benchmark_op['op_type'], operator=operator))
            t2 = time.time()
            benchmark_results[benchmark_op["operator"]] = t2 - t1

    print("\nEmitting Benchmark Results...")
    client.emit("benchmark_callback", data=json.dumps(benchmark_results), namespace="/client")
    client.sleep(1)
    setTimeout(waitInterval, initialTimeout)
    return benchmark_results