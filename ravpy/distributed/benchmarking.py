import json
import time

from ravop import functions

from ..config import SOCKET_SERVER_URL, BENCHMARK_FILE_NAME
from ..utils import download_file, get_key, setTimeout
from ..globals import g
from .compute import compute_locally_bm
from .evaluate import waitInterval

def benchmark():
    client = g.client
    initialTimeout = g.initialTimeout

    # Receive benchmarking ops
    download_file(url="{}/ravenjs/get/benchmark/".format(SOCKET_SERVER_URL), file_name=BENCHMARK_FILE_NAME)

    # Load benchmark file and execute ops
    with open(BENCHMARK_FILE_NAME, "rb") as f:
        benchmark_ops = json.load(f)
        benchmark_results = {}

        for benchmark_op in benchmark_ops:
            print("BM OP inside enumerate: ",benchmark_op)
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