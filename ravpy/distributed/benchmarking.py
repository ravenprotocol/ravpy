import json
import os
import time

from .compute import compute_locally_bm
from .evaluate import waitInterval
from ..config import RAVENVERSE_URL, BENCHMARK_FILE_NAME
from ..globals import g
from ..strings import functions
from ..utils import download_file, get_key, setTimeout


def benchmark(graph_id=None):
    g.logger.debug("")
    g.logger.debug("Starting benchmarking...")

    client = g.client
    initialTimeout = g.initialTimeout

    # Receive benchmarking ops
    download_file(url="{}/ravenjs/get/benchmark/".format(RAVENVERSE_URL), file_name=BENCHMARK_FILE_NAME)

    # Load benchmark file and execute ops
    with open(BENCHMARK_FILE_NAME, "rb") as f:
        benchmark_ops = json.load(f)
        benchmark_results = {}

        for benchmark_op in benchmark_ops:
            operator = get_key(benchmark_op['operator'], functions)
            t1 = time.time()
            res = compute_locally_bm(*benchmark_op['values'], op_type=benchmark_op['op_type'], operator=operator)
            t2 = time.time()
            benchmark_results[benchmark_op["operator"]] = t2 - t1

    for file in os.listdir():
        if file.endswith(".zip"):
            os.remove(file)

    if graph_id is None:
        graph_id = "None"

    benchmark_data = {
        'benchmark_data':benchmark_results,
        'graph_id':graph_id
    }

    client.emit("benchmark_callback", data=json.dumps(benchmark_data), namespace="/client")
    client.sleep(1)
    g.logger.debug("Benchmarking Complete!")

    setTimeout(waitInterval, initialTimeout)
