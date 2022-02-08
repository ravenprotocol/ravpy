import json
import time

from ravop import compute_locally, functions

from ..config import SOCKET_SERVER_URL, BENCHMARK_FILE_NAME
from ..utils import download_file, get_key


def benchmark():
    # Receive benchmarking ops
    download_file(url="{}/ravenjs/get/benchmark/".format(SOCKET_SERVER_URL), file_name=BENCHMARK_FILE_NAME)

    # Load benchmark file and execute ops
    with open(BENCHMARK_FILE_NAME, "rb") as f:
        benchmark_ops = json.load(f)

        benchmark_results = {}

        for index, benchmark_op in enumerate(benchmark_ops):
            # print(benchmark_op)
            operator = get_key(benchmark_op['operator'], functions)
            t1 = time.time()
            print(compute_locally(*benchmark_op['values'], op_type=benchmark_op['op_type'], operator=operator))
            t2 = time.time()
            benchmark_results[benchmark_op["operator"]] = t2 - t1

            if index == 1:
                break

    print(benchmark_results)

    return benchmark_results

    # client.emit("benchmark_callback", data=json.dumps(benchmark_results), namespace="/client")
