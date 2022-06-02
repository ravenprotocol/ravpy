import ast
import json
import time

import speedtest
from ..strings import functions

from .compute import compute_locally_bm
from .evaluate import waitInterval
from ..config import RAVENVERSE_URL, BENCHMARK_FILE_NAME, RAVENVERSE_FTP_HOST
from ..ftp import check_credentials
from ..ftp import get_client as get_ftp_client
from ..globals import g
from ..utils import download_file, get_key, setTimeout, get_ftp_credentials


def initialize():
    credentials = get_ftp_credentials()

    if credentials is None:
        print("Unable to fetch credentials")
        return

    creds = ast.literal_eval(credentials['ftp_credentials'])
    print("Ftp credentials: ", creds)
    time.sleep(2)

    if RAVENVERSE_FTP_HOST != 'localhost' and RAVENVERSE_FTP_HOST != '0.0.0.0':
        wifi = speedtest.Speedtest()
        upload_speed = int(wifi.upload())
        download_speed = int(wifi.download())
        upload_speed = upload_speed / 8
        download_speed = download_speed / 8
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
        g.ftp_upload_blocksize = 8192 * 1000
        g.ftp_download_blocksize = 8192 * 1000

    print("FTP Upload Blocksize: ", g.ftp_upload_blocksize, "  ----   FTP Download Blocksize: ",
          g.ftp_download_blocksize)
    g.ftp_client = get_ftp_client(creds['username'], creds['password'])
    print("Check creds:", check_credentials(creds['username'], creds['password']))
    g.ftp_client.list_server_files()


def benchmark():
    print("benchmarking")
    initialize()
    client = g.client
    initialTimeout = g.initialTimeout

    # Receive benchmarking ops
    download_file(url="{}/ravenjs/get/benchmark/".format(RAVENVERSE_URL), file_name=BENCHMARK_FILE_NAME)

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
