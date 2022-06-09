import os


def participate():
    # Connect
    download_path = "./ravpy/distributed/downloads/"
    temp_files_path = "./ravpy/distributed/temp_files/"
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    if not os.path.exists(temp_files_path):
        os.makedirs(temp_files_path)

    from .benchmarking import benchmark
    bm_results = benchmark()
    print("Benchmark Results: ", bm_results)
