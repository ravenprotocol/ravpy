import json
import os
import time
import psutil
import random
import numpy as np
import torch
from torch import nn
from collections import OrderedDict

from .compute import compute_locally_bm
from .evaluate import waitInterval
from ..config import RAVENVERSE_URL, BENCHMARK_FILE_NAME
from ..globals import g
from ..strings import functions
from ..utils import download_file, get_key, setTimeout, get_total_RAM, check_gpu

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
        'benchmark_data': benchmark_results,
        'graph_id': graph_id,
        'upload_speed': g.upload_speed,
        'download_speed': g.download_speed,
        'total_RAM': get_total_RAM(),
        'gpu_available': check_gpu(),
    }

    client.emit("benchmark_callback", data=json.dumps(benchmark_data), namespace="/client")
    client.sleep(1)
    g.logger.debug("Benchmarking Complete!")

    setTimeout(waitInterval, initialTimeout)

def benchmark_model(seed, graph_id=None):
    g.logger.debug("")
    g.logger.debug("Starting Model benchmarking...")

    client = g.client
    initialTimeout = g.initialTimeout

    random.seed(seed)
    np.random.seed(seed)
    device = torch.device('mps')
    in_channels = random.randint(3,6)
    height = random.randint(8,32)
    width = random.randint(8,32)
    target_shape = random.randint(1,5)

    X = torch.tensor(np.random.randn(128,in_channels,height,width)).type(torch.float32).to(device=device)
    y = torch.tensor(np.random.randint(0,2,size=(128,target_shape))).type(torch.float32).to(device=device)

    out_channels_1st_layer = random.randint(6,12)
    l1 = [('conv_0', nn.Conv2d(in_channels,out_channels_1st_layer,(3,3), padding='same'))]
    l1.extend([('conv_{}'.format(i), nn.Conv2d(out_channels_1st_layer,out_channels_1st_layer,(2,2), padding='same')) for i in range(1,4)])
    l1.append(('relu1', nn.ReLU()))
    l1.append(('flatten', nn.Flatten()))
    l1.append(('fin_lin', nn.Linear(height*width*out_channels_1st_layer,target_shape)))
    l1.append(('relu2', nn.ReLU()))

    print(OrderedDict(l1))

    model = nn.Sequential(OrderedDict(l1))
    model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters())
    epochs = 100

    t1 = time.time()
    for i in range(epochs):
        op = model(X)

        loss = nn.functional.mse_loss(op, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    t = time.time() -  t1

    if graph_id is None:
        graph_id = "None"

    benchmark_result = {'stake':1/t}
    benchmark_data = {
        'benchmark_data': benchmark_result,
        'graph_id': graph_id,
        'upload_speed': g.upload_speed,
        'download_speed': g.download_speed,
        'total_RAM': get_total_RAM(),
        'gpu_available': check_gpu(),
    }

    client.emit("benchmark_callback", data=json.dumps(benchmark_data), namespace="/client")
    client.sleep(1)
    g.logger.debug("Benchmarking Complete!")

    setTimeout(waitInterval, initialTimeout)
