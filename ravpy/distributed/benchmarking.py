import json
import time
import random
import numpy as np
import torch
from torch import nn
from collections import OrderedDict

from .evaluate import waitInterval
from ..globals import g
from ..utils import setTimeout, get_total_RAM, check_gpu

async def benchmark_model(seed, graph_id=None):
    g.logger.debug("")
    g.logger.debug("Starting Model benchmarking...")

    initialTimeout = g.initialTimeout

    random.seed(seed)
    np.random.seed(seed)
    device = torch.device('cpu')
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
        'client_sid': g.client.get_sid(namespace='/client')
    }

    g.logger.debug("Emitting Benchmarking Data: {}".format(benchmark_data))
    await g.client.emit("benchmark_callback", data=json.dumps(benchmark_data), namespace="/comm")
    await g.client.sleep(1)
    g.logger.debug("Benchmarking Complete!")

    setTimeout(waitInterval, initialTimeout)
