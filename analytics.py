import ast
import asyncio
import functools
import logging
import os
import pickle
import signal
import sys
import numpy as np
import socketio
import tenseal as ts

from config import CONTEXT_FOLDER, PARAMS_DIR
from ftp_client import get_client
from helpers import load_context

logger = logging.getLogger(__name__)

sio = socketio.AsyncClient(logger=True, engineio_logger=True)
client = None
data_silo = [1, 2, 3]
cid = None
# avg = sum(data_silo) / len(data_silo)
#
# min_val = min(data_silo)
# max_val = max(data_silo)
#
# variance = sum((i - avg) ** 2 for i in data_silo) / len(data_silo)
#
objective = None


# params = {'size': [len(data_silo)],
#           'Average': [avg],
#           'Minimum': [min_val],
#           'Maximum': [max_val],
#           'Variance': [variance]
#           }
#
# print(data_silo)
# print(params)


@sio.event
async def connect():
    print('Connected to server')


@sio.event
async def disconnect():
    print('Disconnected from server')


def check_compatibility(data, obj):
    rules = ast.literal_eval(obj["rules"])
    props = analyze_data(data)

    if rules['lower-limit'] < props['min'] and rules['upper-limit'] > props['max']:
        return True
    else:
        return False


@sio.on("receive_objective", namespace='/analytics')
async def receive_objective(obj=None):
    global objective
    global ckks_context

    print("receive_objective")

    if obj is not None:
        objective = obj
        print(objective)
        # Check compatibility here
        if check_compatibility(data=data_silo, obj=obj):

            print("", objective['ftp_credentials'])
            ftp_client = get_client(**ast.literal_eval(objective['ftp_credentials']))

            print(ftp_client, os.path.join(CONTEXT_FOLDER, objective['context_filename']), objective['context_filename'])

            ftp_client.download(os.path.join(CONTEXT_FOLDER, "f_"+objective['context_filename']),
                                objective['context_filename'])

            await sio.sleep(10)
            print(os.path.join(CONTEXT_FOLDER, objective['context_filename']))
            ckks_context = load_context(os.path.join(CONTEXT_FOLDER, objective['context_filename']))

            if ckks_context is not None:
                await cal_params(ftp_client)
        else:
            print("Your data is not compatible")
    else:
        print("Obj is none")


def analyze_data(data):
    rank = len(np.array(data).shape)

    if rank == 0:
        return {"rank": rank, "dtype": np.array(data).dtype.__class__.__name__}
    elif rank == 1:
        return {"rank": rank, "max": max(data), "min": min(data), "dtype": np.array(data).dtype.__class__.__name__}
    else:
        return {"rank": rank, "dtype": np.array(data).dtype.__class__.__name__}


async def wait_for_objective():
    while objective is None:
        if sio.connected:
            await sio.emit('handshake', {}, namespace='/analytics')
            await sio.sleep(10)


async def cal_params(ftp_client):
    global objective
    params = dict()
    encryption = objective.get("encryption", True)
    rank = len(np.array(data_silo).shape)

    mean = None
    variance = None
    standard_deviation = None
    size = len(data_silo)
    maximum = max(data_silo)
    minimum = min(data_silo)

    if rank == 0:
        # TODO: Apply differential privacy
        mean = data_silo
    elif rank == 1:
        if objective["operator"] == "mean":
            mean = sum(data_silo) / len(data_silo)
        elif objective['operator'] == "variance":
            mean = sum(data_silo) / len(data_silo)
            variance = sum((i - mean) ** 2 for i in data_silo) / len(data_silo)
        elif objective['operator'] == "standard_deviation":
            mean = sum(data_silo) / len(data_silo)
            variance = sum((i - mean) ** 2 for i in data_silo) / len(data_silo)
            standard_deviation = np.sqrt(variance)

    if encryption:
        print('Encrypting params...')
        mean = ts.ckks_tensor(ckks_context, [mean]).serialize()
        variance = ts.ckks_tensor(ckks_context, [variance]).serialize()
        standard_deviation = ts.ckks_tensor(ckks_context, [standard_deviation]).serialize()
        size = ts.ckks_tensor(ckks_context, [size]).serialize()
        minimum = ts.ckks_tensor(ckks_context, [minimum]).serialize()
        maximum = ts.ckks_tensor(ckks_context, [maximum]).serialize()

    params["mean"] = mean
    params["variance"] = variance
    params["standard_deviation"] = standard_deviation
    params["size"] = size
    params["minimum"] = minimum
    params["maximum"] = maximum
    params["objective_id"] = objective["id"]
    params["encryption"] = encryption

    params_filename = "params_{}.pkl".format(objective["id"])
    params_file = os.path.join(PARAMS_DIR, params_filename)

    with open(params_file, "wb") as f:
        pickle.dump(params, f)

    print("Uploading")
    ftp_client.upload(params_file, params_filename)
    print("Uploaded")

    print('Emitting..')
    for i in range(2):
        if not sio.connected:
            print("Not connected")
        await sio.emit('receive_params', {"status": "success", "params_file": params_filename}, namespace='/analytics')
        await sio.sleep(10)
    print("Emitted")
    objective = None

    await wait_for_objective()


def shutdown(loop):
    print('exit')
    # await sio.emit("disconnect", {}, namespace="/analytics")
    # await sio.disconnect()
    # await sio.sleep(1)
    # for talk in asyncio.Task.all_tasks():
    #     talk.cancel()
    # sio.disconnect()
    # sio.sleep(1)
    # loop.close()

    # tasks = [task for task in asyncio.all_tasks() if task is not
    #          asyncio.current_task()]
    # list(map(lambda task: task.cancel(), tasks))
    # results = asyncio.gather(*tasks, return_exceptions=True)
    # print('finished awaiting cancelled tasks, results: {0}'.format(results))
    # loop.stop()

    sys.exit(1)


# signal.signal(signal.SIGINT, sigint_handler)


async def start_client():
    await sio.connect('http://localhost:9999?type=analytics&cid={}'.format(8888), namespaces=["/analytics"],
                      transports=['websocket'])
    await sio.start_background_task(wait_for_objective)

    await sio.wait()


@sio.on("check", namespace="/analytics")
async def check(data):
    print("check")
    return data


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_client())
    loop.add_signal_handler(signal.SIGINT, functools.partial(asyncio.ensure_future, shutdown(loop)))
    loop.close()

