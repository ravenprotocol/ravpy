import os
import pickle as pkl
import shutil

import numpy as np
import requests

from .config import ENCRYPTION

if ENCRYPTION:
    import tenseal as ts

from .config import BASE_DIR, CONTEXT_FOLDER, RAVENVERSE_URL, FTP_TEMP_FILES_FOLDER

from threading import Timer

from .globals import g


def download_file(url, file_name):
    headers = {"token": g.ravenverse_token}
    with requests.get(url, stream=True, headers=headers) as r:
        with open(file_name, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    print("file downloaded")


def get_key(val, dict):
    for key, value in dict.items():
        if val == value:
            return key
    return "key doesn't exist"


def analyze_data(data):
    rank = len(np.array(data).shape)

    if rank == 0:
        return {"rank": rank, "dtype": np.array(data).dtype.__class__.__name__}
    elif rank == 1:
        return {"rank": rank, "max": max(data), "min": min(data), "dtype": np.array(data).dtype.__class__.__name__}
    else:
        return {"rank": rank, "dtype": np.array(data).dtype.__class__.__name__}


def dump_context(context, cid):
    filename = "context_{}.txt".format(cid)
    fpath = os.path.join(BASE_DIR, filename)
    with open(fpath, "wb") as f:
        f.write(context.serialize())

    return filename, fpath


def load_context(file_path):
    with open(file_path, "rb") as f:
        return ts.context_from(f.read())


def fetch_and_load_context(client, context_filename):
    client.download(os.path.join(CONTEXT_FOLDER, context_filename), context_filename)
    ckks_context = load_context(os.path.join(CONTEXT_FOLDER, context_filename))
    return ckks_context


def get_ftp_credentials():
    # Get
    print("Fetching credentials")
    headers = {"token": g.ravenverse_token}
    r = requests.get(url="{}/client/ftp_credentials/".format(RAVENVERSE_URL), headers=headers)
    print(r.text)
    if r.status_code == 200:
        return r.json()
    return None


def get_graph(graph_id):
    # Get graph
    r = requests.get(url="{}/graph/get/?id={}".format(RAVENVERSE_URL, graph_id))
    if r.status_code == 200:
        return r.json()
    return None


def get_graphs():
    # Get graphs
    r = requests.get(url="{}/graph/get/all/?approach=federated".format(RAVENVERSE_URL))
    print(r.text)
    if r.status_code != 200:
        return None

    graphs = r.json()
    return graphs


def print_graphs(graphs):
    print("\nGraphs")
    for graph in graphs:
        print("\nGraph id:{}\n"
              "Name:{}\n"
              "Approach:{}\n"
              "Rules:{}".format(graph['id'], graph['name'], graph['approach'], graph['rules']))


def get_subgraph_ops(graph_id):
    # Get subgraph ops
    r = requests.get(url="{}/subgraph/ops/get/?graph_id={}".format(RAVENVERSE_URL, graph_id))
    if r.status_code == 200:
        return r.json()['subgraph_ops']
    return None


def get_rank(data):
    rank = len(np.array(data).shape)
    return rank


def apply_rules(data_columns, rules, final_column_names):
    data_silo = []

    for index, column_name in enumerate(final_column_names):
        data_column_rules = rules['rules'][column_name]

        if len(data_column_rules.keys()) == 0:
            data_column_values = []
            for value in data_columns[index]:
                data_column_values.append(value)
            data_silo.append(data_column_values)
            continue

        data_column_values = []
        for value in data_columns[index]:
            if data_column_rules['min'] < value < data_column_rules['max']:
                data_column_values.append(value)

        data_silo.append(data_column_values)
    return data_silo


def setTimeout(fn, ms, *args, **kwargs):
    timeoutId = Timer(ms / 1000., fn, args=args, kwargs=kwargs)
    timeoutId.start()
    return timeoutId


def stopTimer(timeoutId):
    # print("Timer stopped")
    if timeoutId is not None:
        timeoutId.cancel()


def dump_data(op_id, value):
    """
    Dump ndarray to file
    """
    file_path = os.path.join(FTP_TEMP_FILES_FOLDER, "temp_{}.pkl".format(op_id))
    if os.path.exists(file_path):
        os.remove(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pkl.dump(value, f)
    return file_path


def load_data(path):
    """
    Load ndarray from file
    """
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return np.array(data)
