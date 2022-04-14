import os
import shutil

import numpy as np
import requests

from .config import ENCRYPTION

if ENCRYPTION:
    import tenseal as ts

from .config import BASE_DIR, CONTEXT_FOLDER, SOCKET_SERVER_URL, FTP_TEMP_FILES_FOLDER

from threading import Timer  

class Singleton:
    def __init__(self, cls):
        self._cls = cls

    def Instance(self):
        try:
            return self._instance
        except AttributeError:
            self._instance = self._cls()
            return self._instance

    def __call__(self):
        raise TypeError("Singletons must be accessed through `Instance()`.")

    def __instancecheck__(self, inst):
        return isinstance(inst, self._cls)


def download_file(url, file_name):
    with requests.get(url, stream=True) as r:
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
    from .utils import load_context
    client.download(os.path.join(CONTEXT_FOLDER, context_filename), context_filename)
    ckks_context = load_context(os.path.join(CONTEXT_FOLDER, context_filename))
    return ckks_context


def get_ftp_credentials(cid):
    # Get
    r = requests.get(url="{}/client/ftp_credentials/?cid={}".format(SOCKET_SERVER_URL, cid))
    if r.status_code == 200:
        return r.json()
    return None


def get_graph(graph_id):
    # Get graph
    r = requests.get(url="{}/graph/get/?id={}".format(SOCKET_SERVER_URL, graph_id))
    if r.status_code == 200:
        return r.json()
    return None


def get_graphs():
    # Get graphs
    r = requests.get(url="{}/graph/get/all/?approach=federated".format(SOCKET_SERVER_URL))
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


def get_subgraph_ops(graph_id, cid):
    # Get subgraph ops
    r = requests.get(url="{}/subgraph/ops/get/?graph_id={}&cid={}".format(SOCKET_SERVER_URL, graph_id, cid))
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
    file_path = os.path.join(FTP_TEMP_FILES_FOLDER, "temp_{}.npy".format(op_id))
    if os.path.exists(file_path):
        os.remove(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(file_path, value)
    return file_path

def load_data(path):
    """
    Load ndarray from file
    """
    return np.load(path, allow_pickle=True)