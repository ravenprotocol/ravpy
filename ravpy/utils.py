import ast
import os
import pickle as pkl
import speedtest
import time

from .config import ENCRYPTION, RAVENVERSE_FTP_URL, RAVENAUTH_TOKEN_VERIFY_URL

if ENCRYPTION:
    import tenseal as ts

from .ftp import get_client as get_ftp_client
import shutil
import numpy as np
import requests
from terminaltables import AsciiTable
from .config import ENCRYPTION

if ENCRYPTION:
    import tenseal as ts
from .config import BASE_DIR, CONTEXT_FOLDER, RAVENVERSE_URL, FTP_TEMP_FILES_FOLDER
from threading import Timer
from .globals import g

import json
import urllib.request
from pip._internal.operations.freeze import freeze

def isLatestVersion(pkgName):
    # Get the currently installed version
    current_version = ''
    for requirement in freeze(local_only=False):
        pkg = requirement.split('==')
        if pkg[0] == pkgName:
            current_version = pkg[1]
    # Check pypi for the latest version number
    contents = urllib.request.urlopen('https://pypi.org/pypi/'+pkgName+'/json').read()
    data = json.loads(contents)
    latest_version = data['info']['version']
    # print(‘Current version of ‘+pkgName+’ is ’+current_version)
    # print(‘Latest version of ‘+pkgName+’ is ’+latest_version)
    return latest_version == current_version

def download_file(url, file_name):
    g.logger.debug("Downloading benchmark data")
    headers = {"token": g.ravenverse_token}
    with requests.get(url, stream=True, headers=headers) as r:
        with open(file_name, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    g.logger.debug("Benchmark data downloaded")


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
    """
    Fetch ftp credentials
    :return: json response
    """
    g.logger.debug("Fetching ftp credentials...")
    headers = {"token": g.ravenverse_token}
    r = requests.get(url="{}/client/ftp_credentials/".format(RAVENVERSE_URL), headers=headers)
    if r.status_code == 200:
        g.logger.debug("Credentials fetched successfully")
        return r.json()
    else:
        g.logger.debug("Unable to fetch ftp credentials. Try again after some time or "
                       "contact our team at team@ravenprotocol.com")
        return None


def get_graph(graph_id):
    # Get graph
    g.logger.debug("get_graph")
    headers = {"token": g.ravenverse_token}
    r = requests.get(url="{}/graph/get/?id={}".format(RAVENVERSE_URL, graph_id), headers=headers)
    if r.status_code == 200:
        return r.json()
    return None


def get_federated_graph(graph_id):
    # Get graph
    g.logger.debug("get_federated_graph")
    headers = {"token": g.ravenverse_token}
    r = requests.get(url="{}/graph/get_federated/?id={}".format(RAVENVERSE_URL, graph_id), headers=headers)
    if r.status_code == 200:
        return r.json()
    return None


def list_graphs(approach=None):
    # Get graphs
    headers = {"token": g.ravenverse_token}
    r = requests.get(url="{}/graph/get/all/?approach={}".format(RAVENVERSE_URL, approach), headers=headers)
    if r.status_code != 200:
        return None

    graphs = r.json()
    g.logger.debug(AsciiTable([["{} Graphs".format(approach)]]).table)
    table_data = [["Id", "Name", "Approach", "Algorithm", "Rules"]]

    for graph in graphs:
        table_data.append([graph['id'], graph['name'], graph['approach'], graph['algorithm'], graph['rules']])

    g.logger.debug(AsciiTable(table_data).table)

    return graphs


def print_graphs(graphs):
    g.logger.debug("\nGraphs")
    for graph in graphs:
        g.logger.debug("\nGraph id:{}\n"
                       "Name:{}\n"
                       "Approach:{}\n"
                       "Rules:{}".format(graph['id'], graph['name'], graph['approach'], graph['rules']))


def get_subgraph_ops(graph_id):
    # Get subgraph ops
    headers = {"token": g.ravenverse_token}
    r = requests.get(url="{}/subgraph/ops/get/?graph_id={}".format(RAVENVERSE_URL, graph_id), headers=headers)
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
        pkl.dump(value, f, protocol=pkl.HIGHEST_PROTOCOL)
    return file_path


def load_data(path):
    """
    Load ndarray from file
    """
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return np.array(data)

def load_data_raw(path):
    """
    Load data from file
    """
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data


def initialize_ftp_client():
    credentials = get_ftp_credentials()

    if credentials is None:
        return None

    creds = ast.literal_eval(credentials['ftp_credentials'])
    time.sleep(2)

    try:
        g.logger.debug("")
        g.logger.debug("Testing network speed...")
        if RAVENVERSE_FTP_URL != 'localhost' and RAVENVERSE_FTP_URL != '0.0.0.0':
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

    except Exception as e:
        g.ftp_upload_blocksize = 8192 * 1000
        g.ftp_download_blocksize = 8192 * 1000

    g.logger.debug("FTP Upload Blocksize:{}".format(g.ftp_upload_blocksize))
    g.logger.debug("FTP Download Blocksize:  {}\n".format(g.ftp_download_blocksize))

    """
    Create ftp client
    """
    g.ftp_client = get_ftp_client(creds['username'], creds['password'])

    return g.ftp_client


def verify_token(token):
    """
    Verify user token
    :param token: token
    :return: valid or not
    """
    r = requests.post(RAVENAUTH_TOKEN_VERIFY_URL, data={"token": token})
    if r.status_code != 200:
        g.logger.debug("Error:{}".format(r.text))
        return False
    else:
        return True


def disconnect():
    if g.client.connected:
        g.logger.debug("Disconnecting...")
        if g.client.connected:
            g.client.emit("disconnect", namespace="/client")
            g.logger.debug("Disconnected")

    return True
