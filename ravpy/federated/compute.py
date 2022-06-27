import ast
import logging
import os
import pickle

import numpy as np

from ..config import PARAMS_DIR, ENCRYPTION
from ..ftp import get_client
from ..globals import g
from ..utils import get_ftp_credentials, fetch_and_load_context

logger = logging.getLogger(__name__)


def compute(data_silo, graph, subgraph_ops, final_column_names):
    print("Get ftp credentials")
    credentials = get_ftp_credentials()#cid)
    if credentials is None:
        raise Exception("Error")
    print("FTP credentials: {}".format(credentials['ftp_credentials']))

    ftp_client = get_client(**ast.literal_eval(credentials['ftp_credentials']))

    if ENCRYPTION:
        print("Downloading context file...")
        ckks_context = fetch_and_load_context(client=ftp_client,
                                              context_filename="context_without_private_key_{}.txt".format(cid))

    final_params = {}
    for subgraph_dict in subgraph_ops:
        subgraph_params = dict()
        for op in subgraph_dict['ops']:
            op_params = []
            operator = op['operator']
            for data_column in data_silo:
                maximum = max(data_column)
                minimum = min(data_column)

                mean = None
                variance = None
                standard_deviation = None
                size = len(data_column)

                if operator == "federated_mean":
                    mean = sum(data_column) / len(data_column)
                elif operator == "federated_variance":
                    mean = sum(data_column) / len(data_column)
                    variance = sum((i - mean) ** 2 for i in data_column) / len(data_column)
                elif operator == "federated_standard_deviation":
                    mean = sum(data_column) / len(data_column)
                    variance = sum((i - mean) ** 2 for i in data_column) / len(data_column)
                    standard_deviation = np.sqrt(variance)

                if ENCRYPTION:
                    import tenseal as ts
                    print('Encrypting params...')
                    if mean is not None:
                        mean = ts.ckks_tensor(ckks_context, [mean]).serialize()

                    if variance is not None:
                        variance = ts.ckks_tensor(ckks_context, [variance]).serialize()

                    if standard_deviation is not None:
                        standard_deviation = ts.ckks_tensor(ckks_context, [standard_deviation]).serialize()

                    if size is not None:
                        size = ts.ckks_tensor(ckks_context, [size]).serialize()

                    if minimum is not None:
                        minimum = ts.ckks_tensor(ckks_context, [minimum]).serialize()

                    if maximum is not None:
                        maximum = ts.ckks_tensor(ckks_context, [maximum]).serialize()

                op_params.append({
                    "federated_mean": mean,
                    "federated_variance": variance,
                    "federated_standard_deviation": standard_deviation,
                    "size": size,
                    "minimum": minimum,
                    "maximum": maximum
                })
            subgraph_params[op['id']] = op_params
        final_params[subgraph_dict['id']] = subgraph_params

    result = dict()
    result["graph_id"] = graph['id']
    result["encryption"] = ENCRYPTION
    result["params"] = final_params
    result["column_names"] = final_column_names

    params_filename = "params_{}.pkl".format(graph['id'])
    params_file = os.path.join(PARAMS_DIR, params_filename)

    with open(params_file, "wb") as f:
        pickle.dump(result, f)

    print("Uploading")
    ftp_client.upload(params_file, params_filename)
    print("Uploaded")

    print('Emitting..')
    if not g.client.connected:
        print("Not connected")
    g.client.emit('params',
                  {"status": "success", "graph_id": graph['id'], "params_file": params_filename},
                  namespace='/client')
    g.client.sleep(10)
    print("Emitted")
    g.client.disconnect()
