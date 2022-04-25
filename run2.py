import ast
import json
import logging
import os
import pickle
import sys
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from ravpy.config import PARAMS_DIR, ENCRYPTION
from ravpy.ftp import get_client
from ravpy.globals import g
from ravpy.utils import get_ftp_credentials, fetch_and_load_context
from ravpy.utils import get_graphs, print_graphs, get_graph, get_subgraph_ops, apply_rules

logger = logging.getLogger(__name__)


def print_json(response):
    print(json.dumps(response))


def compute(cid, data_silo, graph, subgraph_ops, final_column_names):
    print_json({"status": "success", "message": "Participation started", "code": 1000})
    credentials = get_ftp_credentials(cid)
    if credentials is None:
        print_json({"status": "failure", "message": "No credentials"})

    ftp_client = get_client(**ast.literal_eval(credentials['ftp_credentials']))

    if ENCRYPTION:
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

    print_json({"status": "success", "message": "Uploading"})
    ftp_client.upload(params_file, params_filename)
    print_json({"status": "success", "message": "Uploaded"})

    print_json({"status": "success", "message": "Emitting"})
    if not g.client.connected:
        print_json({"status": "failure", "message": "Not connected to ravsock"})
    g.client.emit('params',
                  {"status": "success", "graph_id": graph['id'], "params_file": params_filename},
                  namespace='/client')
    g.client.sleep(10)
    print_json({"status": "success", "message": "Emitted"})
    g.client.disconnect()

    print_json({"status": "success", "message": "Participation ended", "code": 1001})


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("-a", "--action", type=str, help="Enter action", default=None)
    argparser.add_argument("-c", "--cid", type=str, help="Enter client id", default=None)
    argparser.add_argument("-g", "--graph_id", type=str, help="Id of the graph", default=None)
    argparser.add_argument("-d", "--data", type=str, help="Data to use", default=None)
    argparser.add_argument("-f", "--file_path", type=str, help="File path containing samples to use", default=None)

    if len(sys.argv) == 1:
        print_json({"status": "failure", "message": "Args are missing"})
    else:
        args = argparser.parse_args()

        if args.action is None:
            print_json({"status": "failure", "message": "Action is missing"})
        else:
            if args.action == "list":
                graphs = get_graphs()
                print_graphs(graphs)

            elif args.action == "participate":
                if args.cid is None:
                    print_json({"status": "failure", "message": "Client id is required"})
                else:
                    if args.graph_id is None:
                        print_json({"status": "failure", "message": "Graph id is missing"})
                    else:
                        # connect
                        g.cid = args.cid
                        client = g.client

                        if client is None:
                            g.client.disconnect()
                            print_json({"status": "failure",
                                        "message": "Unable to connect to ravsock. "
                                                   "Make sure you are using the right hostname and port"})
                        else:
                            # Connect
                            graph = get_graph(graph_id=args.graph_id)
                            if graph is None:
                                g.client.disconnect()
                                print_json({"status": "failure",
                                            "message": "Invalid graph id"})
                            else:

                                subgraph_ops = get_subgraph_ops(graph["id"], cid=args.cid)
                                graph_rules = ast.literal_eval(graph['rules'])

                                if args.data is None and args.file_path is None:
                                    g.client.disconnect()
                                    print_json({"status": "failure",
                                                "message": "Provide values or file path to use"})
                                else:
                                    if args.data is not None:
                                        pass
                                        # data_columns = args.data
                                        # data_columns = ast.literal_eval(args.data)
                                        # data_silo = apply_rules(data_columns, rules=graph_rules)
                                        # print_json(data_silo)
                                        # if data_silo is not None:
                                        #     compute(data_silo, graph, subgraph_ops, final_column_names)
                                        # else:
                                        #     print_json("You can't participate as your data is it in the wrong format")

                                    elif args.file_path is not None:
                                        dataframe = pd.read_csv(args.file_path)
                                        column_names = []
                                        for col in dataframe.columns:
                                            column_names.append(col)
                                        # column_names.sort()

                                        continue_ = True
                                        final_column_names = []
                                        for key, value in graph_rules['rules'].items():
                                            if key in column_names:
                                                final_column_names.append(key)
                                            else:
                                                continue_ = False
                                                break

                                        if continue_ and len(final_column_names) > 0:
                                            final_column_names.sort()

                                            data_columns = []
                                            for column_name in final_column_names:
                                                column = dataframe[column_name].tolist()
                                                data_columns.append(column)

                                            data_silo = apply_rules(data_columns, rules=graph_rules,
                                                                    final_column_names=final_column_names)
                                            if data_silo is not None:
                                                compute(args.cid, data_silo, graph, subgraph_ops, final_column_names)
                                            else:
                                                print_json({"status": "failure",
                                                            "message": "You can't participate as your data "
                                                                       "is it in the wrong format"})
                                        else:
                                            print_json({"status": "failure",
                                                        "message": "Incorrect rules format"})
                                    else:
                                        print_json({"status": "failure",
                                                    "message": "Provide data or file path"})
