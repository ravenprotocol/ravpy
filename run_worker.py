import ast
import json
import os
import shutil
import time
from argparse import ArgumentParser

import requests

from compute import compute
from config import SOCKET_SERVER_URL, BENCHMARK_FILE_NAME, CONTEXT_FOLDER, CID

from globals import g
from ftp_client import get_client
from helpers import load_context
from utils import download_file, get_key
from ravop import compute_locally, functions, np

"""
1. Connect
2. Receive benchmarking ops
3. Receive ops/graphs
4. Get data and apply rules
5. If true, fetch ftp credentials
6. Calculate params
9. Upload params
"""

def get_graphs():
    # Get graphs
    r = requests.get(url="{}/graph/get/all/?approach=federated".format(SOCKET_SERVER_URL))
    if r.status_code != 200:
        return None

    graphs = r.json()
    for graph in graphs:
        print("{}:{}: {}, {}".format(graph['id'], graph['name'], graph['approach'], graph['algorithm']))


def get_graph(graph_id):
    # Get graph
    r = requests.get(url="{}/graph/get/?id={}".format(SOCKET_SERVER_URL, graph_id))
    if r.status_code == 200:
        return r.json()
    return None


def get_subgraph_ops(graph_id):
    # Get subgraph ops
    r = requests.get(url="{}/subgraph/ops/get/?graph_id={}&cid={}".format(SOCKET_SERVER_URL, graph_id, CID))
    if r.status_code == 200:
        return r.json()['subgraph_ops']
    print(r.status_code, r.text)
    return None


def get_rank(data):
    rank = len(np.array(data).shape)
    return rank


def apply_rules(data_columns, rules):
    data_silo = []
    if len(data_columns) == len(rules['rules']):
        for index, data_column in enumerate(data_columns):
            data_column_rules = rules['rules'][index]

            data_column_values = []
            for value in data_column:
                if data_column_rules['min'] < value < data_column_rules['max']:
                    data_column_values.append(value)

            data_silo.append(data_column_values)
        return data_silo
    else:
        return None


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--action", type=str, help="Enter action", default=None)
    argparser.add_argument("--federated_id", type=str, help="Id of the federated graph", default=None)

    args = argparser.parse_args()

    if args.action is None:
        raise Exception("Enter action")

    if args.action == "list":
        get_graphs()

    elif args.action == "participate":
        if args.federated_id is None:
            raise Exception("Enter id of the federated")

        print("Let's participate")

        # Connect
        client = g.client
        print(client)

        graph = get_graph(graph_id=args.federated_id)
        print(graph)
        subgraph_ops = get_subgraph_ops(graph["id"])
        print(subgraph_ops)
        graph_rules = ast.literal_eval(graph['rules'])

        print(graph, graph_rules)

        user_choice = input("How do you want to input your data samples?(0: file, 1: other): ")

        while user_choice not in ["0", "1"]:
            user_choice = input("How do you want to input your data samples?(0: file, 1: other): ")

        if user_choice == "0":
            file_path = input("Enter file path: ")
            data_columns = []
            with open(file_path, "r") as f:
                for line in f:
                    data_column = json.loads(line)
                    data_columns.append(data_column)

            data_silo = apply_rules(data_columns, rules=graph_rules)
            print(data_silo)
            if data_silo is not None:
                # Fetch ftp credentials

                compute(data_silo, graph, subgraph_ops)
            else:
                print("You can't participate as your data is it in the wrong format")

        elif user_choice == "1":
            data_columns = input("Enter data: ")
            data_columns = ast.literal_eval(data_columns)
            data_silo = apply_rules(data_columns, rules=graph_rules)
            print(data_silo)
            if data_silo is not None:
                compute(data_silo, graph, subgraph_ops)
            else:
                print("You can't participate as your data is it in the wrong format")
