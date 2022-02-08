import ast
from argparse import ArgumentParser

import pandas as pd
import requests
import numpy as np

from ravpy.config import SOCKET_SERVER_URL, CID
from ravpy.federated import compute
from ravpy.globals import g


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

        print("\n\nGRAPH RULEZZ", graph_rules)

        user_choice = input("How do you want to input your data samples?(0: file, 1: other): ")

        while user_choice not in ["0", "1"]:
            user_choice = input("How do you want to input your data samples?(0: file, 1: other): ")

        if user_choice == "0":
            file_path = input("Enter file path: ")

            dataframe = pd.read_csv(file_path)
            column_names = []
            for col in dataframe.columns:
                column_names.append(col)
            # column_names.sort()

            final_column_names = []
            for key, value in graph_rules['rules'].items():
                if key in column_names:
                    final_column_names.append(key)
                else:
                    raise Exception('Incorrect Rules Format.')
            final_column_names.sort()

            data_columns = []
            for column_name in final_column_names:
                column = dataframe[column_name].tolist()
                data_columns.append(column)

            data_silo = apply_rules(data_columns, rules=graph_rules, final_column_names=final_column_names)
            print(data_silo)
            if data_silo is not None:
                compute(data_silo, graph, subgraph_ops, final_column_names)
            else:
                print("You can't participate as your data is it in the wrong format")

        # elif user_choice == "1":
        #     data_columns = input("Enter data: ")
        #     data_columns = ast.literal_eval(data_columns)
        #     data_silo = apply_rules(data_columns, rules=graph_rules)
        #     print(data_silo)
        #     if data_silo is not None:
        #         compute(data_silo, graph, subgraph_ops, final_column_names)
        #     else:
        #         print("You can't participate as your data is it in the wrong format")
