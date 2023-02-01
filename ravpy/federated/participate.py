import ast
import os
import pandas as pd

from .compute import compute
from ..utils import apply_rules, get_subgraph_ops, get_federated_graph


def participate(graph_id, file_path):
    graph = get_federated_graph(graph_id=graph_id)
    if graph is None:
        raise Exception("Invalid graph id")

    subgraph_ops = get_subgraph_ops(graph["id"])
    if subgraph_ops is not None:
        graph_rules = ast.literal_eval(graph['rules'])

        # Read file
        df = pd.read_csv(file_path)
        column_names = df.columns.tolist()

        # Find columns required for the graph
        final_column_names = []
        for key, value in graph_rules['rules'].items():
            if key in column_names:
                final_column_names.append(key)
            else:
                raise Exception('Incorrect Rules Format.')
        final_column_names.sort()

        data_columns = []
        for column_name in final_column_names:
            column = df[column_name].tolist()
            data_columns.append(column)

        data_silo = apply_rules(data_columns, rules=graph_rules, final_column_names=final_column_names)
        if data_silo is not None:
            compute(data_silo, graph, subgraph_ops, final_column_names)
        else:
            print("You can't participate as your data is it in the wrong format")
    else:
        print("Invalid graph")
        os.exit(1)
