import sys
from argparse import ArgumentParser

from dotenv import load_dotenv

load_dotenv()

from ravpy import participate_federated
from ravpy.globals import g
from ravpy.utils import get_graphs, print_graphs, get_graph

if __name__ == '__main__':
    print("Executed")
    argparser = ArgumentParser()
    argparser.add_argument("-a", "--action", type=str, help="Enter action", default=None)
    argparser.add_argument("-c", "--cid", type=str, help="Enter client id", default=None)
    argparser.add_argument("-g", "--graph_id", type=str, help="Id of the graph", default=None)
    argparser.add_argument("-d", "--data", type=str, help="Data to use", default=None)
    argparser.add_argument("-f", "--file_path", type=str, help="File path containing samples to use", default=None)

    print(argparser.parse_args())

    if len(sys.argv) == 1:
        print("Args missing")
        argparser.print_help(sys.stderr)
        sys.exit(1)

    args = argparser.parse_args()

    if args.action is None:
        print("Enter action")
        raise Exception("Enter action")

    if args.action == "list":
        graphs = get_graphs()
        print_graphs(graphs)

    elif args.action == "participate":
        if args.cid is None:
            print("Client id is required")
            raise Exception("Client id is required")

        if args.graph_id is None:
            print("Enter id of the graph to join")
            raise Exception("Enter id of the graph to join")

        print("Let's participate")

        # connect
        g.cid = args.cid
        client = g.client

        if client is None:
            g.client.disconnect()
            raise Exception("Unable to connect to ravsock. Make sure you are using the right hostname and port")

        # Connect
        graph = get_graph(graph_id=args.graph_id)
        if graph is None:
            g.client.disconnect()
            raise Exception("Invalid graph id")

        # Check file path
        if args.file_path is None:
            g.client.disconnect()
            raise Exception("Provide values or file path to use")

        participate_federated(args.cid, graph_id=args.graph_id, file_path=args.file_path)
