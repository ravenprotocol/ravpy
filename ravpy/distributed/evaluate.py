import json

from .compute import compute_locally
from ..globals import g
from ..utils import setTimeout, stopTimer

timeoutId = g.timeoutId
opTimeout = g.opTimeout
initialTimeout = g.initialTimeout
client = g.client


@g.client.on('subgraph', namespace="/client")
def compute_subgraph(d):
    global client, timeoutId
    g.logger.debug("Received subgraph: {} of graph: {}".format(d["subgraph_id"], d["graph_id"]))
    g.logger.debug("Computing subgraphs(ops)...")
    g.has_subgraph = True
    subgraph_id = d["subgraph_id"]
    graph_id = d["graph_id"]
    data = d["payloads"]
    results = []
    compute_success = True
    g.error = False

    for index in data:
        # Perform
        operation_type = index["op_type"]
        operator = index["operator"]
        if operation_type is not None and operator is not None:
            result_payload = compute_locally(index, subgraph_id, graph_id)

            if not g.error:
                results.append(result_payload)
            else:
                break

        # stopTimer(timeoutId)
        # timeoutId = setTimeout(waitInterval,opTimeout)  
    if not g.error:
        emit_result_data = {"subgraph_id": d["subgraph_id"], "graph_id": d["graph_id"], "token": g.ravenverse_token,
                            "results": results}
        client.emit("subgraph_completed", json.dumps(emit_result_data), namespace="/client")
        g.logger.debug('Subgraph results emitted successfully')

    g.has_subgraph = False

    stopTimer(timeoutId)
    timeoutId = setTimeout(waitInterval, opTimeout)

    for ftp_file in g.delete_files_list:
        g.ftp_client.delete_file(ftp_file)

    g.delete_files_list = []
    g.outputs = {}
    # g.ops = {}

    g.logger.debug("")
    g.logger.debug("Waiting for more ops and subgraphs...")


# # Check if the client is connected
# @g.client.on('check_status', namespace="/client")
# def check_status(d):
#     global client
#     # g.logger.debug("check_status:{}".format(d))
#     client.emit('check_callback', d, namespace='/client')

@g.client.on('ping', namespace="/client")
def ping(d):
    global client
    # g.logger.debug("\n\n\nPing: {}".format(d))
    client.emit('pong', d, namespace='/client')


def waitInterval():
    # g.logger.debug("waitInterval")
    global client, timeoutId, opTimeout, initialTimeout
    client = g.client

    # g.logger.debug("{} {}".format(g.client, g.client.connected))

    if g.client.connected:
        if not g.has_subgraph:
            client.emit("get_op", json.dumps({
                "message": "Send me an aop"
            }), namespace="/client")

        stopTimer(timeoutId)
        timeoutId = setTimeout(waitInterval, opTimeout)
