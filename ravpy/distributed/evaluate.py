import time
import json
import os
from ravop import functions
from .compute import compute_locally, emit_error

from ..utils import setTimeout, stopTimer

from ..globals import g

timeoutId = g.timeoutId
opTimeout = g.opTimeout
initialTimeout = g.initialTimeout
client = g.client

@g.client.on('subgraph', namespace="/client")
def compute_subgraph(d):
    global client, timeoutId
    print("Received Subgraph : ",d["subgraph_id"]," of Graph : ",d["graph_id"])
    g.has_subgraph = True
    subgraph_id = d["subgraph_id"]
    graph_id = d["graph_id"]
    data = d["payloads"]
    results = []
    compute_success = True
    for index in data:
        # g.ops[index["op_id"]] = {
        #     "id": index["op_id"],
        #     "status": "pending",
        #     "startTime": int(time.time() * 1000),
        #     "endTime": None,
        #     "data": index
        # }

        # #Acknowledge op
        # client.emit("acknowledge", json.dumps({
        #         "op_id": index["op_id"],
        #         "message": "Op received"
        # }), namespace="/client")

        #Perform
        operation_type = index["op_type"]
        operator = index["operator"]
        if operation_type is not None and operator is not None:
            result_payload = compute_locally(index, subgraph_id, graph_id)
            if result_payload is not None:
                results.append(result_payload)
            else:
                compute_success = False
                break

        # stopTimer(timeoutId)
        # timeoutId = setTimeout(waitInterval,opTimeout)  
    if compute_success:
        emit_result_data = {"subgraph_id": d["subgraph_id"],"graph_id":d["graph_id"],"results":results}
        client.emit("subgraph_completed", json.dumps(emit_result_data), namespace="/client")
        print('Emitted subgraph_completed')

    g.has_subgraph = False
    
    stopTimer(timeoutId)
    timeoutId = setTimeout(waitInterval,opTimeout)


    for ftp_file in g.delete_files_list:
        g.ftp_client.delete_file(ftp_file)

    g.delete_files_list = []
    g.outputs = {}
    # g.ops = {}

# Check if the client is connected
@g.client.on('check_status', namespace="/client")
def check_status(d):
    global client
    client.emit('check_callback', d, namespace='/client')
    
def waitInterval():
    global client, timeoutId, opTimeout, initialTimeout
    client = g.client

    # for key in g.ops:
    #     op = g.ops[key]

    #     if op["status"] == "pending" or int(time.time() * 1000) - op["startTime"] < opTimeout:
    #         stopTimer(timeoutId)
    #         timeoutId = setTimeout(waitInterval,opTimeout)
    #         return
        
    #     if op["status"] == "pending" and int(time.time() * 1000) - op["startTime"] > opTimeout:
    #         op["status"] = "failure"
    #         op["endTime"] = int(time.time() * 1000)
    #         g.ops[key] = g.ops
    #         emit_error(op["data"], {"message": "OpTimeout error"})

    if not g.has_subgraph:
        client.emit("get_op", json.dumps({
                "message": "Send me an aop"
        }), namespace="/client")

    stopTimer(timeoutId)
    timeoutId = setTimeout(waitInterval,opTimeout)
