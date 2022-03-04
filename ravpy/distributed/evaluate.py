import time
import json

from ravop import functions
from .compute import compute_locally, emit_error

from ..utils import setTimeout, stopTimer

from ..globals import g

timeoutId = g.timeoutId
ops = g.ops
opTimeout = g.opTimeout
initialTimeout = g.initialTimeout
outputs = g.outputs
client = None

@g.client.on('subgraph', namespace="/client")
def compute_subgraph(d):
    global ops, client, timeoutId
    print("Subgraph Received...")
    print(d)

    data = d

    for index in data:
        ops[index["op_id"]] = {
            "id": index["op_id"],
            "status": "pending",
            "startTime": int(time.time() * 1000),
            "endTime": None,
            "data": index
        }

        #Acknowledge op
        client.emit("acknowledge", json.dumps({
                "op_id": index["op_id"],
                "message": "Op received"
        }), namespace="/client")

        #Perform
        operation_type = index["op_type"]
        operator = index["operator"]
        if operation_type is not None and operator is not None:
            compute_locally(index)

        stopTimer(timeoutId)
        timeoutId = setTimeout(waitInterval,opTimeout)

# Check if the client is connected
@g.client.on('check_status', namespace="/client")
def check_status(d):
    global client
    client.emit('check_callback', d, namespace='/client')
    
def waitInterval():
    global client, timeoutId, ops, opTimeout, initialTimeout
    client = g.client

    print("Function Called")
    for key in ops:
        op = ops[key]

        if op["status"] == "pending" or int(time.time() * 1000) - op["startTime"] < opTimeout:
            stopTimer(timeoutId)
            timeoutId = setTimeout(waitInterval,opTimeout)
            return
        
        if op["status"] == "pending" and int(time.time() * 1000) - op["startTime"] > opTimeout:
            op["status"] = "failure"
            op["endTime"] = int(time.time() * 1000)
            ops[key] = ops
            emit_error(op["data"], {"message": "OpTimeout error"})

    client.emit("get_op", json.dumps({
            "message": "Send me an aop"
    }), namespace="/client")

    stopTimer(timeoutId)
    timeoutId = setTimeout(waitInterval,opTimeout)
