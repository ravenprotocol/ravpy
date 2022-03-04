from ast import operator
from distutils.log import error
import numpy as np
import json
import time

from ..globals import g
from ..utils import get_key
from ravop import functions

outputs = g.outputs
ops = g.ops

numpy_functions = {
    "neg": "np.negative",
    "pos": "np.positive",
    "add": "np.add",
    "sub": "np.subtract",
    "exp": "np.exp",
    "natlog": "np.log",
    "square":"np.square",
    "pow":"np.power",
    "square_root":"np.sqrt",
    "cube_root":"np.cbrt",
    "abs":"np.abs",
    "sum":"np.sum",
    "sort":"np.sort",
    "reverse":"np.flip",
    "min":"np.min",
    "max":"np.max",
    "argmax":"np.argmax",
    "argmin":"np.argmin",
    "transpose":"np.transpose",
    "div":"np.divide",
    # "concat":"np." needs tuple of arrays
}


def compute_locally_bm(*args, **kwargs):
    operator = kwargs.get("operator", None)
    op_type = kwargs.get("op_type", None)
    print("Operator", operator)
    if op_type == "unary":
        value1 = args[0]

        return eval("{}({})".format(numpy_functions[operator], value1))

    elif op_type == "binary":
        value1 = args[0]
        value2 = args[1]

        return eval("{}({}, {})".format(numpy_functions[operator], value1, value2))

def compute_locally(payload):
    global outputs

    print("Computing ",payload["operator"])
    print(payload)

    values = []
    for i in range(len(payload["values"])):
        if "value" in payload["values"][i].keys():
            print("From server")
            values.append(payload["values"][i]["value"])

        elif "op_id" in payload["values"][i].keys():
            print("From client")
            values.append(outputs[payload["values"][i]["op_id"]])

    payload["values"] = values

    print("Payload Values: ", payload["values"])

    op_type = payload["op_type"]
    operator = payload["operator"]

    try:
        if op_type == "unary":
            value1 = payload["values"][0]
            short_name = get_key(operator,functions)
            result = eval("{}({})".format(numpy_functions[short_name], value1))

        elif op_type == "binary":
            value1 = payload["values"][0]
            value2 = payload["values"][1]
            short_name = get_key(operator,functions)
            result = eval("{}({}, {})".format(numpy_functions[short_name], value1, value2))
        emit_result(payload, result)

    except Exception as error:
        emit_error(payload, error)

def emit_result(payload, result):
    global outputs, ops
    client = g.client
    result = result.tolist()
    print("Emit Success")
    print(payload)
    print(result, json.dumps({
        'op_type': payload["op_type"],
        'result': result,
        'values': payload["values"],
        'operator': payload["operator"],
        "op_id": payload["op_id"],
        "status": "success"
    }))

    outputs[payload["op_id"]] = result

    client.emit("op_completed", json.dumps({
        'op_type': payload["op_type"],
        'result': result,
        'values': payload["values"],
        'operator': payload["operator"],
        "op_id": payload["op_id"],
        "status": "success"
    }), namespace='/client')

    op = ops[payload["op_id"]]
    op["status"] = "success"
    op["endTime"] = int(time.time() * 1000)
    ops[payload["op_id"]] = op
 


def emit_error(payload, error):
    print("Emit Error")
    print(payload)
    print(error)
    global ops
    client = g.client
    client.emit("op_completed", json.dumps({
            'op_type': payload["op_type"],
            'result': error["message"],
            'values': payload["values"],
            'operator': payload["operator"],
            "op_id": payload["op_id"],
            "status": "failure"
    }), namespace="/client")

    op = ops[payload["op_id"]]
    op["status"] = "failure"
    op["endTime"] = int(time.time() * 1000)
    ops[payload["op_id"]] = op