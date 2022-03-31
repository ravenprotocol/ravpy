from ast import operator
from distutils.log import error
import numpy as np
import json
import time
#import tensorflow
from scipy import stats


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
            'mul': 'np.multiply',
            'matmul': 'np.matmul',
            'multiply':'np.multiply',
            'dot': 'np.dot',
            'split': 'np.split', 
            'reshape':'reshape', 
            'unique': 'np.unique', 
            'expand_dims':'expand_dims', 
            'inv': 'np.linalg.inv', 
            'gather': 'gather', 
            'stack': 'np.stack', 
            'tile': 'np.tile', 
            'slice': 'slice',

            'find_indices': 'find_indices',
            'shape':'np.shape',

            #Comparision ops
            'greater': 'np.greater',
            'greater_equal':'np.greater_equal' ,
            'less': 'np.less',
            'less_equal':'np.less_equal' ,
            'equal':'np.equal' ,
            'not_equal': 'np.not_equal',
            'logical_and':'np.logical_and' ,
            'logical_or': 'np.logical_or',
            'logical_not': 'np.logical_not',
            'logical_xor': 'np.logical_xor',

            #statistics
            'mean': 'np.mean',
            'average': 'np.average',
            'mode': 'stats.mode',
            'variance': 'np.var',
            'std': 'np.std', 
            'percentile': 'np.percentile',
            'random': 'np.random',
            'bincount': 'np.bincount',
            'where': 'where',
            #'sign': Operators.SIGN,  
            'foreach': 'foreach',
            'set_value': 'set_value',


            'concat': 'np.concatenate',
            'cube': 'np.cbrt'
    }


def compute_locally_bm(*args, **kwargs):
    operator = kwargs.get("operator", None)
    op_type = kwargs.get("op_type", None)
    param_args =kwargs.get("params",None)
    print("Operator", operator,"Op Type:",op_type)
    if op_type == "unary":
        value1 = args[0]
        t1=time.time()
        params=""
        if param_args is not None:
            params=[op_param_mapping[operator][str(_)] for _ in param_args ]
        print(params)
        expression="{}({})".format(numpy_functions[operator], value1['value'])
        eval(expression)
        t2=time.time()
        return t2-t1

    elif op_type == "binary":
        value1 = args[0]['value']
        value2 = args[1]['value']
        t1=time.time()
        eval("{}({},{})".format(numpy_functions[operator], value1, value2))
        t2=time.time()
        return t2-t1

# async 
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
            try:
                values.append(outputs[payload['values'][i]['op_id']])
            except Exception as e:
                emit_error(payload,e)

    payload["values"] = values

    print("Payload Values: ", payload["values"])

    op_type = payload["op_type"]
    operator = payload["operator"]
    # param='params': {'begin': 0, 'size': 25}
    params=payload['params']
    param_string=""
    for i in params.keys():
        if type(params[i]) == str:
            param_string+=","+i+"=\'"+str(params[i])+"\'"
        else:
            param_string+=","+i+"="+str(params[i])

    print("\n\n\n",param_string,"\n\n")


    try:
        if op_type == "unary":
            value1 = payload["values"][0]
            short_name = get_key(operator,functions)
            result = eval("{}({}{})".format(numpy_functions[short_name], value1,param_string))

        elif op_type == "binary":
            value1 = payload["values"][0]
            value2 = payload["values"][1]
            short_name = get_key(operator,functions)
            expression="{}({}, {}{})".format(numpy_functions[short_name], value1, value2,param_string)



            result = eval(expression)
        print("Result : \n",result)
        # await 
        emit_result(payload, result)

    except Exception as error:
        # await 
        emit_error(payload, error)



# async 
def emit_result(payload, result):
    global outputs, ops
    client = g.client
    try:
        result = result.tolist()
    except:
        result=result
    
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

    # await 
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
# async 
def emit_error(payload, error):
    print("Emit Error")
    print(payload)
    print(error)
    error=str(error)
    global ops
    client = g.client
    # await 
    client.emit("op_completed", json.dumps({
            'op_type': payload["op_type"],
            'result': error,
            'values': payload["values"],
            'operator': payload["operator"],
            "op_id": payload["op_id"],
            "status": "failure"
    }), namespace="/client")

    op = ops[payload["op_id"]]
    op["status"] = "failure"
    op["endTime"] = int(time.time() * 1000)
    ops[payload["op_id"]] = op




#ops:
def slice(tensor,begin=None,size=None):
    result=tensor[begin:begin+size]
    return result

def gather(tensor,indices):
    result=[]
    for i in indices:
        result.append(tensor[i])
    return result
    
def where(a,b,condition=None):
    if condition is None:
        raise Exception("condition is missing")
    else:
        result=np.where(a,condition,b)
    return result
 
def split(arr,numOrSizeSplits=None,axis=None):
    result=np.split(arr,numOrSizeSplits,axis=axis)
    return result

def expand_dims(arr,**kwargs):
    axis=kwargs.get('axis')
    if axis is not None:
        result=np.expand_dims(arr,axis=axis)
    else:
        result= np.expand_dims(arr,axis=0)
    return result



def tile(arr,reps):
    if reps is not None:
        result=np.tile(arr,reps)
    else:
        emit_error()
    return result



def one_hot_encoding(arr,depth):
    return np.squeeze(np.eye(depth)[arr.reshape(-1)])


def foreach(val,**kwargs):
    print(val,kwargs)
    operator=kwargs.get("operation")
    print("operation")
    result=[]
    paramstr=""
    del kwargs['operation']
    for _ in kwargs.keys():
        paramstr+=","+_+"="+str(kwargs.get(_))
    for i in val:
        evalexp="{}({}{})".format(numpy_functions[operator],i,paramstr)
        res=eval(evalexp)
        if type(res) is np.ndarray:
            result.append(res.tolist())
        else:
            result.append(res)
    return result


def find_indices(arr,val):
    result=[]
    for i in val:
        indices = [_ for _, arr in enumerate(arr) if arr == i]
        result.append(indices)
    if len(val) == 1:
        return indices
    else:
        return result

def reshape(tens,shape=None):
    if shape is None:
        return None
    else:
        return np.reshape(tens,newshape=shape)
    pass


