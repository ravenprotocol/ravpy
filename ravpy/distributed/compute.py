from ast import operator
from distutils.log import error
import os
import numpy as np
import json
import sys
import time
#import tensorflow
from scipy import stats


from ..globals import g
from ..utils import get_key, dump_data, get_ftp_credentials, load_data
from ..ftp import get_client as get_ftp_client
from ..ftp import check_credentials as check_credentials
from ..config import FTP_DOWNLOAD_FILES_FOLDER
from ..strings import functions
import ast


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
            "sum":"sum",
            "sort":"np.sort",
            "reverse":"np.flip",
            "min":"np.min",
            "max":"max",
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
            'squeeze':'np.squeeze',

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
            'mode': 'mode',
            'variance': 'np.var',
            'std': 'np.std', 
            'percentile': 'np.percentile',
            'random': 'np.random',
            'bincount': 'np.bincount',
            'where': 'where',
            #'sign': Operators.SIGN,  
            'foreach': 'foreach',
            'set_value': 'set_value',
            'clip': 'clip',
            'random_uniform': 'np.random.uniform',
            'prod': 'np.prod',
            'flatten': 'flatten',
            'ravel': 'np.ravel',

            'concat': 'concatenate',
            'cube': 'np.cbrt'
    }


def compute_locally_bm(*args, **kwargs):
    operator = kwargs.get("operator", None)
    op_type = kwargs.get("op_type", None)
    param_args =kwargs.get("params",None)
    # print("Operator", operator,"Op Type:",op_type)
    if op_type == "unary":
        value1 = args[0]
        t1=time.time()
        params=""
        if param_args is not None:
            params=[op_param_mapping[operator][str(_)] for _ in param_args ]
        # print(params)
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
def compute_locally(payload, subgraph_id, graph_id):
    try:
        # print("Computing ",payload["operator"])
        # print('\n\nPAYLOAD: ',payload)

        values = []


        for i in range(len(payload["values"])):
            if "value" in payload["values"][i].keys():
                # print("From server")
                if "path" not in payload["values"][i].keys():
                    values.append(payload["values"][i]["value"])

                else:
                    server_file_path = payload["values"][i]["path"]

                    download_path = os.path.join(FTP_DOWNLOAD_FILES_FOLDER,os.path.basename(payload["values"][i]["path"]))

                    # try:
                    g.ftp_client.download(download_path, os.path.basename(server_file_path))
                    value = load_data(download_path).tolist()
                    print('Loaded Data Value: ',value)
                    values.append(value)

                    # except Exception as error:
                    #     print('Error: ', error)
                    #     emit_error(payload, error, subgraph_id, graph_id)

                    if os.path.basename(server_file_path) not in g.delete_files_list and payload["values"][i]["to_delete"] == 'True':
                        g.delete_files_list.append(os.path.basename(server_file_path))

                    if os.path.exists(download_path):
                        os.remove(download_path)

            elif "op_id" in payload["values"][i].keys():
                # print("From client")
                # try:
                values.append(g.outputs[payload['values'][i]['op_id']])
                # except Exception as e:
                #     emit_error(payload,e, subgraph_id, graph_id)

        payload["values"] = values

        # print("Payload Values: ", payload["values"])

        op_type = payload["op_type"]
        operator = payload["operator"]
        params=payload['params']
        param_string=""
        for i in params.keys():
            if type(params[i]) == str:
                param_string+=","+i+"=\'"+str(params[i])+"\'"
            else:
                param_string+=","+i+"="+str(params[i])


        # try:
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

        if not isinstance(result, np.ndarray):
            result = np.array(result)

        result_byte_size = result.size * result.itemsize

        if result_byte_size < (30 * 1000000)//10000:
            try:
                result = result.tolist()
            except:
                result = result

            g.outputs[payload["op_id"]] = result

            return json.dumps({
            'op_type': payload["op_type"],
            'result': result,
            # 'username': g.cid,
            # 'token': g.ravenverse_token,
            'operator': payload["operator"],
            "op_id": payload["op_id"],
            "status": "success"
            })

        else:

            file_path = upload_result(payload, result)

            g.outputs[payload["op_id"]] = result.tolist()

            # op = g.ops[payload["op_id"]]
            # op["status"] = "success"
            # op["endTime"] = int(time.time() * 1000)
            # g.ops[payload["op_id"]] = op

            return json.dumps({
                'op_type': payload["op_type"],
                'file_name': os.path.basename(file_path),
                # 'username': g.cid,
                # 'token': g.ravenverse_token,
                'operator': payload["operator"],
                "op_id": payload["op_id"],
                "status": "success"
            })

    except Exception as error:
        print('Error: ', error)
        if 'broken pipe' in str(error).lower():
            print('\n\nYou have encountered an IO based Broken Pipe Error. \nRestart terminal and try connecting again')
            sys.exit()

        emit_error(payload, error, subgraph_id, graph_id)


def upload_result(payload, result):
    result_size = result.size * result.itemsize
    try:
        result = result.tolist()
    except:
        result=result
    
    # print("Emit Success")

    file_path = dump_data(payload['op_id'],result)
    g.ftp_client.upload(file_path, os.path.basename(file_path))
    
    print("\nFile uploaded!", file_path, ' Size: ', result_size)
    os.remove(file_path)
  
    return file_path
    

def emit_error(payload, error, subgraph_id, graph_id):
    print("Emit Error")
    # print(payload)
    print(error)
    g.error = True
    error=str(error)
    client = g.client
    print(error,payload)
    client.emit("op_completed", json.dumps({
            'op_type': payload["op_type"],
            'error': error,
            'operator': payload["operator"],
            "op_id": payload["op_id"],
            "status": "failure",
            "subgraph_id": subgraph_id,
            "graph_id": graph_id
    }), namespace="/client")

    # op = g.ops[payload["op_id"]]
    # op["status"] = "failure"
    # op["endTime"] = int(time.time() * 1000)
    # g.ops[payload["op_id"]] = op

    try:
        for ftp_file in g.delete_files_list:
            g.ftp_client.delete_file(ftp_file)
    except Exception as e:

        g.delete_files_list = []
        g.outputs = {}
        g.has_subgraph = False

    g.delete_files_list = []
    g.outputs = {}
    g.has_subgraph = False


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
        result=np.where(condition,a,b)
    return result

def clip(a,lower_limit=None,upper_limit=None):
    if lower_limit is None:
        raise Exception("lower limit is missing")
    elif upper_limit is None:
        raise Exception("upper limit is missing")
    else:
        result = np.clip(a,lower_limit,upper_limit)
    return result

def max(a,axis=None,keepdims=False):
    if str(keepdims) == 'True':
        keepdims = True
    else:
        keepdims = False
    result=np.max(a,axis=axis,keepdims=keepdims)
    return result

def sum(a,axis=None,keepdims=False):
    if str(keepdims) == 'True':
        keepdims = True
    else:
        keepdims = False
    result=np.sum(a,axis=axis,keepdims=keepdims)
    return result

def flatten(a):
    a = np.array(a)
    return a.flatten()

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


def foreach(val=None,**kwargs):
    operator=kwargs.get("operation")
    result=[]
    paramstr=""
    del kwargs['operation']
    print(kwargs)
    for _ in kwargs.keys():
        paramstr+=","+_+"="+str(kwargs.get(_))
    for i in val:
        evalexp="{}({}{})".format(numpy_functions[operator],i,paramstr)
        print("\n\nevaluating:",evalexp)
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

def mode(arr,axis=0):
    result=stats.mode(arr,axis=axis)
    return result.mode

def concatenate(*args,**kwargs):
    param_string=""
    for i in kwargs.keys():
        if type(params[i]) == str:
            param_string+=","+i+"=\'"+str(params[i])+"\'"
        else:
            param_string+=","+i+"="+str(params[i])
    result=eval("np.concatenate(args"+param_string+")")
    return result