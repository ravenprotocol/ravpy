import torch
import numpy as np
from scipy import stats

# standard numpy functions
def np_negative(value, params=None):
    return np.negative(value)

def np_positive(value, params=None):
    return np.positive(value)

def np_add(value1, value2, params=None):
    key = params.get("key", None)

    if key is not None:
        if isinstance(value1, dict):
            value1 = np.array(value1[key["key"]])
        
        if isinstance(value2, dict):
            value2 = np.array(value2[key["key"]])

        return np.add(value1, value2)

    return np.add(value1, value2)

def np_subtract(value1, value2, params=None):
    return np.subtract(value1, value2)

def np_exp(value1, params=None):
    return np.exp(value1)

def np_log(value1, params=None):
    return np.log(value1)

def np_square(value1, params=None):
    return np.square(value1)

def np_power(value1, value2, params=None):
    return np.power(value1, value2)

def np_sqrt(value1, params=None):
    return np.sqrt(value1)

def np_cbrt(value1, params=None):
    return np.cbrt(value1)

def np_abs(value1, params=None):
    return np.abs(value1)

def np_sort(value1, params=None):
    axis = params.get('axis',-1)
    return np.sort(value1, axis=axis)

def np_flip(value1, params=None):
    axis=params.get('axis',None)
    return np.flip(value1, axis=axis)

def np_min(value1, params=None):
    axis=params.get('axis',None)
    keepdims=params.get('keepdims',False)
    return np.min(value1, axis=axis, keepdims=keepdims)

def np_argmax(value1, params=None):
    if isinstance(value1, dict):
        value1 = value1['result']
    if isinstance(value1, list) or isinstance(value1, np.ndarray):
        value1 = torch.tensor(value1)

    dim=params.get('axis',None)
    keepdims=params.get('keepdims',None)

    if keepdims:
        return value1.argmax(dim=dim, keepdim=keepdims)
    return value1.argmax(dim=dim)

def np_argmin(value1, params=None):
    axis=params.get('axis',None)
    keepdims=params.get('keepdims',None)
    if keepdims:
        return np.argmin(value1, axis=axis, keepdims=keepdims)
    return np.argmin(value1, axis=axis)

def np_transpose(value1, params=None):
    axes=params.get('axes',None)
    return np.transpose(value1, axes=axes)

def np_divide(value1, value2, params=None):
    return np.divide(value1, value2)

def np_multiply(value1, value2, params=None):
    return np.multiply(value1, value2)

def np_matmul(value1, value2, params=None):
    if isinstance(value1, dict):
        value1 = value1['result']
    if isinstance(value2, dict):
        value2 = value2['result']
    return np.matmul(value1, value2)

def np_dot(value1, value2, params=None):
    return np.dot(value1, value2)

def np_split(value1, value2, params=None):
    axis=params.get('axis',0)
    return np.split(value1, value2, axis=axis)

def np_unique(value1, params=None):
    axis = params.get('axis',None)
    return_inverse = params.get('return_inverse',False)
    return_counts = params.get('return_counts',False)
    return np.unique(value1, axis=axis, return_inverse=return_inverse, return_counts=return_counts)

def np_linalg_inv(value1, params=None):
    return np.linalg.inv(value1)

def np_stack(value1, params=None):
    axis=params.get('axis',0)
    return np.stack(value1, axis=axis)

def np_tile(value1, value2, params=None):
    return np.tile(value1, value2)

def np_squeeze(value1, params=None):
    axis=params.get('axis',None)
    return np.squeeze(value1, axis=axis)

def np_greater(value1, value2, params=None):
    return np.greater(value1, value2)

def np_greater_equal(value1, value2, params=None):
    return np.greater_equal(value1, value2)

def np_less(value1, value2, params=None):
    return np.less(value1, value2)

def np_less_equal(value1, value2, params=None):
    return np.less_equal(value1, value2)

def np_equal(value1, value2, params=None):
    return np.equal(value1, value2)

def np_not_equal(value1, value2, params=None):
    return np.not_equal(value1, value2)

def np_logical_and(value1, value2, params=None):
    return np.logical_and(value1, value2)

def np_logical_or(value1, value2, params=None):
    return np.logical_or(value1, value2)

def np_logical_not(value1, params=None):
    return np.logical_not(value1)

def np_logical_xor(value1, value2, params=None):
    return np.logical_xor(value1, value2)

def np_average(value1, params=None):
    axis=params.get('axis',None)
    weights=params.get('weights',None)
    return np.average(value1, axis=axis, weights=weights)

def np_std(value1, params=None):
    axis=params.get('axis',None)
    ddof=params.get('ddof',0)
    return np.std(value1, axis=axis, ddof=ddof)

def np_percentile(value1, value2, params=None):
    axis=params.get('axis',None)
    keepdims=params.get('keepdims',False)
    method=params.get('method','linear')
    return np.percentile(value1, value2, axis=axis, keepdims=keepdims, method=method)

def np_bincount(value1, params=None):
    weights=params.get('weights',None)
    minlength=params.get('minlength',0)
    return np.bincount(value1, weights=weights, minlength=minlength)

def np_random_uniform(value1, value2, params=None):
    size=params.get('size',None)
    return np.random.uniform(value1, value2, size=size)

def np_prod(value1, params=None):
    axis=params.get('axis',None)
    keepdims=params.get('keepdims',None)
    if keepdims:
        return np.prod(value1, axis=axis, keepdims=keepdims)
    return np.prod(value1, axis=axis)

def np_ravel(value1, params=None):
    order=params.get('order','C')
    return np.ravel(value1, order=order)

def np_arange(value1, value2, params=None):
    step=params.get('step',1)
    return np.arange(value1, value2, step)

def np_zeros(value1, params=None):
    dtype=params.get('dtype',float)
    return np.zeros(value1, dtype=dtype)


#ops:
def ravslice(tensor,params=None):#begin=None,size=None):
    begin = params.get('begin', None)
    size = params.get('size', None)
    result=tensor[begin:begin+size]
    return result

def gather(tensor,indices, params=None):
    result=[]
    for i in indices:
        result.append(tensor[i])
    return result
    
def where(a,b,params=None):#condition=None):
    condition = params.get('condition', None)
    if condition is None:
        raise Exception("condition is missing")
    else:
        result=np.where(condition,a,b)
    return result

def clip(a,params=None):#lower_limit=None,upper_limit=None):
    lower_limit = params.get('lower_limit', None)
    upper_limit = params.get('upper_limit', None)
    if lower_limit is None:
        raise Exception("lower limit is missing")
    elif upper_limit is None:
        raise Exception("upper limit is missing")
    else:
        result = np.clip(a,lower_limit,upper_limit)
    return result

def max(a,params=None):#axis=None,keepdims=False):
    axis = params.get('axis', None)
    keepdims = params.get('keepdims', False)
    if str(keepdims) == 'True':
        keepdims = True
    else:
        keepdims = False
    if isinstance(axis,list):
        axis= tuple(axis)
    result=np.max(a,axis=axis,keepdims=keepdims)
    return result

def mean(a,params=None):#axis=None,keepdims=False):
    axis = params.get('axis', None)
    keepdims = params.get('keepdims', None)

    if str(keepdims) == 'True':
        keepdims = True
    else:
        keepdims = False

    if isinstance(axis,list):
        axis= tuple(axis)

    if keepdims:
        return np.mean(a,axis=axis,keepdims=keepdims)
    return np.mean(a,axis=axis)

def variance(a,params=None):#axis=None,keepdims=False):
    axis = params.get('axis', None)
    keepdims = params.get('keepdims', None)

    if str(keepdims) == 'True':
        keepdims = True
    else:
        keepdims = False

    if isinstance(axis,list):
        axis= tuple(axis)

    if keepdims:
        return np.var(a,axis=axis,keepdims=keepdims)
    return np.var(a,axis=axis)

def sum(a,params=None):#axis=None,keepdims=False):
    axis = params.get('axis', None)
    keepdims = params.get('keepdims', None)
    if str(keepdims) == 'True':
        keepdims = True
    else:
        keepdims = False
    if isinstance(axis,list):
        axis= tuple(axis)
    if keepdims:
        return np.sum(a,axis=axis,keepdims=keepdims)
    return np.sum(a,axis=axis)

def flatten(a, params=None):
    a = np.array(a)
    return a.flatten()

def split(arr,params=None):#numOrSizeSplits=None,axis=None):
    numOrSizeSplits = params.get('numOrSizeSplits', None)
    axis = params.get('axis', None)
    result=np.split(arr,numOrSizeSplits,axis=axis)
    return result

def expand_dims(arr,params=None):
    axis=params.get('axis', None)
    if axis is not None:
        result=np.expand_dims(arr,axis=axis)
    else:
        result= np.expand_dims(arr,axis=0)
    return result

def tile(arr,reps, params=None):
    if reps is not None:
        result=np.tile(arr,reps)
    else:
        raise Exception("reps is missing")
    return result
    

def one_hot_encoding(arr,depth, params=None):
    return np.squeeze(np.eye(depth)[arr.reshape(-1)])

def foreach(val,params=None):

    numpy_functions = {
            "neg": "np_negative",
            "pos": "np_positive",
            "add": "np_add",
            "sub": "np_subtract",
            "exp": "np_exp",
            "natlog": "np_log",
            "square":"np_square",
            "pow":"np_power",
            "square_root":"np_sqrt",
            "cube_root":"np_cbrt",
            "abs":"np_abs", 
            "sum":"sum", 
            "sort":"np_sort",
            "reverse":"np_flip",
            "min":"np_min",
            "max":"max",
            "argmax":"np_argmax",
            "argmin":"np_argmin",
            "transpose":"np_transpose",
            "div":"np_divide",
            'mul': 'np_multiply',
            'matmul': 'np_matmul',
            'multiply':'np_multiply',
            'dot': 'np_dot',
            'split': 'np_split', 
            'reshape':'reshape', 
            'unique': 'np_unique', 
            'expand_dims':'expand_dims', 
            'inv': 'np_linalg_inv', 
            'gather': 'gather', 
            'stack': 'np_stack', 
            'tile': 'np_tile', 
            'slice': 'ravslice',

            'find_indices': 'find_indices',
            'shape':'shape',
            'squeeze':'np_squeeze',
            'pad':'pad',
            'index':'index',

            #Comparision ops
            'greater': 'np_greater',
            'greater_equal':'np_greater_equal' ,
            'less': 'np_less',
            'less_equal':'np_less_equal' ,
            'equal':'np_equal' ,
            'not_equal': 'np_not_equal',
            'logical_and':'np_logical_and' ,
            'logical_or': 'np_logical_or',
            'logical_not': 'np_logical_not',
            'logical_xor': 'np_logical_xor',

            #statistics
            'mean': 'mean',
            'average': 'np_average',
            'mode': 'mode',
            'variance': 'variance',
            'std': 'np_std', 
            'percentile': 'np_percentile',
            'bincount': 'np_bincount',
            'where': 'where',
            #'sign': Operators.SIGN,  
            'foreach': 'foreach',
            'set_value': 'set_value',
            'clip': 'clip',
            'random_uniform': 'np_random_uniform',
            'prod': 'np_prod',
            'flatten': 'flatten',
            'ravel': 'np_ravel',

            'concat': 'concatenate',
            'cube': 'np_cbrt',
            'arange':'np_arange',
            'repeat':'repeat',
            'join_to_list': 'join_to_list',
            'combine_to_list': 'combine_to_list',
            'zeros':'np_zeros',
            'ravint':'ravint',
            'cnn_index':'cnn_index',
            'cnn_add_at':'cnn_add_at',
            'cnn_index_2':'cnn_index_2',
            'size': 'size',
            'one_hot_encoding': 'one_hot_encoding'            

    }

    operator=params.get("operation", None)
    result=[]
    paramstr=""
    del params['operation']
    for _ in params.keys():
        paramstr+=","+_+"="+str(params.get(_))
    for i in val:
        evalexp="{}({}{})".format(numpy_functions[operator],i,paramstr)
        print("\n\nevaluating:",evalexp)
        res=eval(evalexp)
        if type(res) is np.ndarray:
            result.append(res.tolist())
        else:
            result.append(res)
    return result

def find_indices(arr,val, params=None):
    result=[]
    
    for i in val:
        indices = [_ for _, arr in enumerate(arr) if arr == i]
        result.append(indices)
    if len(val) == 1:
        return indices
    else:
        return result

def reshape(X,params=None):#shape=None):
    shape = params.get('shape', None)
    
    if isinstance(X, dict):
        X = X['result']
    if isinstance(X, list) or isinstance(X, np.ndarray):
        X = torch.tensor(X)
    
    if shape is None:
        return None
    else:
        return X.view(*shape) #np.reshape(tens,newshape=shape)

def mode(arr,params=None):#axis=0):
    axis = params.get('axis', 0)
    result=stats.mode(arr,axis=axis)
    return result.mode

def concatenate(a,b,params=None):
    axis = params.get('axis', 0)

    if isinstance(a, dict):
        a = a['result']
    if isinstance(a, list) or isinstance(a, np.ndarray):
        a = torch.tensor(a)

    if isinstance(b, dict):
        b = b['result']
    if isinstance(b, list) or isinstance(b, np.ndarray):
        b = torch.tensor(b)

    result = torch.cat((a,b),dim=axis)

    return result

def shape(arr,params=None):#index=None):
    index = params.get('index', None)
    arr = np.array(arr)
    if index is None:
        return arr.shape
    else:
        return arr.shape[int(index)]

def pad(arr,params=None):#sequence=None,mode=None):
    sequence = params.get('sequence', None)
    mode = params.get('mode', None)
    if sequence is None:
        raise Exception("sequence param is missing")
    elif mode is None:
        raise Exception("mode is missing")
    arr = np.array(arr)
    result = np.pad(arr,sequence,mode=mode)
    return result

def repeat(arr,params=None):#repeats=None, axis=None):
    repeats = params.get('repeats', None)
    axis = params.get('axis', None)
    if repeats is None:
        raise Exception("repeats param is missing")

    arr = np.array(arr)
    result = np.repeat(arr,repeats=repeats,axis=axis)
    return result

def index(arr,params=None):#indices=None):
    indices = params.get('indices', None)

    if isinstance(arr, dict):
        arr = arr['result']
    if isinstance(arr, list) or isinstance(arr, np.ndarray):
        arr = torch.tensor(arr)

    print("\n\nindex arr: ",arr, arr.shape)

    if indices is None:
        raise Exception("indices param is missing")
    indices = indices['indices']
    if isinstance(indices, str):
        # arr = np.array(arr)
        result = eval("arr"+indices)
    else:
        result = eval("arr[{}]".format(tuple(indices)))
    return result

def join_to_list(a,b, params=None):
    a = np.array(a)
    result = np.append(a,b)
    return result

def combine_to_list(a,b, params=None):    
    result = np.array([a,b])
    return result

def ravint(a, params=None):
    return int(a)

def cnn_index(arr,params=None):#index1=None,index2=None,index3=None):
    index1 = params.get('index1', None)
    index2 = params.get('index2', None)
    index3 = params.get('index3', None)
    if index1 is None or index2 is None or index3 is None:
        raise Exception("index1, index2 or index3 param is missing")
    
    result = eval("np.array(arr)"+"[:,{},{},{}]".format(index1,index2,index3))
    return result

def cnn_index_2(a,params=None):#pad_h=None, height=None, pad_w=None, width=None):
    pad_h = params.get('pad_h', None)
    height = params.get('height', None)
    pad_w = params.get('pad_w', None)
    width = params.get('width', None)
    if pad_h is None or height is None or pad_w is None or width is None:
        raise Exception("index1, index2 or index3 param is missing")

    a = np.array(a)
    result = a[:, :, pad_h:height+pad_h, pad_w:width+pad_w]
    return result

def cnn_add_at(a, b, params=None):#index1=None,index2=None,index3=None):
    index1 = params.get('index1', None)
    index2 = params.get('index2', None)
    index3 = params.get('index3', None)
    if index1 is None or index2 is None or index3 is None:
        raise Exception("index1, index2 or index3 param is missing")
    
    a = np.array(a)
    b = np.array(b)
    index1 = np.array(index1)
    index2 = np.array(index2)
    index3 = np.array(index3)

    np.add.at(a, (slice(None), index1, index2, index3), b)
    return a

def set_value(a,b,params=None):#indices=None):
    indices = params.get('indices', None)
    if indices is None:
        raise Exception("indices param is missing")
    if isinstance(indices, str):
        exec("a"+indices+'='+'b')
    else:
        print("\n\n Indices in set value: ", indices)
        a = np.array(a)
        a[tuple(indices)] = b
    return a

def size(a, params=None):
    a = np.array(a)
    return a.size
