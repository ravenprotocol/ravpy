import ast
import json
import os
import sys
import time

from terminaltables import AsciiTable
from ..config import FTP_DOWNLOAD_FILES_FOLDER
from ..globals import g
from ..strings import functions
from ..utils import get_key, load_data, load_data_raw
from .op_functions import *


def compute_locally_bm(*args, **kwargs):
    operator = kwargs.get("operator", None)
    op_type = kwargs.get("op_type", None)
    if op_type == "unary":
        value1 = args[0]
        params = {}
        t1 = time.time()
        bm_result = get_unary_result(value1['value'], params, operator)
        t2 = time.time()
        return t2 - t1

    elif op_type == "binary":
        value1 = args[0]['value']
        value2 = args[1]['value']
        params = {}
        t1 = time.time()
        bm_result = get_binary_result(value1, value2, params, operator)
        t2 = time.time()
        return t2 - t1


# async
def compute_locally(payload, subgraph_id, graph_id, to_upload=False):
    try:
        values = []
        for i in range(len(payload["values"])):
            if "value" in payload["values"][i].keys():
                if "path" not in payload["values"][i].keys():
                    values.append(torch.tensor(payload["values"][i]["value"]))

                else:
                    download_path = os.path.join(FTP_DOWNLOAD_FILES_FOLDER,
                                                    os.path.basename(payload["values"][i]["path"]))
                    value = load_data_raw(download_path)

                    if isinstance(value, dict):
                        value = value['result']

                    if isinstance(value, list) or isinstance(value, np.ndarray):
                        value = torch.tensor(value)

                    values.append(value)

            elif "op_id" in payload["values"][i].keys():
                values.append(g.forward_computations[payload['values'][i]['op_id']])
                
        payload["values"] = values

        op_type = payload["op_type"]
        operator = get_key(payload["operator"], functions)
        params = payload['params']
        instance = payload.get('instance', None)
        optimizer = None

        if instance is not None:
            download_path = os.path.join(FTP_DOWNLOAD_FILES_FOLDER, os.path.basename(instance))
            instance_dict = load_data_raw(download_path)
            instance = instance_dict.get('instance', None)
            optimizer = instance_dict.get('optimizer', None)

        params['instance'] = instance
        params['optimizer'] = optimizer

        params_dict = {}
        for i in params.keys():
            if i == "previous_forward_pass":
                if 'op_id' in params[i].keys():
                    previous_instance = g.forward_computations[params[i]['op_id']]
                    params_dict[i] = previous_instance
                elif 'value' in params[i].keys():
                    download_path = os.path.join(FTP_DOWNLOAD_FILES_FOLDER,
                                                    os.path.basename(params[i]['path']))
                    previous_instance = load_data_raw(download_path)
                    params_dict['previous_batch_layer_data'] = previous_instance
        
                continue

            if type(params[i]) == str:
                try:
                    temp = ast.literal_eval(params[i])
                    if type(temp) == dict or type(temp) == bool:
                        params_dict[i] = temp
                except:
                    params_dict[i] = params[i]
            elif type(params[i]) == dict:
                if 'op_id' in params[i].keys():
                    op_id = params[i]["op_id"]
                    param_value = g.forward_computations[op_id].numpy().tolist()
                elif 'value' in params[i].keys():
                    download_path = os.path.join(FTP_DOWNLOAD_FILES_FOLDER,
                                                    os.path.basename(params[i]["path"]))
                    param_value = load_data(download_path).tolist()
                
                params_dict[i] = param_value

        if op_type == "unary":
            val_1 = payload["values"][0]
            result = get_unary_result(val_1, params_dict, operator)

        elif op_type == "binary":
            val_1 = payload["values"][0]
            val_2 = payload["values"][1]
            result = get_binary_result(val_1, val_2, params_dict, operator)

        g.forward_computations[payload['op_id']] = result
                
        if not to_upload:
            return json.dumps({
                'op_type': payload["op_type"],
                'operator': payload["operator"],
                "op_id": payload["op_id"],
                "status": "success"
            })
        else:
            return None

    except Exception as error:
        os.system('clear')
        g.dashboard_data[-1][2] = "Failed"
        print(AsciiTable([['Provider Dashboard']]).table)
        print(AsciiTable(g.dashboard_data).table)
        emit_error(payload, error, subgraph_id, graph_id)
        if 'broken pipe' in str(error).lower() or '421' in str(error).lower():
            print('\n\nYou have encountered an IO based Broken Pipe Error. \nRestart terminal and try connecting again')
            sys.exit()


def  get_unary_result(value1, params, operator):
    if operator == "neg":
        result = np_negative(value1, params=params)
    elif operator == "pos":
        result = np_positive(value1, params=params)
    elif operator == "exp":
        result = np_exp(value1, params=params)
    elif operator == "natlog":
        result = np_log(value1, params=params)
    elif operator == "square":
        result = np_square(value1, params=params)
    elif operator == "square_root":
        result = np_sqrt(value1, params=params)
    elif operator == "cube_root":
        result = np_cbrt(value1, params=params)
    elif operator == "abs":
        result = np_abs(value1, params=params)
    elif operator == "sort":
        result = np_sort(value1, params=params)
    elif operator == "reverse":
        result = np_flip(value1, params=params)
    elif operator == "min":
        result = np_min(value1, params=params)
    elif operator == "argmax":
        result = np_argmax(value1, params=params)
    elif operator == "argmin":
        result = np_argmin(value1, params=params)
    elif operator == "transpose":
        result = np_transpose(value1, params=params)
    elif operator == "unique":
        result = np_unique(value1, params=params)
    elif operator == "inv":
        result = np_linalg_inv(value1, params=params)
    elif operator == "stack":
        result = np_stack(value1, params=params)
    elif operator == "squeeze":
        result = np_squeeze(value1, params=params)
    elif operator == "logical_not":
        result = np_logical_not(value1, params=params)
    elif operator == "average":
        result = np_average(value1, params=params)
    elif operator == "std":
        result = np_std(value1, params=params)
    elif operator == "bincount":
        result = np_bincount(value1, params=params)
    elif operator == "prod":
        result = np_prod(value1, params=params)
    elif operator == "ravel":
        result = np_ravel(value1, params=params)
    elif operator == "zeros":
        result = np_zeros(value1, params=params)


    elif operator == 'ravslice':
        result = ravslice(value1, params=params)
    elif operator == 'clip':
        result = clip(value1, params=params)
    elif operator == 'max':
        result = max(value1, params=params)
    elif operator == 'mean':
        result = mean(value1, params=params)
    elif operator == 'variance':
        result = variance(value1, params=params)
    elif operator == 'sum':
        result = sum(value1, params=params)
    elif operator == 'flatten':
        result = flatten(value1, params=params)
    elif operator == 'split':
        result = split(value1, params=params)
    elif operator == 'expand_dims':
        result = expand_dims(value1, params=params)
    elif operator == 'foreach':
        result = foreach(value1, params=params)
    elif operator == 'reshape':
        result = reshape(value1, params=params)
    elif operator == 'mode':
        result = mode(value1, params=params)
    elif operator == 'shape':
        result = shape(value1, params=params)
    elif operator == 'pad':
        result = pad(value1, params=params)
    elif operator == 'repeat':
        result = repeat(value1, params=params)
    elif operator == 'index':
        result = index(value1, params=params)
    elif operator == 'ravint':
        result = ravint(value1, params=params)
    elif operator == 'cnn_index':
        result = cnn_index(value1, params=params)
    elif operator == 'cnn_index_2':
        result = cnn_index_2(value1, params=params)
    elif operator == 'size':
        result = size(value1, params=params)

    # Machine Learning Algorithms
    elif operator == 'kmeans':
        result = kmeans(value1, params=params)

    # Deep Learning Layers
    elif operator == "forward_pass_dense":
        result = forward_pass_dense(value1, params=params)
    elif operator == "forward_pass_batchnorm1d":
        result = forward_pass_batchnorm1d(value1, params=params)
    elif operator == "forward_pass_batchnorm2d":
        result = forward_pass_batchnorm2d(value1, params=params)
    elif operator == "forward_pass_layernorm":
        result = forward_pass_layernorm(value1, params=params)
    elif operator == "forward_pass_dropout":
        result = forward_pass_dropout(value1, params=params)
    elif operator == "forward_pass_activation":
        result = forward_pass_activation(value1, params=params)
    elif operator == "forward_pass_conv2d":
        result = forward_pass_conv2d(value1, params=params)
    elif operator == "forward_pass_maxpool2d":
        result = forward_pass_maxpool2d(value1, params=params)
    elif operator == "forward_pass_flatten":
        result = forward_pass_flatten(value1, params=params)
    elif operator == "forward_pass_embedding":
        result = forward_pass_embedding(value1, params=params)
    elif operator == "forward_pass_reshape":
        result = forward_pass_reshape(value1, params=params)
    elif operator == "forward_pass_transpose":
        result = forward_pass_transpose(value1, params=params)
    elif operator == "forward_pass_power":
        result = forward_pass_power(value1, params= params)
    return result


def get_binary_result(value1, value2, params, operator):
    if operator == "add":
        result = np_add(value1, value2, params=params)
    elif operator == "sub":
        result = np_subtract(value1, value2, params=params)
    elif operator == "pow":
        result = np_power(value1, value2, params=params)
    elif operator == "div":
        result = np_divide(value1, value2, params=params)
    elif operator == "mul" or operator == "multiply":
        result = np_multiply(value1, value2, params=params)
    elif operator == "matmul":
        result = np_matmul(value1, value2, params=params)
    elif operator == "dot":
        result = np_dot(value1, value2, params=params)
    elif operator == "split":
        result = split(value1, value2, params=params)
    elif operator == "greater":
        result = np_greater(value1, value2, params=params)
    elif operator == "greater_equal":
        result = np_greater_equal(value1, value2, params=params)
    elif operator == "less":
        result = np_less(value1, value2, params=params)
    elif operator == "less_equal":
        result = np_less_equal(value1, value2, params=params)
    elif operator == "equal":
        result = np_equal(value1, value2, params=params)
    elif operator == "not_equal":
        result = np_not_equal(value1, value2, params=params)
    elif operator == "logical_and":
        result = np_logical_and(value1, value2, params=params)
    elif operator == "logical_or":
        result = np_logical_or(value1, value2, params=params)
    elif operator == "logical_xor":
        result = np_logical_xor(value1, value2, params=params)
    elif operator == "percentile":
        result = np_percentile(value1, value2, params=params)
    elif operator == "random_uniform":
        result = np_random_uniform(value1, value2, params=params)
    elif operator == "arange":
        result = np_arange(value1, value2, params=params)


    elif operator == 'gather':
        result = gather(value1, value2, params=params)
    elif operator == 'where':
        result = where(value1, value2, params=params)
    elif operator == 'tile':
        result = tile(value1, value2, params=params)
    elif operator == 'one_hot_encoding':
        result = one_hot_encoding(value1, value2, params=params)
    elif operator == 'find_indices':
        result = find_indices(value1, value2, params=params)
    elif operator == 'concat':
        result = concatenate(value1, value2, params=params)
    elif operator == 'join_to_list':
        result = join_to_list(value1, value2, params=params)
    elif operator == 'combine_to_list':
        result = combine_to_list(value1, value2, params=params)
    elif operator == 'cnn_add_at':
        result = cnn_add_at(value1, value2, params=params)
    elif operator == 'set_value':
        result = set_value(value1, value2, params=params)



    # Machine Learning Algorithms
    elif operator == "linear_regression":
        result = linear_regression(value1, value2, params=params)
    elif operator == "knn_classifier":
        result = knn_classifier(value1, value2, params=params)
    elif operator == "knn_regressor":
        result = knn_regressor(value1, value2, params=params)
    elif operator == "logistic_regression":
        result = logistic_regression(value1, value2, params=params)
    elif operator == "naive_bayes":
        result = naive_bayes(value1, value2, params=params)
    elif operator == "svm_svc":
        result = svm_svc(value1, value2, params=params)
    elif operator == "svm_svr":
        result = svm_svr(value1, value2, params=params)
    elif operator == "decision_tree_classifier":
        result = decision_tree_classifier(value1, value2, params=params)
    elif operator == "decision_tree_regressor":
        result = decision_tree_regressor(value1, value2, params=params)
    elif operator == "random_forest_classifier":
        result = random_forest_classifier(value1, value2, params=params)
    elif operator == "random_forest_regressor":
        result = random_forest_regressor(value1, value2, params=params)

    # Deep Learning Ops
    elif operator == "forward_pass_concat":
        result = forward_pass_concat(value1, value2, params=params)
    elif operator == "forward_pass_add":
        result = forward_pass_add(value1, value2, params=params)
    elif operator == "forward_pass_subtract":
        result = forward_pass_subtract(value1, value2, params=params)
    elif operator == "forward_pass_dot":
        result = forward_pass_dot(value1, value2, params=params)
    elif operator == "forward_pass_multiply":
        result = forward_pass_multiply(value1, value2, params=params)
    elif operator == "forward_pass_division":
        result = forward_pass_division(value1, value2, params=params)

    # Losses
    elif operator == 'square_loss':
        result = square_loss(value1, value2, params=params)
    elif operator == 'cross_entropy_loss':
        result = cross_entropy_loss(value1, value2, params=params)

    return result


def emit_error(payload, error, subgraph_id, graph_id):
    print("Emit Error")
    g.error = True
    error = str(error)
    client = g.client
    client.emit("error_handler", json.dumps({
        'op_type': payload["op_type"],
        'error': error,
        'operator': payload["operator"],
        "op_id": payload["op_id"],
        "status": "failure",
        "subgraph_id": subgraph_id,
        "graph_id": graph_id
    }), namespace="/client")

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
