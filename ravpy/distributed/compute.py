import ast
import json
import os
import sys
import time

from ..config import FTP_DOWNLOAD_FILES_FOLDER
from ..globals import g
from ..strings import functions
from ..utils import get_key, dump_data, load_data
from .op_functions import *


def compute_locally_bm(*args, **kwargs):
    operator = kwargs.get("operator", None)
    op_type = kwargs.get("op_type", None)
    param_args = kwargs.get("params", None)
    # print("Operator", operator,"Op Type:",op_type)
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
def compute_locally(payload, subgraph_id, graph_id):
    try:
        # print("Computing ",payload["operator"])
        # print('\n\nPAYLOAD: ',payload)

        values = []

        for i in range(len(payload["values"])):
            if "value" in payload["values"][i].keys():
                if "path" not in payload["values"][i].keys():
                    values.append(payload["values"][i]["value"])

                else:
                    download_path = os.path.join(FTP_DOWNLOAD_FILES_FOLDER,
                                                 os.path.basename(payload["values"][i]["path"]))
                    value = load_data(download_path).tolist()
                    values.append(value)

            elif "op_id" in payload["values"][i].keys():
                values.append(g.outputs[payload['values'][i]['op_id']])

        payload["values"] = values

        # print("Payload Values: ", payload)

        op_type = payload["op_type"]
        operator = get_key(payload["operator"], functions)
        params = payload['params']

        for i in params.keys():
            if type(params[i]) == str:
                temp = ast.literal_eval(params[i])
                if type(temp) == dict:
                    params[i] = temp
            elif type(params[i]) == dict and 'op_id' in params[i].keys():
                op_id = params[i]["op_id"]
                param_value = g.outputs[op_id]
                params[i] = param_value

        if op_type == "unary":
            result = get_unary_result(payload["values"][0], params, operator)
        elif op_type == "binary":
            result = get_binary_result(payload["values"][0], payload["values"][1], params, operator)

        if 'sklearn' in str(type(result)):
            file_path = upload_result(payload, result, subgraph_id=subgraph_id,
                                      graph_id=graph_id)  # upload_result(payload, result)
            return json.dumps({
                'op_type': payload["op_type"],
                'file_name': os.path.basename(file_path),
                'operator': payload["operator"],
                "op_id": payload["op_id"],
                "status": "success"
            })

        if 'dict' in str(type(result)):
            file_path = upload_result(payload, result, subgraph_id=subgraph_id,
                                      graph_id=graph_id)  # upload_result(payload, result)
            g.outputs[payload["op_id"]] = result

            return json.dumps({
                'op_type': payload["op_type"],
                'file_name': os.path.basename(file_path),
                'operator': payload["operator"],
                "op_id": payload["op_id"],
                "status": "success"
            })

        if not isinstance(result, np.ndarray):
            result = np.array(result)

        result_byte_size = result.size * result.itemsize

        if result_byte_size < (30 * 1000000) // 10000:
            try:
                result = result.tolist()
            except:
                result = result

            g.outputs[payload["op_id"]] = result

            return json.dumps({
                'op_type': payload["op_type"],
                'result': result,
                'operator': payload["operator"],
                "op_id": payload["op_id"],
                "status": "success"
            })

        else:

            file_path = upload_result(payload, result, subgraph_id=subgraph_id,
                                      graph_id=graph_id)  # upload_result(payload, result)
            g.outputs[payload["op_id"]] = result.tolist()

            return json.dumps({
                'op_type': payload["op_type"],
                'file_name': os.path.basename(file_path),
                'operator': payload["operator"],
                "op_id": payload["op_id"],
                "status": "success"
            })

    except Exception as error:
        print('Error: ', error)
        emit_error(payload, error, subgraph_id, graph_id)
        if 'broken pipe' in str(error).lower() or '421' in str(error).lower():
            print('\n\nYou have encountered an IO based Broken Pipe Error. \nRestart terminal and try connecting again')
            sys.exit()


def get_unary_result(value1, params, operator):
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
    elif operator == "backward_pass_dense":
        result = backward_pass_dense(value1, params=params)
    elif operator == "forward_pass_batchnorm":
        result = forward_pass_batchnorm(value1, params=params)
    elif operator == "backward_pass_batchnorm":
        result = backward_pass_batchnorm(value1, params=params)
    elif operator == "forward_pass_dropout":
        result = forward_pass_dropout(value1, params=params)
    elif operator == "backward_pass_dropout":
        result = backward_pass_dropout(value1, params=params)
    elif operator == "forward_pass_activation":
        result = forward_pass_activation(value1, params=params)
    elif operator == "backward_pass_activation":
        result = backward_pass_activation(value1, params=params)
    elif operator == "forward_pass_conv2d":
        result = forward_pass_conv2d(value1, params=params)
    elif operator == "backward_pass_conv2d":
        result = backward_pass_conv2d(value1, params=params)
    elif operator == "forward_pass_maxpool2d":
        result = forward_pass_maxpool2d(value1, params=params)
    elif operator == "backward_pass_maxpool2d":
        result = backward_pass_maxpool2d(value1, params=params)
    elif operator == "forward_pass_flatten":
        result = forward_pass_flatten(value1, params=params)
    elif operator == "backward_pass_flatten":
        result = backward_pass_flatten(value1, params=params)

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
    elif operator == 'concatenate':
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


    # Losses
    elif operator == 'square_loss':
        result = square_loss(value1, value2, params=params)
    elif operator == 'square_loss_gradient':
        result = square_loss_gradient(value1, value2, params=params)
    elif operator == 'cross_entropy_loss':
        result = cross_entropy_loss(value1, value2, params=params)
    elif operator == 'cross_entropy_gradient':
        result = cross_entropy_gradient(value1, value2, params=params)
    elif operator == 'cross_entropy_accuracy':
        result = cross_entropy_accuracy(value1, value2, params=params)

    return result


def upload_result(payload, result, subgraph_id=None, graph_id=None):
    try:
        result = result.tolist()
    except:
        result = result

    file_path = dump_data(payload['op_id'], result)

    from zipfile import ZipFile
    with ZipFile('local_{}_{}.zip'.format(subgraph_id, graph_id), 'a') as zipObj2:
        zipObj2.write(file_path, os.path.basename(file_path))

    os.remove(file_path)

    return file_path


def emit_error(payload, error, subgraph_id, graph_id):
    print("Emit Error")
    g.error = True
    error = str(error)
    client = g.client
    print(error, payload)
    client.emit("op_completed", json.dumps({
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
