import json
import os
import sys
import socket
import ast
import pickle as pkl
from terminaltables import AsciiTable
import zipfile
from .compute import compute_locally, emit_error
from ..config import FTP_DOWNLOAD_FILES_FOLDER, FTP_TEMP_FILES_FOLDER
from ..globals import g
from ..utils import setTimeout, stopTimer, dump_data, dump_torch_model, dump_result_data
import time
import subprocess as sp
import torch
import time
import psutil
import gc

timeoutId = g.timeoutId
opTimeout = g.opTimeout
initialTimeout = g.initialTimeout

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    # memory_free_values = [psutil.virtual_memory()[3]/1000000000]
    return memory_free_values

@g.client.on('subgraph_forward', namespace="/client")
async def compute_subgraph_forward(d):
    g.param_queue['param'] = d

async def compute_thread():
    while True:
        if g.param_queue.get('param',None) is not None:
            await subgraph_forward_process(g.param_queue['param'])
            del g.param_queue['param']
        time.sleep(1.5)

async def subgraph_forward_process(d):
    global timeoutId
    gc.collect()
    total_t = time.time()
    os.system('clear')
    print(AsciiTable([['Provider Dashboard']]).table)
    g.dashboard_data.append([d["subgraph_id"], d["graph_id"], "Computing"])
    print(AsciiTable(g.dashboard_data).table)

    g.has_subgraph = True
    subgraph_id = d["subgraph_id"]
    graph_id = d["graph_id"]
    data = d["payloads"]
    gpu_required = ast.literal_eval(d["gpu_required"])

    subgraph_outputs = d["subgraph_outputs_list"]
    persist_forward_pass_results_list = d["persist_forward_pass_results"]
    persist_model_list = d["persist_model_list"]

    subgraph_zip_file_flag = d["subgraph_zip_file_flag"]
    results = []
    g.error = False
    forward_computations = {}

    if subgraph_zip_file_flag == "True":
        server_file_path = 'zip_{}_{}.zip'.format(subgraph_id, graph_id)

        download_path = os.path.join(FTP_DOWNLOAD_FILES_FOLDER, server_file_path)
        filesize = d['zip_file_size']
        try:
            g.ftp_client.ftp.voidcmd('NOOP')
            g.ftp_client.download(download_path, os.path.basename(server_file_path), filesize)

        except Exception as error:
            os.system('clear')
            g.dashboard_data[-1][2] = "Failed"
            print(AsciiTable([['Provider Dashboard']]).table)
            print(AsciiTable(g.dashboard_data).table)
            g.has_subgraph = False
            
            delete_dir = FTP_DOWNLOAD_FILES_FOLDER
            for f in os.listdir(delete_dir):
                os.remove(os.path.join(delete_dir, f))

            g.delete_files_list = []

            print('Error: ', error)
            emit_error(data[0], error, subgraph_id, graph_id)
            if 'broken pipe' in str(error).lower() or '421' in str(error).lower():
                print(
                    '\n\nYou have encountered an IO based Broken Pipe Error. \nRestart terminal and try connecting again')
                sys.exit()

        if os.path.exists(download_path):
            extract_to_path = FTP_DOWNLOAD_FILES_FOLDER
            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to_path)

            os.remove(download_path)
            g.ftp_client.delete_file(server_file_path)

    for index in data:
        if index['op_id'] in subgraph_outputs:
            to_upload = True
        else:
            to_upload = False
        operation_type = index["op_type"]
        operator = index["operator"]
        t1 = time.time()
        if operator == "start_backward_marker":
            marker_params = index["params"]
            step = ast.literal_eval(marker_params.get("step", "True"))
            for input_value in index["values"]:
                if "op_id" in input_value:
                    forward_computations[input_value['op_id']].backward()
                    if input_value['op_id'] in subgraph_outputs:
                        detached_loss = forward_computations[input_value['op_id']].detach()
                        forward_computations[input_value['op_id']] = detached_loss
                    else:
                        del forward_computations[input_value['op_id']]

            for key in forward_computations.keys():
                if isinstance(forward_computations[key], dict):
                    if forward_computations[key].get('result', None) is not None:
                        
                        # del g.forward_computations[key]['result']
                        if key in persist_forward_pass_results_list:
                            forward_computations[key]['result'] = forward_computations[key]['result'].detach()
                        else:
                            forward_computations[key]['result'] = forward_computations[key]['result'].detach()
                            forward_computations[key]['result'] = None

            if step:
                for key in forward_computations.keys():
                    # op_result = g.forward_computations[key]
                    if isinstance(forward_computations[key], dict):
                        # op_optimizer = g.forward_computations[key].get('optimizer', None)
                        if forward_computations[key].get('optimizer', None) is not None:                            
                            forward_computations[key]['optimizer'].step()
                            forward_computations[key]['optimizer'].zero_grad()

                            # g.forward_computations[key]['optimizer'] = op_optimizer

            results.append(json.dumps({
                'operator': operator,
                'status': 'success',
                'op_id': index['op_id']
            }))
            continue

        if operation_type is not None and operator is not None:
            result_payload, forward_computations = compute_locally(payload=index, subgraph_id=subgraph_id, graph_id=graph_id, forward_computations=forward_computations, to_upload=to_upload, gpu_required = gpu_required)
            if not g.error:
                if result_payload is not None:
                    results.append(result_payload)
            else:
                break
    
        t2 = time.time()
        # print("Time taken for operation: ", t2-t1, ' operator: ', operator)
        
    if not g.error:
        optimized_results_list = []
        for index in data:
            if index['op_id'] in subgraph_outputs:
                to_upload = True
            else:
                to_upload = False
            if to_upload:
                results_dict = {}
                previous_instance_dict = index['params'].get('previous_forward_pass', None)
                if previous_instance_dict is not None and 'op_id' in previous_instance_dict.keys():
                    previous_instance_id = previous_instance_dict['op_id']
                    if forward_computations[previous_instance_id].get('instance', None) is not None:
                        results_dict['instance'] = forward_computations[previous_instance_id]['instance']
                    if forward_computations[previous_instance_id].get('optimizer', None) is not None:
                        results_dict['optimizer'] = forward_computations[previous_instance_id]['optimizer']

                    results_dict['result'] = forward_computations[index['op_id']]['result']

                else:
                    results_dict = forward_computations[index['op_id']]

                persisted_result_path = None

                if index['operator'] == 'forward_pass':
                    if index['op_id'] in persist_model_list:
                        model_path = dump_torch_model(index['op_id'], results_dict['instance'].to('cpu'))
                    else:
                        model_path = dump_torch_model(index['op_id'], results_dict['instance'])
                    del results_dict['instance']
                    if index['op_id'] not in persist_forward_pass_results_list:
                        if results_dict.get('result', None) is not None:
                            del results_dict['result']
                    else:
                        if results_dict.get('result', None) is not None:
                            persisted_result = results_dict['result'].to('cpu')
                            persisted_result_path = dump_result_data(index['op_id'], persisted_result)

                    file_path = dump_data(index['op_id'], results_dict)
                    
                    if persisted_result_path is not None:
                        persisted_result_path = os.path.basename(persisted_result_path)

                    dumped_result = json.dumps({
                        'model_file_name': os.path.basename(model_path),
                        'file_name': os.path.basename(file_path),
                        'persisted_result_file_name': persisted_result_path,
                        "op_id": index['op_id'],
                        "status": "success"
                    })
                    optimized_results_list.append(dumped_result) 
                else:
                    if 'forward_pass' in index["operator"]:
                        if index['op_id'] not in persist_forward_pass_results_list:
                            if results_dict.get('result', None) is not None:
                                del results_dict['result']
                        else:
                            if results_dict.get('result', None) is not None:
                                persisted_result = results_dict['result'].to('cpu')
                                persisted_result_path = dump_result_data(index['op_id'], persisted_result)
                    
                    if isinstance(results_dict, torch.Tensor):
                        results_dict = results_dict.to('cpu')

                    file_path = dump_data(index['op_id'], results_dict)
                    if persisted_result_path is not None:
                        persisted_result_path = os.path.basename(persisted_result_path)

                    dumped_result = json.dumps({
                        'file_name': os.path.basename(file_path),
                        'persisted_result_file_name': persisted_result_path,
                        "op_id": index['op_id'],
                        "status": "success"
                    })
                    optimized_results_list.append(dumped_result)

        optimized_results_list.extend(results) 
        results = optimized_results_list       
        
        for temp_file in os.listdir(FTP_TEMP_FILES_FOLDER):
            if 'temp_' in temp_file:
                file_path = os.path.join(FTP_TEMP_FILES_FOLDER, temp_file)

                try:
                    with zipfile.ZipFile('local_{}_{}.zip'.format(subgraph_id, graph_id), 'a') as zipObj2:
                        zipObj2.write(file_path, os.path.basename(file_path))
                except zipfile.BadZipFile as error:
                    print(error)
                os.remove(file_path)

        zip_file_name = 'local_{}_{}.zip'.format(subgraph_id, graph_id)
        if os.path.exists(zip_file_name):
            g.ftp_client.upload(zip_file_name, zip_file_name)
            os.remove(zip_file_name)

        emit_result_data = {"subgraph_id": d["subgraph_id"], 
                            "graph_id": d["graph_id"], 
                            "token": g.ravenverse_token,
                            "results": results}
        await g.client.emit("forward_subgraph_completed", json.dumps(emit_result_data), namespace="/client")

        os.system('clear')
        g.dashboard_data[-1][2] = "Computed"
        print(AsciiTable([['Provider Dashboard']]).table)
        print(AsciiTable(g.dashboard_data).table)

    g.has_subgraph = False
    
    delete_dir = FTP_DOWNLOAD_FILES_FOLDER
    for f in os.listdir(delete_dir):
        os.remove(os.path.join(delete_dir, f))

    g.delete_files_list = []
    model_key, model_k = None, None
    for key, val in forward_computations.items():
        if isinstance(val, torch.Tensor):
            val.detach()
        elif isinstance(val, dict):
            for k, v in val.items():
                if isinstance(v, torch.Tensor):
                    v.detach()
                elif isinstance(v, torch.nn.Module):
                    model_key = key
                    model_k = k
        
    if model_key is not None and model_k is not None:
        del forward_computations[model_key][model_k]

    forward_computations = {}

    if gpu_required:
        torch.cuda.empty_cache()
    gc.collect()
    return

@g.client.on('redundant_subgraph', namespace="/client")
async def redundant_subgraph(d):
    subgraph_id = d['subgraph_id']
    graph_id = d['graph_id']
    for i in range(len(g.dashboard_data)):
        if g.dashboard_data[i][0] == subgraph_id and g.dashboard_data[i][1] == graph_id:
            g.dashboard_data[i][2] = "redundant_computation"
    os.system('clear')
    print(AsciiTable([['Provider Dashboard']]).table)
    print(AsciiTable(g.dashboard_data).table)
    return

@g.client.on('share_completed', namespace="/client")
async def share_completed(d):
    print("You have computed your share of subgraphs for this Graph, disconnecting...")
    await exit_handler()
    os._exit(1)


def waitInterval():
    global timeoutId, opTimeout, initialTimeout
    try:
        sock = socket.create_connection(('8.8.8.8',53))
        sock.close()
    except Exception as e:
        print('\n ----------- Device offline -----------')
        os._exit(1)

    if g.client.connected:
        stopTimer(timeoutId)
        timeoutId = setTimeout(waitInterval, opTimeout)

    if not g.is_downloading:
        if not g.is_uploading:
            if g.noop_counter % 17 == 0:
                try:
                    g.ftp_client.ftp.voidcmd('NOOP')

                except Exception as e:
                    print('\n Crashing...')
                    exit_handler()
                    os._exit(1)

    g.noop_counter += 1

async def exit_handler():
    g.logger.debug('Application is Closing!')
    if g.client is not None:
        g.logger.debug("Disconnecting...")
        if g.client.connected:
            await g.client.emit("disconnect", namespace="/client")

    dir = FTP_TEMP_FILES_FOLDER
    if os.path.exists(dir):
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))