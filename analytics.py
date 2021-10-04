import ast

import numpy as np
import socketio
import tenseal as ts

sio = socketio.Client()

data_silo = [26, 45, 67]
avg = sum(data_silo) / len(data_silo)

min_val = min(data_silo)
max_val = max(data_silo)

variance = sum((i - avg) ** 2 for i in data_silo) / len(data_silo)

objective = None
params = {'size': [len(data_silo)],
          'Average': [avg],
          'Minimum': [min_val],
          'Maximum': [max_val],
          'Variance': [variance]
          }

print(data_silo)
print(params)


def analyze_data(data):
    rank = len(np.array(data).shape)

    if rank == 0:
        return {"rank": rank, "dtype": np.array(data).dtype.__class__.__name__}
    elif rank == 1:
        return {"rank": rank, "max": max(data), "min": min(data), "dtype": np.array(data).dtype.__class__.__name__}
    else:
        return {"rank": rank, "dtype": np.array(data).dtype.__class__.__name__}


@sio.event
def connect():
    print('Connected to server')
    sio.start_background_task(transmit_params)


@sio.event
def disconnect():
    print('Disconnected from server')


ckks_context = None


def check_compatibility(data, obj):
    rules = ast.literal_eval(obj["rules"])
    props = analyze_data(data)

    if rules['lower-limit'] < props['min'] and rules['upper-limit'] > props['max']:
        return True
    else:
        return False


@sio.event(namespace='/analytics')
def receive_objective(obj=None):
    global objective

    if obj is not None:
        objective = obj
        print(objective)
        # Check compatibility here
        if check_compatibility(data=data_silo, obj=obj):
            if ckks_context is None:
                while ckks_context is None:
                    sio.emit('context_vector', {}, namespace='/analytics', callback=receive_context_vector)
                    sio.sleep(5)
            else:
                cal_params()
        else:
            print("Your data is not compatible")
    else:
        print("Obj is none")


@sio.event(namespace='/analytics')
def receive_context_vector(context=None):
    global ckks_context
    global objective
    ckks_context = ts.context_from(context)
    print('Received context vector:', ckks_context)

    if ckks_context is not None:
        cal_params()


def cal_params():
    global objective
    params = dict()

    if objective["operator"] == "mean":
        rank = len(np.array(data_silo).shape)

        if rank == 0:
            # TODO: Apply differential privacy
            params["mean"] = data_silo
        elif rank == 1:
            params["mean"] = sum(data_silo) / len(data_silo)

    print('Encrypting params...')
    print(params['mean'])
    params["objective_id"] = objective["id"]
    # bb = ts.ckks_tensor(ckks_context, [params['mean']]).serialize()
    # print(len(bb))
    sio.emit('receive_params', params, namespace='/analytics')
    sio.sleep(10)

    objective = None

    wait_for_objective()


def wait_for_objective():
    while objective is None:
        sio.emit('handshake', {}, namespace='/analytics', callback=receive_objective)
        sio.sleep(5)


@sio.event(namespace='/analytics')
def transmit_params():
    global objective
    print('Performing Handshake')

    wait_for_objective()

    # print('Encrypting params...')
    # print(len(ts.ckks_tensor(ckks_context, params['size']).serialize()))
    # # params['size'] = ts.ckks_tensor(ckks_context, params['size']).serialize()
    # # params['Average'] = ts.ckks_tensor(ckks_context, params['Average']).serialize()
    # # params['Minimum'] = ts.ckks_tensor(ckks_context, params['Minimum']).serialize()
    # # params['Maximum'] = ts.ckks_tensor(ckks_context, params['Maximum']).serialize()
    # # params['Variance'] = ts.ckks_tensor(ckks_context, params['Variance']).serialize()
    #
    # # print('Emitting', params)
    # bb = ts.ckks_tensor(ckks_context, params['size']).serialize()
    # print(len(bb))
    # sio.emit('receive_params', {"size": bb}, namespace='/analytics')
    # sio.sleep(5)
    # sio.emit('receive_params', {"Average": ts.ckks_tensor(ckks_context, params['Average']).serialize()}, namespace='/analytics')
    # sio.sleep(5)
    # sio.emit('receive_params', {"Minimum": ts.ckks_tensor(ckks_context, params['Minimum']).serialize()}, namespace='/analytics')
    # sio.sleep(5)
    # sio.emit('receive_params', {"Maximum": ts.ckks_tensor(ckks_context, params['Maximum']).serialize()}, namespace='/analytics')
    # sio.emit('receive_params', {"Variance": ts.ckks_tensor(ckks_context, params['Variance']).serialize()}, namespace='/analytics')


sio.connect('http://localhost:9999?client_name=analytics&client_id=8888888888', headers={'client_name': 'analytics', "client_id": "8888888888"}, namespaces=['/analytics'])
sio.wait()
