import socketio
import random
import tenseal as ts

sio = socketio.Client()

data_silo = [random.randint(1,100) for _ in range(random.randint(1,10))]
avg = sum(data_silo)/len(data_silo)

min_val = min(data_silo)
max_val = max(data_silo)

variance = sum((i - avg) ** 2 for i in data_silo) / len(data_silo)

params = {  'size': [len(data_silo)],
            'Average':[avg],
            'Minimum':[min_val],
            'Maximum':[max_val],
            'Variance':[variance]
            }

print(data_silo)
print(params)

@sio.event
def connect():
    print('Connected to server')
    sio.start_background_task(transmit_params)

@sio.event
def disconnect():
    print('Disconnected from server')

ckks_context = None

@sio.event(namespace='/analytics')
def get_context_vector(context=None):
    global ckks_context
    ckks_context = context
    print('Received context vector:', ckks_context)

@sio.event(namespace='/analytics')
def transmit_params():
    global ckks_context
    print('Performing Handshake')
    while ckks_context is None:
        sio.emit('handshake', {}, namespace='/analytics', callback=get_context_vector)    
        sio.sleep(5)

    print('Encrypting params...')
    params['size'] = ts.ckks_tensor(ckks_context, params['size'])
    params['Average'] = ts.ckks_tensor(ckks_context, params['Average'])
    params['Minimum'] = ts.ckks_tensor(ckks_context, params['Minimum'])
    params['Maximum'] = ts.ckks_tensor(ckks_context, params['Maximum'])    
    params['Variance'] = ts.ckks_tensor(ckks_context, params['Variance'])
    
    print('Emitting')
    sio.emit('fed_analytics', params, namespace='/analytics')
    sio.sleep(5)
        
sio.connect('http://localhost:9999', headers={'client_name':'analytics'}, namespaces=['/analytics'])
sio.wait()