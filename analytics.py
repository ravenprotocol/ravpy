import socketio
import random

sio = socketio.Client()

data_silo = [random.randint(1,100) for _ in range(random.randint(1,10))]
avg = sum(data_silo)/len(data_silo)

min_val = min(data_silo)
max_val = max(data_silo)

variance = sum((i - avg) ** 2 for i in data_silo) / len(data_silo)

params = {  'size': len(data_silo),
            'Average':avg,
            'Minimum':min_val,
            'Maximum':max_val,
            'Variance':variance
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

@sio.event(namespace='/analytics')
def transmit_params():
    for i in range(2):
        sio.emit('fed_analytics', params, namespace='/analytics')
        sio.sleep(5)
        
sio.connect('http://localhost:9999', headers={'client_name':'analytics'}, namespaces=['/analytics'])
sio.wait()