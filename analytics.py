import socketio
import random

sio = socketio.Client()

data_silo = [random.randint(1,100) for _ in range(random.randint(1,10))]
avg = sum(data_silo)/len(data_silo)
print(avg)
params = {'Average':avg}

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