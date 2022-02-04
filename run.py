from ftp_client import FTPClient
import socketio
import time

sio = socketio.Client()

client_status = {}
epoch = 0

ftp_client = FTPClient('54.201.212.222', 'Unni', '12345')
ftp_client.list_server_files()

@sio.event
def connect():
    print('Connected to server')
    sio.start_background_task(transmit_data)

@sio.event
def disconnect():
    print('Disconnected from server')

@sio.event(namespace='/raven-federated')
def server_payload(payload=None):
    global ftp_client
    if payload == None:
        print('Payload is None')

    elif payload["operator"] == 'federated_training':
        x = payload["values"][0];
        result = x
        print(type(result))
        
        print(result,{
            'op_type': payload["op_type"],
            'result': result,
            'values': payload["values"],
            'operator': payload["operator"],
            "op_id": payload["op_id"],
            "status": "success"})
        
        # ftp_client.download('model/global.h5','global.h5')
        ftp_client.upload('model/local.h5','collected_model/local{}.h5'.format(payload['op_id']))
        # ftp_client.close()
        time.sleep(20)
        sio.emit("op_completed", {
            'op_type': payload["op_type"],
            'result': result,
            'values': payload["values"],
            'operator': payload["operator"],
            "op_id": payload["op_id"],
            "status": "success"
        }, namespace='/raven-federated')
    
    
@sio.event(namespace='/raven-federated')
def transmit_data():
    global client_status,epoch
    while True:
        sio.emit('client_status', client_status, namespace='/raven-federated', callback=server_payload)
        sio.sleep(10)
    
sio.connect('http://localhost:9999', headers={'client_name':'raven-federated'}, namespaces=['/raven-federated'])
sio.wait()