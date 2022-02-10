# ravpy

## Raven's python client for distributed and federated learning

### How to Run

Firstly, make sure Ravsock is up and running.

#### 0. Set variables in the ravpy/config.py file

    # Specify hostname and port for ravsock
    SOCKET_SERVER_HOST = "localhost"
    SOCKET_SERVER_PORT = "9999"
    SOCKET_SERVER_URL = "http://{}:{}".format(SOCKET_SERVER_HOST, SOCKET_SERVER_PORT)
    
    # Set CID (CID is the client's unique id and cannot used by any other client)
    CID = "123"
    
    # Enable/disable encryption
    ENCRYPTION = True
    
#### 1. Fetch the list of federated analytics graphs

    $ python run.py --action list

#### 2. Join a federated analytics graph

    $ python run.py --action participate --federated_id <Enter graph_id>

#### 3. Enter data file path 

    $ Enter data file path: ravpy/data/data1.csv
    

## Sample data format:

    "age","salary","bonus","fund"
    51,3.5,1.4,.2
    49,3,1.4,.2
    
Kindly, wait for the process to terminate