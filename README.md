# raven-federated-client

## How to Run

Firstly, make sure Ravsock is up and running.

### 1. Run the Scheduler Script
```bash
$ python run_scheduler.py
```

### 2. Run Worker
```bash
$ python run_worker.py --action participate --federated_id <Enter graph_id>
```