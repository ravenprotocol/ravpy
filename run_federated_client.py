import os

os.environ['RAVENVERSE_HOST'] = "https://server.ravenverse.ai"
os.environ['RAVENVERSE_PORT'] = "80"
os.environ['RAVENVERSE_FTP_HOST'] = "server.ravenverse.ai"

from ravpy.federated.participate import participate
from ravpy.initialize import initialize
from ravpy.utils import list_graphs

if __name__ == '__main__':
    initialize("eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNjU4MTgwNDM0LCJpYXQiOjE2NTc5NjQ0MzQsImp0aSI6ImQ5MWE3M2ZkZDk3ZDRlZjM4YWM1ZDI5MzhjOTAwNWZlIiwidXNlcl9pZCI6IjAwMjY3MDgwNTgifQ.KeYP3y0yQEtYf42LX7A-HbFcQvtIjUD2H04Z3OFVdeA")
    list_graphs(approach="federated")
    participate(graph_id=1, file_path="ravpy/data/data1.csv")
