import os

os.environ['RAVENVERSE_HOST'] = "0.0.0.0"
os.environ['RAVENVERSE_PORT'] = "9999"
os.environ['RAVENVERSE_FTP_HOST'] = "0.0.0.0"

from ravpy.federated.participate import participate
from ravpy.initialize import initialize
from ravpy.utils import list_graphs

if __name__ == '__main__':
    initialize("YOUR_TOKEN")
    list_graphs(approach="federated")
    participate(graph_id=1, file_path="ravpy/data/data1.csv")