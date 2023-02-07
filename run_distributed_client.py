import os

from dotenv import load_dotenv
load_dotenv()

from ravpy import participate
from ravpy import initialize

if __name__ == '__main__':
    print(os.environ.get("TOKEN"))
    client = initialize(os.environ.get("TOKEN"))
    if not client.connected:
        os._exit(1)

    participate()
