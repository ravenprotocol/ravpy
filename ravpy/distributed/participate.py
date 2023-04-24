import os

from ..globals import g
from ..initialize import initialize
import asyncio
import threading

def participate(token=None,graph_id=None):
    from .evaluate import compute_thread

    def compute_callback():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(compute_thread())
        loop.close()
    
    _thread = threading.Thread(target=compute_callback)
    _thread.start()

    asyncio.run(initialize(token, graph_id))
