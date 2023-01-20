import os
import eel

from ..globals import g
from ..utils import initialize_ftp_client, disconnect


def participate():
    # Initialize and create FTP client
    res = initialize_ftp_client()
    if res is None:
        g.logger.error("quitting")
        disconnect()
    else:
        from .benchmarking import benchmark
        benchmark()

        g.logger.debug("")
        g.logger.debug("Ravpy is waiting for ops and subgraphs...")
        g.logger.debug("Warning: Do not close this terminal if you like to "
                       "keep participating and keep earning Raven tokens\n")
