import os

from ..globals import g
from ..utils import initialize_ftp_client


def participate():
    # Initialize and create FTP client
    res = initialize_ftp_client()
    if res is None:
        os._exit(1)

    from .benchmarking import benchmark
    benchmark()

    g.logger.debug("")
    g.logger.debug("Ravpy is waiting for ops and subgraphs...")
    g.logger.debug("Warning: Do not close this terminal if you like to "
                   "keep participating and keep earning Raven tokens\n")
