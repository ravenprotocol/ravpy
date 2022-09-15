import atexit
import os

from .globals import g


def exit_handler():
    g.logger.debug('Application is Closing!')
    if g.client is not None:
        g.logger.debug("Disconnecting...")
        if g.client.connected:
            g.client.emit("disconnect", namespace="/client")


atexit.register(exit_handler)


def initialize(ravenverse_token):
    g.logger.debug("Initializing...")
    g.ravenverse_token = ravenverse_token

    client = g.client
    if client is None:
        g.client.disconnect()
        g.logger.error("Unable to connect to ravsock. Make sure you are using the right hostname and port")
        os._exit(1)
    else:
        g.logger.debug("Initialized successfully\n")
        return client
