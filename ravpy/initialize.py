import atexit
import os

from .globals import g


def exit_handler():
    g.logger.debug('Application is closing!')
    if g.client is not None:
        g.logger.debug("Disconnecting...")
        if g.client.connected:
            g.client.emit("disconnect", namespace="/client")


atexit.register(exit_handler)


def initialize(ravenverse_token):
    g.logger.debug("Initializing...")
    g.ravenverse_token = ravenverse_token

    g.connect_socket_client()
    client = g.client
    if not client.connected:
        g.client.disconnect()
        g.logger.error("Unable to connect to ravenverse. Make sure you are using the right hostname and port")
        return None
    else:
        g.logger.debug("Initialized successfully\n")
        return client
