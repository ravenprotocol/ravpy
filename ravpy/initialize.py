import atexit

from .globals import g


def exit_handler():
    g.logger.debug('Application is Closing!')
    if g.client is not None:
        g.logger.debug("Disconnecting...")
        if g.client.connected:
            g.client.emit("disconnect", namespace="/client")
    else:
        g.logger.debug("No Client Found.")


atexit.register(exit_handler)


def initialize(ravenverse_token):

    g.ravenverse_token = ravenverse_token
    '''Add Token Authorization code here.'''

    client = g.client
    if client is None:
        g.client.disconnect()
        raise Exception("Unable to connect to Ravsock. Make sure you are using the right hostname and port")

    return client
