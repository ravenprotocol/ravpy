import atexit

from .globals import g


def exit_handler():
    g.logger.debug('My application is ending!')
    if g.client is not None:
        g.logger.debug("disconnecting")
        if g.client.connected:
            g.client.emit("disconnect", namespace="/client")
    else:
        g.logger.debug("client is none")


atexit.register(exit_handler)


def initialize(ravenverse_token):

    g.ravenverse_token = ravenverse_token
    '''Add Token Authorization code here.'''

    client = g.client
    if client is None:
        g.client.disconnect()
        raise Exception("Unable to connect to ravsock. Make sure you are using the right hostname and port")

    return client
