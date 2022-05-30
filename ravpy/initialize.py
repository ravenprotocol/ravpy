from .globals import g

def initialize(ravenverse_token):

    g.ravenverse_token = ravenverse_token
    '''Add Token Authorization code here.'''

    client = g.client
    if client is None:
        g.client.disconnect()
        raise Exception("Unable to connect to ravsock. Make sure you are using the right hostname and port")
