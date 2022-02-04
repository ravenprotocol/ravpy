from globals import g

client = g.client

while True:
    client.emit("scheduler", namespace="/client")

    client.sleep(5)
