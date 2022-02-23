const socket_server_url = 'ws://' + RAVSOCK_SERVER_URL + ':' + RAVSOCK_SERVER_PORT + '/' + CLIENT_TYPE;

socket = io(socket_server_url, {
    query: {
        "type": CLIENT_TYPE,
        "cid": 4
    }
});

// On connection started
socket.on('connect', function (d) {
    console.log("Connected");
});

// On connection closed
socket.on("disconnect", function (d) {
    console.log("Disconnected");
});

// Check if the client is connected
socket.on("check_status", function (d) {
    socket.emit("check_callback", d);
});

// To check if this client is still connected or not
socket.on("ping", function (message) {
    console.log(message);
    console.log("Received PING");

    console.log("Sending PONG");
    socket.emit("pong", JSON.stringify({
        "message": "PONG"
    }));
});
