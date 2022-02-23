let $ = require('jquery');
let {PythonShell} = require('python-shell');
let {Dropzone} = require("dropzone");

const RAVSOCK_SERVER_URL = "localhost";
//const RAVSOCK_SERVER_URL = "host.docker.internal";
const RAVSOCK_SERVER_PORT = "9999";
const CLIENT_TYPE = "client";
const GET_GRAPHS_URL = 'http://' + RAVSOCK_SERVER_URL + ':' + RAVSOCK_SERVER_PORT + '/graph/get/all/';
let graphs = null;
let clientId = null;

Dropzone.autoDiscover = false;

Dropzone.options.myGreatDropzone = { // camelized version of the `id`
    paramName: "file", // The name that will be used to transfer the file
    maxFilesize: 10, // MB
    uploadMultiple: false,
    parallelUploads: 1,
    maxFiles: 1,
    accept: function (file, done) {
        console.log("File added:", file, done);

        let options = {
            mode: 'text',
            args: ["--action", "list"]
        };

        PythonShell.run('../run.py', options, function (err, results) {
            if (err) throw err;
            console.log('results', results);
        });
    },
    autoProcessQueue: false,
    acceptedFiles: "text/csv",
    addRemoveLinks: true
};

const myDropzone = new Dropzone("form.dropzone", {url: "/file/post"});

hideAll();

if (localStorage.getItem("clientId") !== null) {
    login(localStorage.getItem("clientId"));
    showGraphs();
} else {
    $("#clientIdForm").show();
    $("#loggedInNav").fadeOut(5, function () {
        $("#loggedOutNav").fadeIn();
    });
}

function showGraphs() {
    $("#graphsCard").fadeOut(5);
    $("#spinnerGraphs").fadeIn(5);

    fetch(GET_GRAPHS_URL).then(r => r.json()).then(r => {
        console.log("Graphs:", r);
        graphs = r;
        $("#spinnerGraphs").fadeOut(5, function () {
            $("#graphsTable tbody").empty();
            for (let i = 0; i < r.length; i++) {
                let graph_data = r[i];
                console.log(r[i]);
                let algorithm = graph_data.algorithm;
                if (algorithm === null) {
                    algorithm = "NA";
                }
                let rules = "";
                if (graph_data.rules !== null) {
                    rules = formatRules(JSON.parse(graph_data.rules).rules);
                } else {
                    rules = "NA";
                }

                $("#graphsTable tbody").append("<tr><td>" + graph_data.id + "</td><td>" + graph_data.name + "</td><td>" + algorithm + "</td><td>" + graph_data.approach + "</td><td style='width: 30%'>" + rules + "</td><td><button type='button'" +
                    " class='btn btn-outline-success participateButton' data-id='" + r[i].id + "'>Participate</button></td></tr>");
            }
            $("#graphsTable").fadeIn();
            $("#graphsCard").fadeIn();
        });
    }).catch(error => {
        $("#spinnerGraphs").fadeOut(5, function () {
            $("#showErrorMessage").fadeIn();
            $("#graphsCard").fadeIn();
        });
    });
}

function formatRules(rules) {
    console.log(rules);
    let output = "<div>";
    for (const key in rules) {
        console.log(key, rules[key]);
        output = output + "<span style='font-weight:bold;text-transform: uppercase;'>" + key + "</span>: ";
        let rules1 = rules[key];
        for (const ruleKye in rules1) {
            output = output + ruleKye + "=" + rules1[ruleKye] + ", ";
        }
        output = output + "</br>";
    }
    output = output + "</div>";
    console.log(output);
    return output;
}


$(document).on('click', '.openButton', function () {
    var data_id = $(this).data("id");
    console.log(data_id);
});

$(document).on('click', '#clientIdButton', function () {
    clientId = $("#clientIdInput").val();
    if (clientId === "") {
        return;
    } else {
        login(clientId);
    }
    return false;
});

$(document).on('click', '#reloadModelsButton', function () {
    showGraphs();
});

function login(clientId) {
    if (clientId === null || clientId === undefined) {
        return;
    }
    localStorage.setItem("clientId", clientId);
    $("#clientIdInput").val("");
    $("#clientIdShow .clientIdValue").text(clientId);
    $("#clientIdForm").fadeOut(5, showGraphs);
    $("#loggedOutNav").fadeOut(5, function () {
        $("#loggedInNav").fadeIn();
        $("#activeModels").fadeIn();
    });
}

function logout() {
    localStorage.removeItem("clientId");
    $("#graphsCard").fadeOut(5, function () {
        hideAll();
        $("#loggedOutNav").fadeIn(5);
        $("#clientIdForm").fadeIn(5);
    });
}

$(document).on('click', '#logoutButton', function () {
    logout();
});

$(document).on('click', '#loginButton', function () {
    login();
});

$(document).on('click', '.participateButton', function () {
    var myModal = new bootstrap.Modal(document.getElementById('enterValuesModal'), {
        keyboard: false
    });
    myModal.show();
});

function hideAll() {
    $("#spinnerGraphs").hide();
    $("#loggedInNav").hide();
    $("#loggedOutNav").hide();
    $("#clientIdForm").hide();
    $("#graphsCard").hide();
    $("#activeModels").hide()
}
