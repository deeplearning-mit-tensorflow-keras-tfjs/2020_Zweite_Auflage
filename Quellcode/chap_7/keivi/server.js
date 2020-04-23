/**
 *  Node.js Keivi Server 
 */ 

// Benutzung des Express-Frameworks
// Falls nicht installiert:
// npm install express
const express = require('express');
const utf8 = require('utf8');
const app = express();

let keras_data = ""
let keras_model = "";

app.use(express.static('static'));

// Wird vom LambdaCallback am anfang des Trainings aufgerufen
app.post("/publish/train/begin",  (req, res)=> {
    var bodyStr = '';
    req.on("data", function (chunk) {
        bodyStr += chunk.toString();
    });
    req.on("end", function () {
        keras_model = bodyStr
        res.send(bodyStr);
    });
});


// Wird vom RemoteMonitor aufgerufen
app.post('/publish/epoch/end/', (req, res)=> {
    var bodyStr = '';
    req.on("data", function (chunk) {
        bodyStr += chunk.toString();
    });
    req.on("end", function () {
        keras_data = bodyStr
        res.send(bodyStr);
    });
});



// Wird von index.html abonniert: liefert das Keras Modell als JSON Struktur zurück
app.get("/subscribe/train/begin/", (req, res)=> {
    res.setHeader("Content-type", "text/event-stream");
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('Cache-Control', 'no-cache');
    res.send("data: " + keras_model + "\n\n");
});

// Wird von index.html abonniert: liefert die Werte von loss, acc und mse zurück
app.get('/subscribe/epoch/end/', (req, res)=> {
    res.setHeader("Content-type", "text/event-stream");
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('Cache-Control', 'no-cache');
    res.send("data: " + keras_data + "\n\n");
});

app.listen(9000, () => {
    console.log('🖥  KeiviServer started on port 9000!');
});