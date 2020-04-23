let monitored = {};
let chart = null;

const trainBeginSource = new EventSource("/subscribe/train/begin");
const epochEndSource = new EventSource('/subscribe/epoch/end/');

// Aufgerufen, wenn das Training startet
trainBeginSource.addEventListener("message", function (e) {
    if (0 < e.data.length) {
        var myModel = e.data;
        $("#modal_dialog").hide()
        // Der eigentliche Inhalt der Nachricht fängt an index 6 bzw. nach dem String  
        // geposteten Parameter model=
        myModel = myModel.toString().substring(6);
        //$("#modal_dialog").hide()
        try {
            myModel = JSON.parse(decodeURIComponent(myModel).replace(/\+/g, ' '))
            buildGraph(myModel); // in js/graph.js definiert
        } catch (e) {
            console.error("Error" + e); 
        }
    } else {
        $("#modal_dialog").show()
         // console.error("No Keras Training started")
    }
});

// Aufgerufen, wenn ein Epoch endet 
epochEndSource.addEventListener('message', function (e) {
    if (0 < e.data.length) {
       
        var data = JSON.parse(e.data);
        updateChart(data)
    } 
}, false);

// Update the charts
function updateChart(data) {

    if (chart === null) {

        // Wenn der Chart nicht initialisiert wurde,
        // wird dieser das erste Mal generiert

        for (key in data) {
            // if (key.toString().indexOf("val_") == -1)
                monitored[key] = [key, data[key]];
        }
        var columns = [];
        for (key in monitored) {
            columns.push(monitored[key]);
        }
        // C3 Chart
        chart = c3.generate({
            bindto: '#visualization',
            data: {
                x: 'epoch',
                columns: columns
            },
            axis: {
                y: {
                    label: {
                        text: "Wert",
                        position: 'outer-middle'
                    },
                    tick: {
                        format: d3.format('.2f')
                    }
                },
                x: {
                    label: {
                        text: "Epoch",
                        position: 'outer-middle'
                    },
                    tick: {
                        format: function (x) { return x+1; } // x + 1, Ansonsten wäre die Epoch-Zahl um 1 zu niedrig
                    }
                }
            }
        });
    } else {
        for (key in data) {
            if (key in monitored) {
                monitored[key].push(data[key]);
            }
            var columns = [];
            for (key in monitored) {
                columns.push(monitored[key]);
            }
            chart.load({
                columns: columns
            });
        }
    }
}

// Refresh Button
$("#refresh_button").on("click", () => {
    location.reload(true);
})
