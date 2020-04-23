const tf = require("@tensorflow/tfjs");
require('@tensorflow/tfjs-node')


// Benutzung des Express-Frameworks
// Falls nicht installiert:
// npm install express

const express = require('express');
const app = express();
const fs = require("fs")

let model = null;
let minMax = []


function predictOccupancy(model, predictionInput) {

  var occupied = [0, 1]
  var tt = tf.tensor2d([predictionInput])
  var predictedValue = model.predict(tt).dataSync();
  return occupied[Math.round(predictedValue)]
}


// Lädt das Modell
// http://localhost:9000/check?measures=24.2, 23.7, 654.666666666667, 697.666666666667, 0.0044264636375791
// Wir gehen davon aus, die Werte sind schon den Parametern korrekt sortiert d.h: 
// Temperature, Humidity, Light, CO2 und HumidityRatio: 0.00447473826972372

app.get('/check', function (req, res) {

  var measures = req.query.measures.split(",");
 
  if (model != null) // 
  {
    // Denormalisierung der Werte

    measures.forEach((value, i, a) => {
      a[i] = (value - minMax[i].min) / (minMax[i].max - minMax[i].min);
    });


    var prediction = predictOccupancy(model, measures)
    console.log(`Prediction: ${prediction}`)
    res.end(JSON.stringify({
      "occupied": prediction
    }));

  } else
    res.end("Modell wurde nicht geladen :-(")

})

app.listen(9000, () => {
  console.log('Occupancy Server started on port 9000!');
});

/* Lädt die exportierten MinMax-Werte */
async function loadMinMax() {

  fs.readFile('minMax.json', (err, data) => {
    minMax = JSON.parse(data)
  });

}

/** Das exportierte Model wird geladen  */
async function loadOccupancyModel() {
  model = await tf.loadLayersModel('file://./export/model.json');
}

loadMinMax();
loadOccupancyModel().then(() => {
  console.log("Model wurde geladen!")
})