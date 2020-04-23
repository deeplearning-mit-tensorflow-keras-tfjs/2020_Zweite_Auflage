/**
 * 
 * https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+
 * 
 * https://codelabs.developers.google.com/codelabs/tfjs-training-regression/index.html#2
 * https://www.researchgate.net/profile/Luis_Candanedo_Ibarra/publication/285627413_Accurate_occupancy_detection_of_an_office_room_from_light_temperature_humidity_and_CO2_measurements_using_statistical_learning_models/links/5b1d843ea6fdcca67b690c28/Accurate-occupancy-detection-of-an-office-room-from-light-temperature-humidity-and-CO2-measurements-using-statistical-learning-models.pdf?origin=publication_detail
 

    /* Hier muss browsersync gestartet werden 
    https://livebook.manning.com/book/deep-learning-with-javascript/chapter-7/v-8/87
    http://localhost:3000/data/datatraining.csv
    */

// Occupancy

const tf = require("@tensorflow/tfjs");
require('@tensorflow/tfjs-node')
const fs = require("fs")
const PrettyTable = require('prettytable');

let URL = "file://./data/datatraining.csv"

// Leider gibt es hier einen "Bug" die Anzahl der Spalten ist nicht gleich 
// Aus dem Grund muss zuerst die Datei konvertiert werden

async function convertFile() {

    fs.readFile("data/datatraining.txt", (error, content) => {

        content = content.toString().replace(/['"]+/g, '')
        content = content.replace("date", "index,date")

        fs.writeFile("data/datatraining.csv", content, (err) => {
            console.log("Datei wurde konvertiert!")
        });
    })

}


/** 
 * 
 *  Wir behalten nur die Spalten, die für das Training gebraucht werden 
 * 
 */
function filterColumns(input) {

    const values = [

        input.xs.Temperature,
        input.xs.Humidity,
        input.xs.Light,
        input.xs.CO2,
        input.xs.HumidityRatio
    ]

    return {
        xs: values,
        ys: input.ys.Occupancy
    }
}


/**  */
async function preprocess_data(csvDataset) {

    var minMax = [];

    // Mehr informationen über den CSVDataset:

    console.log("Spalten: " + await csvDataset.columnNames())

    let dataset = await csvDataset.map(filterColumns);
    let inputArray = []

    tf.util.shuffle(dataset, 42)
    dataset = await dataset.toArray();

    const output = await dataset.map((data) => {
        return [data.ys]
    });



    for (var col = 0; col < 5; col++) {
        let values =
            dataset.map(function (value, index) {
                return value.xs[col]
            });

        const min = Math.min(...values)
        const max = Math.max(...values)

        // Normalisierung der Werte
        values.forEach((value, i, a) => {
            a[i] = (value - min) / (max - min);
        });

        minMax.push({
            min: min,
            max: max
        })
        inputArray.push(values);
    }

    return {

        input: tf.tensor2d(inputArray).reshape([5, inputArray[0].length]).transpose(),
        output: tf.tensor2d(output),
        minMax
    }
}



function testOccupancy(model, inputs, minMax) {

    prettytable = new PrettyTable();
    prettytable.fieldNames(["Temperature", "Humidity", "Light", "CO2", "HumidityRatio", "Ist jemand da?"]);

    for (var p = 0; p < inputs.length; p++) {

        var input = [...inputs[p]]
        input.forEach((value, i, a) => {
            a[i] = (value - minMax[i].min) / (minMax[i].max - minMax[i].min);
        });

        var predictedValue = predictOccupancy(model, input)
        inputs[p].push(predictedValue);
        prettytable.addRow(inputs[p]);

    }
    prettytable.print();
}


function printOutput(output) {
    process.stdout.clearLine();
    process.stdout.cursorTo(0);
    process.stdout.write("Val. Acc:" + output);
}

async function run() {


    const csvDataset = await tf.data.csv(
        URL, {
            columnConfigs: {
                Occupancy: {
                    isLabel: true
                }
            }
        });

    trainingData = await preprocess_data(csvDataset)

    const model = tf.sequential();

    model.add(tf.layers.dense({
        units: 20,
        inputDim: trainingData.input.shape[1],
        activation: 'sigmoid'
    }))

    model.add(tf.layers.dense({
        units: 10,
        activation: 'sigmoid'
    }))

    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }))

    model.compile({
        optimizer: 'rmsprop',
        loss: tf.losses.meanSquaredError,
        metrics: ['accuracy']
    });


    const hist = await model.fit(trainingData.input, trainingData.output, {
        validationSplit: 0.2,
        batchSize: 32,
        epochs: 25,
        verbose: 1,
        callbacks: {
            onEpochEnd: (epoch, logs) => {},
            onTrainEnd: () => {

                console.log('\nTraining ist fertig!');
            },
            onTrainBegin: () => {
                console.log("Training startet...")

            }
        }
    })

    // Speichert die min und Max Werte
    let json = JSON.stringify(trainingData.minMax);
    fs.writeFile('minMax.json', json, 'utf8', () => {
        console.log("MinMax-Datei exportiert")
    });

    // Das Model wird dauerhaft gespeichert
    await model.save("file://./export");

    // Eingabebeispiele
    var testInputs = [
        [24.2, 23.7, 654.666666666667, 697.666666666667, 0.0044264636375791], //1
        [21.15, 26.895, 0, 561.5, 0.00417274569728117], //0, 
        [21.7, 28.5666666666667, 0, 582, 0.00458703092309197], //0
        [21.79, 28.5333333333333, 419, 596.666666666667, 0.00460709710922468], //1
        [21.1, 29.6633333333333, 0, 526.333333333333, 0.00459120807372759] //0

    ]

    // Ergebnisse
    testOccupancy(model, testInputs, trainingData.minMax);

}

function predictOccupancy(model, predictionInput) {

    var occupied = ["✔ nein", "⚠ ja"]
    var tt = tf.tensor2d([predictionInput])
    var predictedValue = model.predict(tt).dataSync();
    return occupied[Math.round(predictedValue)]
}

// convertFile().then(()=>{console.log("Datei konvertiert")});

run().then(() => console.log('Done'));