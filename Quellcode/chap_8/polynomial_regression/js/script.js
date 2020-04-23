/** 
 *  Lineare regression mit TensorFlow.js
 */

let data;
const numbers = 10;

function predictAndDraw(input, model) {

  var resArray = [];
  for (var p = 0; p < numbers; p++)
    resArray.push(model.predict(tf.tensor2d([p], [1, 1])).dataSync());

  data.labels.push(Array.from(input.dataSync()))
  data.series.push({
    className: "ct-series-b",
    data: Array.from(resArray)
  })

  chart.update(data);
  data.labels.pop();
  data.series.pop();
}


/* Die Surface wird mit der aktuellen Epoche aktualisiert */
async function updateSurface(model, x, y) {
  const surface = tfvis.visor().surface({
    name: "Epoch",
    tab: 'Epoch'
  });

  // Training des Modells
  await model.fit(x, y, {
    epochs: 10000,
    batchSize: 5,
    validationData: [x, y],
    callbacks: {

      onEpochEnd: async (epoch, logs) => {
        surface.drawArea.innerHTML = "Aktuelle Epoche: <b>" + epoch + "</b>";
        surface.label.innerHTML = "Epoch: " + epoch;
        predictAndDraw(x, model);

        $("#epochs").html("<span>Epoch: " + epoch + "</span>");
        $("#loss").html("<span>Loss: " + logs.loss + "</span>");

        await tf.nextFrame();
      }
    }
  });
}

/* Trainingsmetriken */
async function visualizeTrainingMetrics(model, x, y) {

  const container = {
    name: 'Trainingsmetriken',
    tab: 'Trainingsmetriken'
  };

  await model.fit(x, y, {
    epochs: 10000,
    batchSize: 5,
    validationData: [x, y],
    callbacks: tfvis.show.fitCallbacks(container, ['loss', 'val_loss'])
  });
}

/* Mehrere Tabs + Modellübersicht */
async function visualizeOnEpochEndMetrics(model, x, y) {

  const historyContainer = {
    name: 'Metriken onEpochEnd ',
    tab: 'Metriken ',
  }

  const history = []

  await model.fit(x, y, {
    epochs: 10000,
    batchSize: 5,
    validationData: [x, y],
    callbacks: {

      onEpochEnd: async (epoch, logs) => {

        $("#epochs").html("<span>Epoch: " + epoch + "</span>");
        $("#loss").html("<span>Loss: " + logs.loss + "</span>");
        predictAndDraw(x, model);
        history.push(logs);
        tfvis.show.history(historyContainer, history, ['loss', "val_loss"]);
        await tf.nextFrame();
      }
    }
  });
}

/* Details über die einzelnen Schichten des Modells */
async function layersInformation(model, x, y) {

  const container = {
    name: 'Layer-Infos',
    tab: 'Layer-Infos'
  };
  tfvis.show.layer(container, model.getLayer('dense_Dense1'));
}

/** Generierung eines randomisierte Datensets */
(async () => {

  const mySurface = tfvis.visor().surface({
    name: 'Bereich',
    tab: 'Mein Tab'
  });


  var x = tf.range(0, numbers, 1).reshape([numbers, 1]);
  const y = x.pow(tf.tensor(2)).add(tf.tensor(5)); 

  // Kurve wird dargestellt 
  drawCurve(x, y)

  // Aufbau des Modells
  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: 10,
    inputShape: [1],
  }));

  model.add(tf.layers.activation({
    activation: "sigmoid"
  }));
  model.add(tf.layers.dense({
    units: 1,
    inputShape: [1],
  }));

  const learningRate = 0.0025;
  const optimizer = tf.train.sgd(learningRate);

  // Vorbereitung des Modells für das Training
  model.compile({
    loss: 'meanSquaredError',
    optimizer: optimizer,
    metrics: ['accuracy']
  });

  // Ausgabe der Modell-Struktur
  model.summary();

  tfvis.show.modelSummary({
    name: 'Modelübersicht',
    tab: 'Modelübersicht'
  }, model);


  // Beispiel zur Benutzung von tfjs-vis während des Trainings
  // Die folgenden Funktionen müssen einzelnd ausgeführt werden

  // Die Surface wird mit der aktuellen Epoche aktualisiert
     updateSurface(model,x,y)

  // Mehrere Tabs + Modellübersicht
  // visualizeOnEpochEndMetrics(model, x, y);

  // Trainingsmetriken
  // visualizeTrainingMetrics(model, x, y);

  // Details über die einzelnen Schichten des Modells
  // layersInformation(model,x,y);

  // Optional: Speichern des Modells am Ende des Trainings
  // await model.save("downloads://my-model")
})();