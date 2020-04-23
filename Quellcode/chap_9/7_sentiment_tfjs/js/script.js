/**
 * Projekt 6: Sentimentanalyse mit TensorFlow.js
 */

let wordIds = []
let emojiArray = ["☹️", "🙁", "😐", "🙂", "😀"]
let sentimentArray = ["Negative", "Rather negative", "Neutral", "Positive", "Very Positive"]
let model;

function getSentimentValue(text) {

    // Word to Ids

    text = text.toLowerCase()
    text = text.replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g, " ")
  
    const sentenceIds = sentenceToIds(text)

    // Padding
    // Achtung: hier müssen Sie den gleichen Wert von PAD_MAX_LENGTH  
    // angeben wie in sentiment.py 
    let PAD_MAX_LENGTH = 1000
    const paddedSentence = padLeft(sentenceIds, PAD_MAX_LENGTH)
    e = tf.tensor([paddedSentence]);
    return model.predict(e).dataSync()[0];
}

function padLeft(sentenceIds, sentenceLength) {
    const paddedSentence = [];
    maxLength = sentenceLength - sentenceIds.length
    return paddedSentence.concat(new Array(maxLength).fill(0),sentenceIds)
}


function sentenceToIds(text) {
    const messageIds = []
    text.split(" ").forEach((word) => {
        messageIds.push(wordIds[word] + 3) // Achtung hier +3 wegen INDEX FROM !!!
    })
    return messageIds;
}

$("#message").hide()
$("#message").keyup(() => {

    if (5 < $("#message").val().length) {
        const message = $("#message").val().toLowerCase();
        const sentiment = getSentimentValue(message);
        const emojiIndex = Math.round(sentiment * 4);

        $("#result").text("Sentiment: " + sentimentArray[emojiIndex] + " (" +
            sentiment
            .toFixed(4) + ")");

        $("#emoji").html(emojiArray[emojiIndex]);
        $("#emoji").show();
        $("#result").show();

    } else {
        $("#emoji").hide();
        $("#result").hide();

    }
});

async function loadModel(){
    model = await tf.loadLayersModel('http://localhost:3000/tfjs_sentiment_model/model.json');
    model.summary()
}

loadModel()

// Wie wir es schon im Python-Code gesehen haben, 
// müssen wir den Satz oder besser gesagt, die einzelnen
// Wörtern zu id umwandeln
// in https://github.com/tensorflow/tensorflow/blob/r1.9/tensorflow/python/keras/datasets/imdb.py
// steht innerhalb der get_word_index() Funktion den Pfad zur 
// word_index.json Datei. Wir werden diese nun herunterladen und somit
// dann in der Lage sein, den Satz in ids umzuwandeln

// var url = "https://s3.amazonaws.com/text-datasets/imdb_word_index.json"

const url = "imdb_word_index.json"

// Erst wenn diese URL geladen ist, können wir was eintippen
$.getJSON(url, (data) => {
    wordIds = data
    $("#message").show()
});