// Das Ergebnis wird im <div> result angezeigt 
// Die Konfidenzwerte werden <div> probability angezeigt 
$("#modelLoading").show();
$("#modelRecognized").hide();
$("#formBlock").hide();

// Initialisierung des Image Classifier mit MobileNet
const classifier = ml5.imageClassifier('MobileNet', modelLoaded);

// Wird aufgerufen, wenn das Modell vollständig geladen wurde
function modelLoaded() {
    $("#modelLoading").hide();
    $("#formBlock").show();
}

// Bildvorhersage mit Konfidenzwerte werden vom Modell zurückgegeben
function predictImage() {
    classifier.predict($("#picture").get(0), function (err, results) {
        $("#modelRecognized").show();
        $("#result").html(results[0].className);
        $("#probability").html(results[0].probability.toFixed(4));
    });
}

// Wenn auf den Button "Analysieren" gedrückt wird, wird die URL aus dem bildInputURL Eingabefeld genommen
$("#myButton").on("click", () => {
    
    $("#picture").on("load", () => {
        predictImage();
    });
    $("#picture").attr("src",$("#pictureInputURL").val())
});