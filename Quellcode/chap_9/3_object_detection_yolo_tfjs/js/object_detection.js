let objects = [];
let yolo;
let now;
let then = Date.now();
let delta = 0;

const c = document.getElementById("myCanvas");
const ctx = c.getContext("2d");

/* Dimension des Videos bzw. von  */
const VIDEO_WIDTH = 640
const VIDEO_HEIGHT = 360

/* Konstanten, die als Parameter-Option zum YOLO-Modell angegeben werden*/
const CLASS_PROBABILITY_THRESHOLD = 0.60; //Liefert keine Erkennungen unter einer bestimmten Klassenwahrscheinlichkeit.
const FILTER_BOXES_THRESHOLD = 0.01; // Gibt keine Boxen zurück, die einen Schwellenwert unter box_prob * class_prob besitzen.
const IOU_THRESHOLD = 0.4 // (Intersection over Union) bzw. IoU = Fläche der Schnittmengen/Fläche Vereinungsmengen -  Kann die vorhergesagte
// Bounding Box mit der tatsächlich erkannten messen. Je näher  der IoU an 1.0 ist, desto präziser war die Erkennung 

/* Frame per Seconds für die Erkennung. Je grösser, desto öfters wird die Erkennung getriggert aber desto langsamer 
die Anzeige der Ergebnisse (wegen Performanz) */
const FPS = 5

/** Liste der Icons/Emojis */
emojis = [{
    name: "bus",
    emoji: "🚍"
  },
  {
    name: "car",
    emoji: "🚘"
  },
  {
    name: "person",
    emoji: "👦"
  },
  {
    name: "traffic light",
    emoji: "🚦"
  }
]

/* Instanzierung des YOLO Modells */
yolo = ml5.YOLO(
  { filterBoxesThreshold:FILTER_BOXES_THRESHOLD, 
    IOUThreshold: IOU_THRESHOLD, 
    classProbThreshold: CLASS_PROBABILITY_THRESHOLD }
  ,modelLoaded);

/* Wenn das Model geladen ist, wird die draw()-Methode getriggert */
function modelLoaded() {
  $("#loading").hide();
  console.log('YOLO Model loaded!');
  requestAnimationFrame(draw)
}

/* Von: http://codetheory.in/controlling-the-frame-rate-with-requestanimationframe/ */
function draw()
{
    requestAnimationFrame(draw)
    now = Date.now();
    delta = now - then;

    if (delta > 1000/FPS)
    {
      
      yolo.detect(document.getElementById("myVideo"), function (err, results) {
        objects = []
        objects = results
        drawDetectedObjects(objects)
      });
      then = now - (delta %  1000/FPS)
    }
  }

/** Falls eine passende Klasse gefunden wurde, wird ein Emoji zurückgegeben,
 * im Gegenfall wird nur der Name der Klasse als String zurückgegeben 
 */
function labelAsEmoji(className) {
  for (var i = 0; i < emojis.length; i++)
    if (emojis[i].name === className)
      return emojis[i].emoji;
  return className
}

/* Die erkannten Objekte werden mit einem roten Rand markiert */
function drawDetectedObjects(objects) {

  // Die vorherigen Zonen werden gelöscht 
  ctx.clearRect(0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);
  ctx.beginPath();

  $("#output").empty();
  $("#output").append("<div id=\"firstLine\"> Erkannte Objekte:</div>")

  objects.forEach((object) => {

    // Labeling
    ctx.beginPath();
    ctx.lineWidth = 0;
    ctx.strokeStyle = "red";
    ctx.fillStyle = "white";
    ctx.arc(object.x * VIDEO_WIDTH + 10, object.y * VIDEO_HEIGHT - 15, 12, 0, 2 * Math.PI);
    ctx.fill()
    ctx.closePath();

    // Umrahmung des Objekts
    ctx.lineWidth = 3;
    ctx.strokeStyle = "red";
    ctx.beginPath();
    ctx.rect(object.x * VIDEO_WIDTH, object.y * VIDEO_HEIGHT, object.w * VIDEO_WIDTH, object.h * VIDEO_HEIGHT);
    ctx.stroke();
    ctx.font = "15px Verdana";
    ctx.closePath()

    recognizedObject =  labelAsEmoji(object.className);

    // Anzeige des passenden Textes bzw. Emoji
    ctx.fillText(recognizedObject, object.x * VIDEO_WIDTH, object.y * VIDEO_HEIGHT - 10);
    $("#output").append("<div class=\"outputLine\"><b>"+ object.className  + "</b> " +  Math.round(object.classProb*100) +  "%  <div class=\"mini\">" + "@ (x:"+object.x + ",y:"+object.y+")</div></div>")
  });
  objects = []
}

