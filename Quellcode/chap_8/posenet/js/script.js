let ctx;
let now;
let then = Date.now();
let delta;
let confidenceValue = 0.5; // Dieser Wert wird vom Slider innerhalb des HTML-Dokuments gesetzt

let net;
let estimatedPose;
let videoElement;

const FPS = 5;
const POSE_SINGLE_MODE = "singlePose";
const IMAGE_SCALE_FACTOR = 1;
const FLIP_HORIZONTAL = false;
const OUTPUT_STRIDE = 16;
const MAX_POSE_DETECTION = 8;
const RADIUS = 20;

// nms Radius: controls the minimum distance between poses that are returned
// defaults to 20, which is probably fine for most use cases

// Wenn das komplette Dokument geladen wurde

ctx = $("#pose_overlay_canvas").get(0).getContext("2d");
videoElement = $("#video").get(0);

$('#videoFile').change(function (event) {
    const video = $("#video")[0]
    video.src = URL.createObjectURL(event.target.files[0]);;
    video.load();
    video.play();
});

$("#confidenceValueInput").change(function (event) {
    confidenceValue = $(this).val()
    $("#confidenceValueText").text(confidenceValue);
});

// Das Posenet-Modell wird geladen
// je nach Internet-Verbindung kann das Laden der Gewichtungen 
// und des Modells etwas dauern 

async function loadPoseNetModel() {
    net = await posenet.load(0.5);
    $('#video').trigger('play');
    detectPose();
}

loadPoseNetModel()

function drawKeypoints(poses, showNames) {

    poses.forEach((singlePose) => {
        singlePose.keypoints.forEach((keypoint) => {
            ctx.beginPath();
            ctx.arc(keypoint.position.x, keypoint.position.y, 5, 0, 2 * Math.PI);
            ctx.fillStyle = 'red';
            ctx.fill();

            if (showNames) {
                ctx.font = "10px Verdana";
                ctx.fillText(keypoint.part, keypoint.position.x + 20, keypoint.position.y);
            }
        });
    });
}

// Zeichnet das Skelett
function drawSkeleton(poses) {
    poses.forEach((singlePose) => {
        const adjacentKeyPoints = posenet.getAdjacentKeyPoints(singlePose.keypoints, confidenceValue);
        adjacentKeyPoints.forEach((keypoints) => {
            ctx.beginPath();
            ctx.moveTo(keypoints[0].position.x, keypoints[0].position.y);
            ctx.lineTo(keypoints[1].position.x, keypoints[1].position.y);
            ctx.lineWidth = 5;
            ctx.strokeStyle = "green";
            ctx.stroke();
        });
    });
}


async function detectPose() {

    now = Date.now();
    delta = now - then;

    if (delta > 1000 / FPS) {

        then = now - delta % 1000 / FPS;
        if (!$('video').paused) {

            ctx.clearRect(0, 0, 640, 480);

            // Modus (eine oder mehrere Posen) wird durch die ComboBox ausgewählt
            if ($("#poseModeSelect").val() === POSE_SINGLE_MODE)
                estimatedPose = await estimateSinglePose();
            else
                estimatedPose = await estimateMultiplePoses();

            // Einblendung der Namen der Parts (durch Anklicken der Checkbox)
            drawKeypoints(estimatedPose, $("#showPartNames").is(":checked"));

            // Visualisierung des Skeletts (durch Anklicken der Checkbox)
            if ($("#showSkeleton").is(":checked"))
                drawSkeleton(estimatedPose)
        }
    }
    requestAnimationFrame(detectPose);
}

/* Aus: https://github.com/tensorflow/tfjs-models/blob/master/posenet/demos/camera.js */

async function estimateSinglePose() {
    var pose = await net.estimateSinglePose(videoElement, IMAGE_SCALE_FACTOR, FLIP_HORIZONTAL,
        OUTPUT_STRIDE);
    pose = [pose];
    return pose;
}

async function estimateMultiplePoses() {
    var poses = net.estimateMultiplePoses(videoElement, IMAGE_SCALE_FACTOR, FLIP_HORIZONTAL,
        OUTPUT_STRIDE,
        MAX_POSE_DETECTION,
        confidenceValue,
        RADIUS);
    return poses;

}