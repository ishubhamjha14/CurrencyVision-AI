// ==========================================================
// CurrencyVision AI
// script.js
// ==========================================================

// ------------------------------
// Elements
// ------------------------------

const uploadTab = document.getElementById("uploadTab");
const cameraTab = document.getElementById("cameraTab");

const uploadSection = document.getElementById("uploadSection");
const cameraSection = document.getElementById("cameraSection");

const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");

const startCamera = document.getElementById("startCamera");
const stopCamera = document.getElementById("stopCamera");
const capture = document.getElementById("capture");

let stream = null;

// ==========================================================
// Upload ↔ Camera Tabs
// ==========================================================

if (uploadTab && cameraTab) {

    uploadTab.addEventListener("click", () => {

        uploadSection.style.display = "block";
        cameraSection.style.display = "none";

        uploadTab.classList.add("active");
        cameraTab.classList.remove("active");

    });

    cameraTab.addEventListener("click", () => {

        uploadSection.style.display = "none";
        cameraSection.style.display = "block";

        cameraTab.classList.add("active");
        uploadTab.classList.remove("active");

    });

}

// ==========================================================
// Image Preview
// ==========================================================

if (imageInput) {

    imageInput.addEventListener("change", function () {

        const file = this.files[0];

        if (!file) return;

        previewImage.src = URL.createObjectURL(file);

        previewImage.style.display = "block";

    });

}

// ==========================================================
// Start Camera
// ==========================================================

if (startCamera) {

    startCamera.addEventListener("click", async () => {

        try {

            stream = await navigator.mediaDevices.getUserMedia({

                video: {

                    width: 1280,

                    height: 720

                }

            });

            video.srcObject = stream;

            video.style.display = "block";

            window.stream = stream;

            startCamera.style.display = "none";

            stopCamera.style.display = "inline-block";

        }

        catch (err) {

            alert("Camera permission denied.");

        }

    });

}
// ==========================================================
// Stop Camera
// ==========================================================

function stopCameraStream() {

    if (stream) {

        stream.getTracks().forEach(track => track.stop());

        stream = null;

    }

    video.srcObject = null;

    video.style.display = "none";

    canvas.style.display = "none";

    startCamera.style.display = "inline-block";

    stopCamera.style.display = "none";

}

if (stopCamera) {

    stopCamera.addEventListener("click", stopCameraStream);

}

// ==========================================================
// Capture Image
// ==========================================================

if (capture) {

    capture.addEventListener("click", () => {

        if (!video.srcObject) {

            alert("Please start camera first.");

            return;

        }

        canvas.width = video.videoWidth;

        canvas.height = video.videoHeight;

        const ctx = canvas.getContext("2d");

        ctx.drawImage(

            video,

            0,

            0,

            canvas.width,

            canvas.height

        );

        const image = canvas.toDataURL(

            "image/jpeg",

            0.95

        );

        predictWebcam(image);

    });

}

// ==========================================================
// Webcam Prediction
// ==========================================================

function predictWebcam(image) {

    fetch("/predict_webcam", {

        method: "POST",

        headers: {

            "Content-Type": "application/json"

        },

        body: JSON.stringify({

            image: image

        })

    })

    .then(response => response.json())

    .then(data => {

        document.getElementById("webcamResult").style.display = "block";

        const prediction = data.prediction;

        const confidence = data.confidence;

        if (

            prediction === "Background" ||

            prediction === "No Currency Detected"

        ) {

            document.getElementById(

                "cameraPrediction"

            ).innerHTML =

            "No Currency Detected";

        }

        else {

            document.getElementById(

                "cameraPrediction"

            ).innerHTML =

            "₹" + prediction;

        }

        document.getElementById(

            "cameraConfidence"

        ).innerHTML =

        confidence + "%";

        document.getElementById(

            "cameraBar"

        ).style.width =

        confidence + "%";

        document.getElementById(

            "cameraBar"

        ).innerHTML =

        confidence + "%";

        const img = document.getElementById(

            "cameraImage"

        );

        if (img) {

            img.src = image;

            img.style.display = "block";

        }

        speakPrediction(

            prediction,

            confidence

        );

    })

    .catch(error => {

        console.log(error);

        alert("Prediction Failed.");

    });

}
// ==========================================================
// Voice Output
// ==========================================================

function speakPrediction(prediction, confidence) {

    if (!("speechSynthesis" in window)) return;

    window.speechSynthesis.cancel();

    let message = "";

    if (
        prediction === "Background" ||
        prediction === "No Currency Detected"
    ) {

        message = "No currency note detected.";

    }

    else {

        message =
            "Detected " +
            prediction +
            " rupees. Confidence " +
            confidence.toFixed(1) +
            " percent.";

    }

    const speech = new SpeechSynthesisUtterance(message);

    speech.lang = "en-IN";
    speech.rate = 1;
    speech.pitch = 1;
    speech.volume = 1;

    window.speechSynthesis.speak(speech);

}

// ==========================================================
// Button Hover Animation
// ==========================================================

document.querySelectorAll(".btn").forEach(btn => {

    btn.addEventListener("mouseenter", () => {

        btn.style.transform = "translateY(-2px)";

    });

    btn.addEventListener("mouseleave", () => {

        btn.style.transform = "translateY(0px)";

    });

});

// ==========================================================
// Glass Card Hover
// ==========================================================

document.querySelectorAll(".glass-card").forEach(card => {

    card.addEventListener("mouseenter", () => {

        card.style.transform = "translateY(-6px)";

    });

    card.addEventListener("mouseleave", () => {

        card.style.transform = "translateY(0px)";

    });

});

// ==========================================================
// Footer Year
// ==========================================================

const year = document.getElementById("year");

if (year) {

    year.innerHTML = new Date().getFullYear();

}

// ==========================================================
// Cleanup on Page Exit
// ==========================================================

window.addEventListener("beforeunload", () => {

    if (stream) {

        stream.getTracks().forEach(track => track.stop());

    }

    window.speechSynthesis.cancel();

});