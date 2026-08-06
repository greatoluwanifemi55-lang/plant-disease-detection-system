// ============================================
// FED-XAI V2
// app.js
// ============================================

// Image Preview

const imageInput = document.getElementById("imageInput");

const imagePreview = document.getElementById("imagePreview");

if (imageInput) {

    imageInput.addEventListener("change", function () {

        const file = this.files[0];

        if (!file) return;

        const reader = new FileReader();

        reader.onload = function (event) {

            imagePreview.src = event.target.result;

            imagePreview.style.display = "block";

        };

        reader.readAsDataURL(file);

    });

}

// ============================================
// Drag & Drop
// ============================================

const dropArea = document.getElementById("dropArea");

if (dropArea && imageInput) {

    [
        "dragenter",
        "dragover",
        "dragleave",
        "drop"

    ].forEach(eventName => {

        dropArea.addEventListener(

            eventName,

            preventDefaults,

            false

        );

    });

    function preventDefaults(e) {

        e.preventDefault();

        e.stopPropagation();

    }

    [
        "dragenter",
        "dragover"

    ].forEach(eventName => {

        dropArea.addEventListener(

            eventName,

            () => {

                dropArea.style.background = "#ECFDF5";

                dropArea.style.borderColor = "#20C997";

            },

            false

        );

    });

    [
        "dragleave",
        "drop"

    ].forEach(eventName => {

        dropArea.addEventListener(

            eventName,

            () => {

                dropArea.style.background = "";

                dropArea.style.borderColor = "";

            },

            false

        );

    });

    dropArea.addEventListener(

        "drop",

        function (e) {

            const dt = e.dataTransfer;

            const files = dt.files;

            imageInput.files = files;

            imageInput.dispatchEvent(

                new Event("change")

            );

        }

    );

}

// ============================================
// Loading Button
// ============================================

const form = document.querySelector("form");

if (form) {

    form.addEventListener("submit", function () {

        const button = document.querySelector(

            ".btn-primary-custom"

        );

        if (button) {

            button.disabled = true;

            button.innerHTML =

                '<span class="spinner-border spinner-border-sm me-2"></span>Analyzing...';

        }

    });

}

// ============================================
// Fade Animation
// ============================================

const fadeElements = document.querySelectorAll(

    ".fade-up"

);

const observer = new IntersectionObserver(

    entries => {

        entries.forEach(entry => {

            if (entry.isIntersecting) {

                entry.target.style.opacity = "1";

                entry.target.style.transform =

                    "translateY(0)";

            }

        });

    },

    {

        threshold: .2

    }

);

fadeElements.forEach(el => {

    el.style.opacity = "0";

    el.style.transform = "translateY(40px)";

    observer.observe(el);

});

// ============================================
// Hover Effect
// ============================================

document.querySelectorAll(

    ".hover-card"

).forEach(card => {

    card.addEventListener(

        "mouseenter",

        () => {

            card.style.transform =

                "translateY(-8px)";

        }

    );

    card.addEventListener(

        "mouseleave",

        () => {

            card.style.transform =

                "translateY(0px)";

        }

    );

});

// ============================================
// Success
// ============================================

console.log(

    "FED-XAI V2 Loaded Successfully"

);
document.addEventListener("DOMContentLoaded", function () {

    const form = document.getElementById("predictionForm");

    if (!form) return;

    form.addEventListener("submit", function () {

        document.getElementById("buttonText").style.display = "none";

        document.getElementById("loadingSpinner").style.display = "inline-flex";

        const button = document.getElementById("analyzeBtn");

        button.disabled = true;

        button.style.opacity = ".8";

    });

});