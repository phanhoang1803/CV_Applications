document.addEventListener('DOMContentLoaded', function() {
    // Initial content to load on page load
    loadContent('home');

    // Add event listeners to navigation links
    document.querySelectorAll('nav a').forEach(link => {
        link.addEventListener('click', function(event) {
            event.preventDefault();
            const target = this.getAttribute('href').substring(1);
            loadContent(target);
        });
    });
});

function loadContent(target) {
    const contentDiv = document.getElementById('content');
    const xhr = new XMLHttpRequest();
    xhr.onload = function() {
        if (xhr.status >= 200 && xhr.status < 300) {
            contentDiv.innerHTML = xhr.responseText;
        } else {
            console.error('Failed to load content:', xhr.statusText);
        }
    };
    xhr.onerror = function() {
        console.error('Network error occurred');
    };
    xhr.open('GET', target + '.html', true);
    xhr.send();
}

function classifyDog() {
    console.log("Classify Dog function triggered");

    const file = document.getElementById('dogImage').files[0];
    const formData = new FormData();
    formData.append('file', file);

    fetch('http://127.0.0.1:5000/dog_classification/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        // Display the image preview
        const previewImg = document.getElementById("preview");
        previewImg.src = URL.createObjectURL(file);

        // Display the image container
        const previewContainer = document.getElementById("preview-container");
        previewContainer.style.display = "flex";

        // Display the prediction result
        document.getElementById("prediction").innerText = data.prediction;
    })
    .catch(error => console.error('Error:', error));
}

function detectCar() {
    console.log("detectCar function triggered");

    const file = document.getElementById('carImage').files[0];
    const formData = new FormData();
    formData.append('file', file);

    fetch('http://127.0.0.1:5000/car_detection/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        // Display the image preview
        const previewImg = document.getElementById("preview");
        previewImg.src = 'data:image/jpeg;base64,' + data.prediction;

        // Display the image container
        const previewContainer = document.getElementById("preview-container");
        previewContainer.style.display = "flex";
    })
    .catch(error => console.error('Error:', error));
}