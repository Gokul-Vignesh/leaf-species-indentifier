// frontend/script.js

function uploadImage() {
  const input = document.getElementById('imageUpload');
  const file = input.files[0];
  const resultDiv = document.getElementById('result');

  if (!file) {
    resultDiv.innerText = "Please choose an image file.";
    return;
  }

  // Show preview
  const reader = new FileReader();
  reader.onload = function (e) {
    const img = document.getElementById('preview');
    img.src = e.target.result;
    img.style.display = 'block';
  };
  reader.readAsDataURL(file);

  const formData = new FormData();
  formData.append('file', file);

  resultDiv.innerText = "Predicting...";

  fetch('http://localhost:5000/predict', {
    method: 'POST',
    body: formData
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.prediction) {
        resultDiv.innerText = `🌿 Predicted Species: ${data.prediction}`;
      } else {
        resultDiv.innerText = `Error: ${data.error || 'Unknown error'}`;
      }
    })
    .catch((error) => {
      resultDiv.innerText = `Error: ${error.message}`;
    });
}

