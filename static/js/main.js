function validateForm() {
    const songFile = document.querySelector('input[type="file"]');
    const youtubeLink = document.querySelector('input[name="link"]');

    if (songFile.files.length === 0 && youtubeLink.value.trim() === '') {
        const errorMessage = document.getElementById('error-message');
        errorMessage.textContent = 'Please upload a song or paste a YouTube link.';
        return false;
    }
}

function showPrediction(genre) {
    const form = document.getElementById('classification-form');
    const prediction = document.getElementById('prediction');
    const genreElement = document.getElementById('genre');
    genreElement.textContent = genre;
    form.style.display = 'none';
    prediction.style.display = 'block';
}

$('#classification-form').on('submit', function (event) {
    event.preventDefault();

    const form = event.target;
    const formData = new FormData(form);
    $('#classification-form').hide()
    // $('#logo').hide()
    $('#loading').show()

    $.ajax({
        url: form.action,
        type: form.method,
        data: formData,
        processData: false,
        contentType: false,
        success: function (data) {
            $('#loading').hide()
            showPrediction(data.genre);
        },
        error: function (error) {
            console.error('Error:', error);
        }
    });
});

$('#new-prediction-button').click(() => {
    $('#classification-form').show()
    $('#logo').show()
    $('#prediction').hide()
})