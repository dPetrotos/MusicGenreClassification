import os
from flask import Flask, render_template, url_for, request, redirect, abort, jsonify

import numpy as np
import keras
import librosa
from pytube import YouTube
import re
import tensorflow as tf
import time

app = Flask(__name__)
genres = ['Blues', 'Classical', 'Country', 'Disco', 'HipHop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
cnn_kfold_mfcc_model = tf.keras.models.load_model('models/saved_models/CNN-Kfold-MFCC-final')
mean_std_mfcc = np.load('models/saved_models/CNN-Kfold-MFCC-final-mean_std/mean_std.npy')
cnn_kfold_mel_model = tf.keras.models.load_model('models/saved_models/CNN-MEL-Kfold')
mean_std_mel = np.load('models/saved_models/CNN-MEL-Kfold-mean_std/mean_std.npy')

# Dummy prediction to trigger PTX compilation which will be cached
dummy_input = np.zeros((1, 20, 130, 1), dtype=np.float32)
_ = cnn_kfold_mfcc_model.predict(dummy_input)


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        pass
    else:
        return render_template('index.html')


@app.route('/classification', methods=['POST'])
def classification():
    downloaded = False
    if request.method == 'POST':
        link = request.form['link']
        if not link:
            song = request.files['song']
        else:
            song_name = get_audio_from_youtube_video(link)
            downloaded = True
            song = f'songs/{song_name}.mp4'
        choice = request.form['model_choice']
        start_time = time.time()
        loaded_song, number_of_segments, samples_per_segment = load_song(song)
        load_time = time.time() - start_time
        if choice == 'mel':
            mel = extract_mel_spectrogram(loaded_song, number_of_segments, samples_per_segment)
            mel = normalise(mel, mean=mean_std_mel[0], std=mean_std_mel[1])
            genre = predict(cnn_kfold_mel_model, mel)
        else:
            mfcc = extract_mfcc(loaded_song, number_of_segments, samples_per_segment)
            mfcc = normalise(mfcc, mean=mean_std_mfcc[0], std=mean_std_mfcc[1])
            genre = predict(cnn_kfold_mfcc_model, mfcc)
        print('Load time = {:.2f}'.format(load_time))
        if downloaded:
            os.remove(f'{song}')
        return jsonify(genre=genre)


def get_audio_from_youtube_video(link):
    yt = YouTube(link)
    audio_stream = yt.streams.filter(only_audio=True, file_extension='mp4').first()
    song_name = yt.title
    song_name = clean_song_name(song_name)
    audio_stream.download(output_path='songs', filename=f'{song_name}.mp4')
    return song_name


def clean_song_name(song_name):
    pattern = r'[<>:"\'/\\|?*\x00-\x1F]'
    return re.sub(pattern, ' ', song_name)


def load_song(song):
    loaded_song, sr = librosa.load(path=song, sr=22050, dtype=np.float32)
    duration = librosa.get_duration(y=loaded_song, sr=sr)
    number_of_segments = int(duration / 3)
    samples_per_segment = int(3 * sr)
    return loaded_song, number_of_segments, samples_per_segment


def extract_mel_spectrogram(loaded_song, number_of_segments, samples_per_segment):
    all_mel = np.empty([number_of_segments, 128, 130], dtype=np.float32)
    for n in range(number_of_segments):
        mel = librosa.feature.melspectrogram(y=loaded_song[samples_per_segment * n: samples_per_segment * (n + 1)],
                                             dtype=np.float32)
        mel_db = librosa.power_to_db(mel)
        if mel_db.shape[0] == all_mel.shape[1] and mel_db.shape[1] == all_mel.shape[2]:
            all_mel[n] = mel_db
    all_mel = all_mel[..., np.newaxis]
    return all_mel


def extract_mfcc(loaded_song, number_of_segments, samples_per_segment):
    all_mfcc = np.empty(shape=(number_of_segments, 20, 130), dtype=np.float32)
    for n in range(number_of_segments):
        mfcc = librosa.feature.mfcc(y=loaded_song[samples_per_segment * n: samples_per_segment * (n + 1)],
                                    dtype=np.float32)
        if mfcc.shape[0] == all_mfcc.shape[1] and mfcc.shape[1] == all_mfcc.shape[2]:
            all_mfcc[n] = mfcc
    all_mfcc = all_mfcc[..., np.newaxis]
    return all_mfcc


def normalise(data, mean, std):
    return (data - mean) / std if std != 0 else data


def predict(model, data):
    predictions = model.predict(data, batch_size=32)
    predictions = np.argmax(predictions, axis=1)
    final_prediction = np.argmax(np.bincount(predictions.astype(int)))
    return genres[final_prediction]


if __name__ == '__main__':
    app.run(debug=False)
