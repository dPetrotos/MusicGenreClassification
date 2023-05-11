from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import tensorflow as tf
import keras
import librosa
from pytube import YouTube
import math



app = Flask(__name__)
genres = ['Blues', 'Classical', 'Country', 'Disco', 'HipHop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
cnn_model = tf.keras.models.load_model('models/saved_models/CNN')
lstm_model = tf.keras.models.load_model('models/saved_models/LSTM')
gru_model = tf.keras.models.load_model('models/saved_models/GRU')
cnn_mel_model = tf.keras.models.load_model('models/saved_models/saved_models/CNN-MEL-91,5test')
cnn_kfold_mfcc_model = tf.keras.models.load_model('models/saved_models/CNN-Kfold-MFCC')
cnn_kfold_mel_model = tf.keras.models.load_model('models/saved_models/saved_models/CNN-MEL-Kfold')


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        pass
    else:
        return render_template('index.html')


@app.route('/classification', methods=['POST', 'GET'])
def classification():
    if request.method == 'POST':
        link = request.form['link']
        if not link:
            song = request.files['song']
            mfcc, mel = preprocess(song)
        else:
            song_name = get_audio_from_youtube_video(link)
            mfcc, mel = preprocess(f'{song_name}.mp4')
        predictions = dict()
        choice = request.form['model_choice']
        if choice == 'cnn':
            predictions['cnn'] = predict(cnn_model, mfcc[..., np.newaxis])
        elif choice == 'lstm':
            predictions['lstm'] = predict(lstm_model, mfcc)
        elif choice == 'gru':
            predictions['gru'] = predict(gru_model, mfcc)
        elif choice == 'cnn_mel':
            # mel = (mel + 12.67344) / 16.546448
            mean_std_mel = np.load('models/saved_models/saved_models/CNN-MEL-Kfold-mean_std/mean_std.npy')
            mean = mean_std_mel[0]
            std = mean_std_mel[1]
            mel = mel[..., np.newaxis]
            mel = (mel - mean) / std
            predictions['cnn_mel'] = predict(cnn_kfold_mel_model, mel)
        elif choice == 'cnn_kfold':
            mean_std = np.load('models/saved_models/CNN-Kfold-MFCC-mean_std/mean_std.npy')
            mean = mean_std[0]
            std = mean_std[1]
            mfcc_cnn_kfold = mfcc[..., np.newaxis]
            mfcc_cnn_kfold = (mfcc_cnn_kfold - mean) / std
            predictions['cnn_kfold'] = predict(cnn_kfold_mfcc_model, mfcc_cnn_kfold)
        else:
            predictions['cnn'] = predict(cnn_model, mfcc[..., np.newaxis])
            predictions['lstm'] = predict(lstm_model, mfcc)
            predictions['gru'] = predict(gru_model, mfcc)
            mel = (mel + 12.67344) / 16.546448
            predictions['cnn_mel'] = predict(cnn_mel_model, mel[..., np.newaxis])
        return render_template('classification.html', predictions=predictions)

@app.route('/model-info/<string:model>')
def model_info(model):
    return render_template(model + '.html')

def preprocess(song):
    sample_rate = 22050
    y, sr = librosa.load(path=song, sr=sample_rate)
    duration = librosa.get_duration(y=y, sr=sr)
    number_of_segments = int(duration / (30 / 10))
    samples_per_segment = int((30 / 10) * sr)
    hop_length = 512
    mfcc = np.empty(shape=(number_of_segments, 20, 130), dtype=np.float32)
    all_mel = np.empty([number_of_segments, 128, 130], dtype=np.float32)
    for n in range(number_of_segments):
        m = librosa.feature.mfcc(
            y=y[samples_per_segment * n: samples_per_segment * (n + 1)],
            sr=sr, n_mfcc=20, n_fft=2048,
            hop_length=hop_length,
            dtype=np.float32)
        # m = m.T
        mel = librosa.feature.melspectrogram(
            y=y[samples_per_segment * n: samples_per_segment * (n + 1)],
            sr=sr, n_fft=2048, hop_length=hop_length, dtype=np.float32)
        mel_db = librosa.power_to_db(mel)
        if m.shape[1] == math.ceil(samples_per_segment / hop_length):
            mfcc[n] = m
        if mel_db.shape[0] == all_mel.shape[1] and mel_db.shape[1] == all_mel.shape[2]:
            all_mel[n] = mel_db
    return mfcc, all_mel


def get_audio_from_youtube_video(link):
    yt = YouTube(link)
    audio_stream = yt.streams.filter(only_audio=True, file_extension='mp4').first()
    song_name = yt.title
    song_name = song_name.replace('/', ' ')
    song_name = song_name.replace("'", ' ')
    song_name = song_name.replace('"', '')
    song_name = song_name.replace('|', '')
    audio_stream.download(filename=f'{song_name}.mp4')
    return song_name


def predict(model, mfcc):
    # predicted_segment_index = np.empty(shape=mfcc.shape[0])
    # print(str(mfcc.shape))
    # for n, segment in enumerate(mfcc):
    #     print("segment shape : ", segment.shape)
    #     segment = segment[np.newaxis, ...]
    #     print("segment shape : ", segment.shape)
    #     segment_prediction = model.predict(segment)
    #     print("n = ", n)
    #     print("segment prediction", str(np.argmax(segment_prediction, axis=1)))
    #     predicted_segment_index[n] = np.argmax(segment_prediction, axis=1)[0]
    # print(predicted_segment_index.astype(int))
    # final_prediction = np.argmax(np.bincount(predicted_segment_index.astype(int)))
    predictions = model.predict(mfcc, batch_size=32)
    predictions = np.argmax(predictions, axis=1)
    print(predictions)
    final_prediction = np.argmax(np.bincount(predictions.astype(int)))
    return genres[final_prediction]

# For LSTM
# def predict(model, mfcc):
#     predicted_index = np.empty(shape=mfcc.shape[2])
#     print(mfcc.shape[2], 'shape')
#     mfcc = mfcc[np.newaxis, ...]
#     for segment in range(mfcc.shape[3]):
#         print(str(mfcc.shape))
#         print(str(mfcc[:, :, :, segment].shape))
#         segment_prediction = model.predict(mfcc[:, :, :, segment])
#         print(str(np.argmax(segment_prediction, axis=1)))
#         predicted_index[segment] = np.argmax(segment_prediction, axis=1)
#     print(predicted_index.astype(int))
#     final_prediction = np.argmax(np.bincount(predicted_index.astype(int)))
#
#     return genres[final_prediction]


if __name__ == '__main__':
    app.run(debug=True)