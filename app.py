import os
import streamlit as st
import numpy as np
import librosa
from keras.models import load_model  # type: ignore
import matplotlib.pyplot as plt
import librosa.display
import tempfile

# Load the trained model
model = load_model('model.h5')

# Define emotion labels (same as in your training)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


# Function to preprocess audio (extract MFCC)
def preprocess_audio(uploaded_file):
    # Save the uploaded file to a temporary path
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    # Load the audio from the temporary file
    try:
        audio, sr = librosa.load(tmp_file_path, sr=None)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    mfccs = mfccs.reshape(1, 40, 1)
    
    # Clean up the temporary file
    os.remove(tmp_file_path)

    return mfccs


# Function to plot waveform
def plot_waveform(audio, sr, title="Waveform"):
    plt.figure(figsize=(10,4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title(title)
    st.pyplot()

# Function to plot spectrogram
def plot_spectrogram(audio, sr, title="Spectrogram"):
    X = librosa.stft(audio)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(10,4))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.title(title)
    st.pyplot()

# Streamlit UI
st.title("Emotion Recognition from Audio")

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Load and preprocess the uploaded audio file
    st.audio(uploaded_file)
    audio_data, sr = librosa.load(uploaded_file, sr=None)

    # Display waveform and spectrogram
    plot_waveform(audio_data, sr, "Uploaded Audio - Waveform")
    plot_spectrogram(audio_data, sr, "Uploaded Audio - Spectrogram")

    # Preprocess the audio for prediction
    audio_features = preprocess_audio(uploaded_file)

    # Make predictions
    predictions = model.predict(audio_features)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_emotion = emotion_labels[predicted_class[0]]

    # Display the predicted emotion
    st.write(f"Predicted Emotion: {predicted_emotion}")

