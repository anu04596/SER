**🎙️ Speech Emotion Recognition using LSTM**

This project aims to detect human emotions from speech using deep learning. We use audio files from the TESS dataset, extract meaningful features (MFCCs), and classify emotions using an LSTM (Long Short-Term Memory) neural network. A web interface is built using Flask and React for real-time emotion detection.

**📌 Project Highlights**
1. 🎧 Real-time emotion detection from speech
2. 📂 Uses TESS dataset (Toronto Emotional Speech Set)
3. 🔍 Feature extraction using MFCC (Mel Frequency Cepstral Coefficients)
4. 🧠 Emotion classification using LSTM neural network
5. 🌐 Deployed using Flask (backend) and React (frontend)

**🔥 Emotions Detected**
1. Angry
2. Disgust
3. Fear
4. Happy
5. Neutral
6. Sad
Surprise

**📁 Dataset**
TESS Dataset: Contains 2800+ audio files of actors reading sentences in 7 different emotional tones.
Link: TESS Dataset on Kaggle

🧪 Tech Stack
* Tool/Library:	Purpose
* Python:	Core programming
* Librosa:	Audio preprocessing & MFCC extraction
* Keras/TensorFlow:	Building the LSTM model
* NumPy, Pandas:	Data handling
* Matplotlib:	Visualization
* Flask:	Backend API
* React.js:	Frontend interface

🛠️ How It Works
1. Preprocessing:
* Load .wav files
* Extract MFCC features
* Normalize and pad sequences

2. Model Architecture:
* Input layer (MFCC features)
* LSTM layers
* Dense layers with Softmax activation

3. Training:
* Trained on labeled emotion audio samples
* Accuracy achieved: 0.9929

4. Prediction:
* User records or uploads audio
* Audio is preprocessed and fed to the model
* Model returns the predicted emotion

5. 🚀 Deployment
* Flask API endpoint: /analyze
* React frontend: Allows recording/uploading audio

