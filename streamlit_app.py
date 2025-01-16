import streamlit as st
import numpy as np
import librosa
import soundfile as sf
from tensorflow.keras.models import load_model
import tempfile
import os

# Load your saved model
MODEL_PATH = 'heart_sound_model.h5'
model = load_model(MODEL_PATH)

# Function to preprocess audio
@st.cache_data
def preprocess_audio(file_path, n_mfcc=20):
    y, sr = librosa.load(file_path, duration=5.0)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = mfcc.T
    mfcc = mfcc.reshape(1, 216, 20)
    return mfcc

# Streamlit App
st.title("Heart Sound Detection Tool")
st.write("Upload a .wav file to get a prediction on whether the heart sound is healthy or unhealthy.")

# File uploader
uploaded_file = st.file_uploader("Choose a .wav file", type="wav")

# Process uploaded file
if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(uploaded_file.getbuffer())
        temp_path = temp_audio.name
    
    # Preprocess the audio file
    st.write("Processing audio...")
    input_data = preprocess_audio(temp_path)

    # Make prediction
    st.write("Making prediction...")
    prediction = model.predict(input_data)

    # Display result
    if prediction > 0.5:
        st.error("Prediction: Unhealthy Heart Sound")
    else:
        st.success("Prediction: Healthy Heart Sound")

    # Clean up temporary file
    os.unlink(temp_path)

# Footer
st.write("\n**Note:** This tool is for preliminary detection purposes only and not for diagnostic use.")
