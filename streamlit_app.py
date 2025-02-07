import streamlit as st
import numpy as np
import librosa
import soundfile as sf
from tensorflow.keras.models import load_model
import tempfile
import os
from audio_recorder_streamlit import audio_recorder

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
st.title("🫀 Heart Sound Detection Tool")
st.write("Upload a .wav file or record your heart sound to get a prediction.")

# File uploader and recorder
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Upload a .wav file")
    uploaded_file = st.file_uploader("", type="wav", label_visibility="collapsed")

with col2:
    st.markdown("#### Record your heart sound")
    audio_bytes = audio_recorder(icon_size="2x", recording_color="red", neutral_color="black")

temp_path = None

# Handle uploaded file
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(uploaded_file.getbuffer())
        temp_path = temp_audio.name
    st.audio(uploaded_file, format="audio/wav")

# Handle recorded audio
elif audio_bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_path = temp_audio.name
    st.audio(temp_path, format="audio/wav")

# Process and predict
if temp_path is not None:
    if st.button("🔍 Analyze Heart Sound"):
        with st.spinner("Processing audio..."):
            input_data = preprocess_audio(temp_path)
        with st.spinner("Making prediction..."):
            prediction = model.predict(input_data)

        # Display result
        if prediction > 0.5:
            st.error("Prediction: Unhealthy Heart Sound")
        else:
            st.success("Prediction: Healthy Heart Sound")

        # Clean up temporary file
        os.unlink(temp_path)

# Footer
st.markdown("---")
st.write("**Note:** This tool is for preliminary detection purposes only and not for diagnostic use.")
