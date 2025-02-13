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

# Streamlit UI Setup
st.set_page_config(page_title="CardioAI - Heart Sound Analysis", page_icon="🫀", layout="centered")
st.title("CardioAI - Heart Sound Analysis")
st.write("Analyze your heart sound for potential abnormalities by uploading or recording a sample.")

# File uploader and recorder with minimal UI
temp_path = None

uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"], label_visibility="visible")
audio_bytes = audio_recorder(icon_size="2x", recording_color="red", neutral_color="black", text="Record (7s)", key="audio_recorder", pause_threshold=7.0)

# Handle uploaded file
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(uploaded_file.getbuffer())
        temp_path = temp_audio.name
    st.audio(uploaded_file, format="audio/wav")

# Handle recorded audio (fixed 7-second duration)
elif audio_bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_path = temp_audio.name
    st.audio(temp_path, format="audio/wav")

# Process and predict
if temp_path is not None:
    if st.button("Analyze Heart Sound", use_container_width=True):
        with st.spinner("Processing audio..."):
            input_data = preprocess_audio(temp_path)
        with st.spinner("Making prediction..."):
            prediction = model.predict(input_data)

        # Display result
        if prediction > 0.5:
            st.error("Unhealthy Heart Sound Detected. Consider consulting a doctor.")
        else:
            st.success("Your heart sound appears normal.")

        # Clean up temporary file
        os.unlink(temp_path)

# Footer
st.markdown("---")
st.info("Disclaimer: CardioAI is not a medical tool and should not be used for diagnosis. Always consult a professional for medical concerns.")
