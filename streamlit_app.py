import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
from scipy import signal
from tensorflow.keras.models import load_model
from audio_recorder_streamlit import audio_recorder

# Load your saved model
MODEL_PATH = 'heart_sound_model.h5'
model = load_model(MODEL_PATH)

# Function to process raw audio data
def process_raw_audio(y, sr, n_mfcc=20):
    if sr != 22050:
        y = librosa.resample(y, orig_sr=sr, target_sr=22050)
        sr = 22050
    nyquist = sr / 2
    low = 20 / nyquist
    high = 400 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    y = signal.filtfilt(b, a, y)
    y = librosa.util.normalize(y)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    target_length = sr * 5
    if len(y_trimmed) > target_length:
        y_trimmed = y_trimmed[:target_length]
    elif len(y_trimmed) < target_length:
        y_trimmed = np.pad(y_trimmed, (0, target_length - len(y_trimmed)))
    mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=n_mfcc)
    mfcc = mfcc.T.reshape(1, 216, 20)
    return mfcc

# Function to preprocess audio file
@st.cache_data
def preprocess_audio(file_path, n_mfcc=20):
    y, sr = librosa.load(file_path, duration=5.0)
    return process_raw_audio(y, sr, n_mfcc)

# Function to make prediction
def predict_heart_sound(preprocessed_data):
    prediction = model.predict(preprocessed_data)
    is_unhealthy = prediction > 0.5
    confidence = prediction[0][0] if is_unhealthy else 1 - prediction[0][0]
    return is_unhealthy, confidence * 100

# Streamlit App
st.title("🫀 Heart Sound Detection Tool")
st.write("Analyze your heart sounds by uploading a file or recording directly.")

# State variables for managing UI
if 'audio_bytes' not in st.session_state:
    st.session_state.audio_bytes = None
if 'temp_path' not in st.session_state:
    st.session_state.temp_path = None

# File upload and recording UI
st.markdown("### 📤 Upload or Record")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Upload a .wav file")
    uploaded_file = st.file_uploader("", type="wav", label_visibility="collapsed")

with col2:
    st.markdown("#### Record your heart sound")
    audio_bytes = audio_recorder(icon_size="2x", recording_color="red", neutral_color="black")

if audio_bytes:
    st.session_state.audio_bytes = audio_bytes
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        st.session_state.temp_path = temp_audio.name

# Redo and playback options
if st.session_state.audio_bytes:
    st.audio(st.session_state.temp_path, format='audio/wav')
    if st.button("🔄 Redo Recording"):
        st.session_state.audio_bytes = None
        st.session_state.temp_path = None

# Prediction and Results
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(uploaded_file.getbuffer())
        st.session_state.temp_path = temp_audio.name
    st.audio(uploaded_file, format='audio/wav')

if st.session_state.temp_path:
    if st.button("🔍 Analyze Heart Sound"):
        with st.spinner("Processing audio and making prediction..."):
            input_data = preprocess_audio(st.session_state.temp_path)
            is_unhealthy, confidence = predict_heart_sound(input_data)

        # Display Results
        if is_unhealthy:
            st.error(f"Prediction: **Unhealthy Heart Sound**\nConfidence: {confidence:.2f}%")
        else:
            st.success(f"Prediction: **Healthy Heart Sound**\nConfidence: {confidence:.2f}%")

        # Cleanup temporary file
        os.unlink(st.session_state.temp_path)
        st.session_state.temp_path = None

st.markdown("---")
st.write("**Note:** This tool is for preliminary detection purposes only. Please consult a healthcare professional for a proper diagnosis.")
