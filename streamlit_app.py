import streamlit as st
import numpy as np
import librosa
import soundfile as sf
from tensorflow.keras.models import load_model
import tempfile
import os
from audio_recorder_streamlit import audio_recorder

# Load the trained model
MODEL_PATH = 'final_best_model.h5'
model = load_model(MODEL_PATH)

# Function to preprocess audio before feeding into the model
@st.cache_data
def preprocess_audio(file_path, n_mfcc=40):
    y, sr = librosa.load(file_path, duration=5.0)
    y = librosa.effects.preemphasis(y)
    y = y / np.max(np.abs(y)) * 0.95  # Normalize amplitude
    
    # Feature extraction
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    combined_features = np.vstack((mfcc, mfcc_delta, mfcc_delta2)).T
    
    # Reshape for model input
    combined_features = combined_features[:216, :]  # Ensure fixed shape
    combined_features = np.expand_dims(combined_features, axis=0)  # Add batch dimension
    return combined_features

# Function to amplify audio
def amplify_audio(file_path, factor=5.0):
    y, sr = librosa.load(file_path, sr=None)
    y = np.clip(y * factor, -1.0, 1.0)  # Amplify and clip to avoid distortion
    amplified_path = file_path.replace(".wav", "_amplified.wav")
    sf.write(amplified_path, y, sr)
    return amplified_path

# Streamlit UI Setup
st.set_page_config(page_title="CardioAI - Heart Sound Analysis", page_icon="🫀", layout="centered")
st.title("CardioAI - Heart Sound Analysis")
st.write("Analyze your heart sound for potential abnormalities by uploading or recording a sample.")

temp_path = None

# File uploader for manual file upload
uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"], label_visibility="visible")

# Audio recorder (basic recording without glitchy timing constraints)
audio_bytes = audio_recorder(icon_size="2x", recording_color="red", neutral_color="black", text="Record", key="audio_recorder")

# Handle uploaded file
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(uploaded_file.getbuffer())
        temp_path = temp_audio.name
    amplified_path = amplify_audio(temp_path)
    st.audio(amplified_path, format="audio/wav")

# Handle recorded audio
elif audio_bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_path = temp_audio.name
    amplified_path = amplify_audio(temp_path)
    st.audio(amplified_path, format="audio/wav")

# Process and predict
if temp_path is not None:
    if st.button("Analyze Heart Sound", use_container_width=True):
        with st.spinner("Processing audio... ⏳"):
            input_data = preprocess_audio(temp_path)
        with st.spinner("Analyzing heart sound... 🔄"):
            prediction = model.predict(input_data)

        # Display result
        if prediction > 0.5:
            st.error("Abnormal Heart Sound Detected. Consider consulting a doctor.")
        else:
            st.success("No abnormalities found -- Healthy Heart")

        # Clean up temporary file
        os.unlink(temp_path)
        os.unlink(amplified_path)

# Footer
st.markdown("---")
st.info("Disclaimer: CardioAI is not a medical tool and should not be used for diagnosis. Always consult a professional for medical concerns.")
