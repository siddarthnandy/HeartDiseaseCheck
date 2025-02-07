import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
import time
from scipy import signal
from tensorflow.keras.models import load_model

# Load your saved model
MODEL_PATH = 'heart_sound_model.h5'
model = load_model(MODEL_PATH)

# Function to process raw audio data
def process_raw_audio(y, sr, n_mfcc=20):
    """Process raw audio data with full preprocessing pipeline"""
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

# Function to validate audio quality
def validate_audio(file_path):
    y, sr = librosa.load(file_path)
    noise_floor = np.mean(np.abs(y[:int(sr * 0.1)]))
    signal_power = np.mean(np.abs(y))
    snr = 20 * np.log10(signal_power / noise_floor) if noise_floor > 0 else 0
    peak_amplitude = np.max(np.abs(y))
    validation_results = {'snr': snr > 15, 'amplitude': 0.1 < peak_amplitude < 0.9, 'duration': len(y) >= sr * 5}
    return validation_results, y, sr

# Function to make prediction
def predict_heart_sound(preprocessed_data):
    prediction = model.predict(preprocessed_data)
    is_unhealthy = prediction > 0.5
    confidence = prediction[0][0] if is_unhealthy else 1 - prediction[0][0]
    return is_unhealthy, confidence * 100

# Streamlit App
st.title("Heart Sound Detection Tool")
st.write("Upload a .wav file or record your heart sound for analysis.")

# Audio upload
uploaded_file = st.file_uploader("Choose a .wav file", type="wav")

# JavaScript-based audio recording
st.markdown(
    """
    <script>
    let mediaRecorder;
    let audioChunks = [];
    function startRecording() {
        navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.start();
            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };
        });
    }
    function stopRecording() {
        mediaRecorder.stop();
        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const audioUrl = URL.createObjectURL(audioBlob);
            const downloadLink = document.createElement('a');
            downloadLink.href = audioUrl;
            downloadLink.download = 'recorded_audio.wav';
            document.body.appendChild(downloadLink);
            downloadLink.click();
            document.body.removeChild(downloadLink);
        };
    }
    </script>
    <button onclick="startRecording()">Start Recording</button>
    <button onclick="stopRecording()">Stop Recording</button>
    """,
    unsafe_allow_html=True
)

temp_path = None
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(uploaded_file.getbuffer())
        temp_path = temp_audio.name
    st.audio(uploaded_file, format='audio/wav')

if temp_path:
    validation_results, audio_data, sr = validate_audio(temp_path)
    if not all(validation_results.values()):
        st.warning("Audio quality issues detected:")
        if not validation_results['snr']:
            st.write("- High background noise detected")
        if not validation_results['amplitude']:
            st.write("- Audio volume is not optimal")
        if not validation_results['duration']:
            st.write("- Audio is shorter than 5 seconds")
        st.write("These issues might affect the accuracy of the prediction.")
    
    if st.button("Analyze Audio"):
        with st.spinner("Processing audio..."):
            input_data = preprocess_audio(temp_path)
        with st.spinner("Making prediction..."):
            is_unhealthy, confidence = predict_heart_sound(input_data)
        if is_unhealthy:
            st.error("Prediction: Unhealthy Heart Sound")
        else:
            st.success("Prediction: Healthy Heart Sound")
        st.write(f"Confidence: {confidence:.2f}%")
        os.unlink(temp_path)

st.write("\n**Note:** This tool is for preliminary detection purposes only and not for diagnostic use. Please consult a healthcare professional for proper diagnosis.")
