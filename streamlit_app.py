import streamlit as st
import numpy as np
import librosa
import soundfile as sf
from tensorflow.keras.models import load_model
import tempfile
import os
from scipy import signal

# Load your saved model
MODEL_PATH = 'heart_sound_model.h5'
model = load_model(MODEL_PATH)

# Function to process raw audio data
def process_raw_audio(y, sr, n_mfcc=20):
    """Process raw audio data with full preprocessing pipeline"""
    # 1. Resample to 22050 Hz if necessary
    if sr != 22050:
        y = librosa.resample(y, orig_sr=sr, target_sr=22050)
        sr = 22050
    
    # 2. Apply bandpass filter (20-400Hz - typical heart sound range)
    nyquist = sr / 2
    low = 20 / nyquist
    high = 400 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    y = signal.filtfilt(b, a, y)
    
    # 3. Normalize audio
    y = librosa.util.normalize(y)
    
    # 4. Remove silence and ensure consistent length
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    
    # 5. Ensure exactly 5 seconds length
    target_length = sr * 5
    if len(y_trimmed) > target_length:
        y_trimmed = y_trimmed[:target_length]
    elif len(y_trimmed) < target_length:
        y_trimmed = np.pad(y_trimmed, (0, target_length - len(y_trimmed)))
    
    # 6. Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=n_mfcc)
    
    # 7. Add delta features
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    # 8. Reshape for model input
    mfcc = mfcc.T
    mfcc = mfcc.reshape(1, 216, 20)
    
    return mfcc

# Function to preprocess audio file
@st.cache_data
def preprocess_audio(file_path, n_mfcc=20):
    """Load and preprocess audio file"""
    y, sr = librosa.load(file_path, duration=5.0)
    return process_raw_audio(y, sr, n_mfcc)

# Function to validate audio quality
def validate_audio(file_path):
    """Validate audio quality metrics"""
    y, sr = librosa.load(file_path)
    
    # Calculate signal quality metrics
    noise_floor = np.mean(np.abs(y[:int(sr * 0.1)]))
    signal_power = np.mean(np.abs(y))
    snr = 20 * np.log10(signal_power / noise_floor) if noise_floor > 0 else 0
    peak_amplitude = np.max(np.abs(y))
    
    validation_results = {
        'snr': snr > 15,
        'amplitude': 0.1 < peak_amplitude < 0.9,
        'duration': len(y) >= sr * 5
    }
    
    return validation_results, y, sr

# Function to make prediction
def predict_heart_sound(preprocessed_data):
    """Make prediction and return result with confidence"""
    prediction = model.predict(preprocessed_data)
    is_unhealthy = prediction > 0.5
    confidence = prediction[0][0] if is_unhealthy else 1 - prediction[0][0]
    return is_unhealthy, confidence * 100

# Streamlit App
st.title("Heart Sound Detection Tool")
st.write("Upload a .wav file to get a prediction on whether the heart sound is healthy or unhealthy.")

uploaded_file = st.file_uploader("Choose a .wav file", type="wav")

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(uploaded_file.getbuffer())
        temp_path = temp_audio.name
    
    # Validate the audio file
    validation_results, audio_data, sr = validate_audio(temp_path)
    
    # Display the audio player
    st.audio(uploaded_file, format='audio/wav')
    
    # Check audio quality
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
        
        # Display results
        if is_unhealthy:
            st.error("Prediction: Unhealthy Heart Sound")
        else:
            st.success("Prediction: Healthy Heart Sound")
        st.write(f"Confidence: {confidence:.2f}%")
        
        # Add detailed analysis
        st.write("\nAudio Analysis:")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Signal Quality Metrics:")
            st.write(f"- Duration: {len(audio_data)/sr:.1f} seconds")
            st.write(f"- Sample Rate: {sr} Hz")
        with col2:
            st.write("Preprocessing Applied:")
            st.write("- Bandpass filtering (20-400 Hz)")
            st.write("- Noise reduction")
            st.write("- Length normalization")
        
        # Cleanup temporary file
        os.unlink(temp_path)

# Instructions for recording audio
st.write("\n### How to Record Heart Sounds")
st.write("""
1. Use a digital stethoscope or high-quality microphone
2. Record in a quiet environment
3. Position the recording device on these locations:
   - Aortic area (2nd right intercostal space)
   - Pulmonic area (2nd left intercostal space)
   - Tricuspid area (4th left intercostal space)
   - Mitral area (5th left intercostal space)
4. Record for at least 5 seconds
5. Save as a WAV file and upload above
""")

# Footer
st.write("\n**Note:** This tool is for preliminary detection purposes only and not for diagnostic use. Please consult a healthcare professional for proper diagnosis.")
