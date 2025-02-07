import streamlit as st
import numpy as np
import librosa
import soundfile as sf
from tensorflow.keras.models import load_model
import tempfile
import os
import sounddevice as sd
import wavio
import time
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
def validate_audio(audio_data, sr):
    """Validate audio quality metrics"""
    noise_floor = np.mean(np.abs(audio_data[:int(sr * 0.1)]))
    signal_power = np.mean(np.abs(audio_data))
    snr = 20 * np.log10(signal_power / noise_floor) if noise_floor > 0 else 0
    peak_amplitude = np.max(np.abs(audio_data))
    
    return {
        'snr': snr > 15,
        'amplitude': 0.1 < peak_amplitude < 0.9,
        'duration': len(audio_data) >= sr * 5
    }

# Function to record audio
def record_audio(duration, samplerate=22050):
    """Record audio with countdown"""
    for i in range(3, 0, -1):
        st.write(f"Recording starts in {i}...")
        time.sleep(1)
    
    st.write("Recording...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    st.write("Recording complete!")
    return recording.flatten(), samplerate  # Ensure 1D array

# Function to play audio
def play_audio(audio_data, samplerate):
    """Play audio data"""
    sd.play(audio_data, samplerate)
    sd.wait()

# Function to make prediction
def predict_heart_sound(preprocessed_data):
    """Make prediction and return result with confidence"""
    prediction = model.predict(preprocessed_data)
    is_unhealthy = prediction > 0.5
    confidence = prediction[0][0] if is_unhealthy else 1 - prediction[0][0]
    return is_unhealthy, confidence * 100

# Streamlit App
st.title("Heart Sound Detection Tool")
st.write("Upload a .wav file or record audio to get a prediction on whether the heart sound is healthy or unhealthy.")

# Create tabs for upload and record options
tab1, tab2 = st.tabs(["Upload Audio", "Record Audio"])

# Upload tab
with tab1:
    uploaded_file = st.file_uploader("Choose a .wav file", type="wav")
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(uploaded_file.getbuffer())
            temp_path = temp_audio.name
        
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("Analyze Uploaded Audio"):
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

# Record tab
with tab2:
    if 'recording_state' not in st.session_state:
        st.session_state.recording_state = 'stopped'
        st.session_state.recorded_audio = None
        st.session_state.recorded_file = None

    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Start Recording"):
            st.session_state.recording_state = 'recording'
            recording, sr = record_audio(7)  # Record 7 seconds
            
            # Validate recording quality
            validation_results = validate_audio(recording, sr)
            
            if all(validation_results.values()):
                # Save recording temporarily
                temp_path = tempfile.mktemp(suffix=".wav")
                wavio.write(temp_path, recording, sr, sampwidth=2)
                
                # Store both raw audio and processed version
                st.session_state.recorded_audio = recording
                st.session_state.recorded_file = temp_path
                st.session_state.recording_state = 'stopped'
                st.experimental_rerun()
            else:
                st.error("Recording quality issues detected:")
                if not validation_results['snr']:
                    st.write("- Too much background noise")
                if not validation_results['amplitude']:
                    st.write("- Recording volume too high or too low")
                if not validation_results['duration']:
                    st.write("- Recording too short")
                st.write("Please try recording again in a quieter environment.")

    with col2:
        if st.session_state.recorded_audio is not None:
            if st.button("Play Recording"):
                play_audio(st.session_state.recorded_audio, 22050)

    if st.session_state.recorded_file is not None:
        st.audio(st.session_state.recorded_file, format='audio/wav')
        
        if st.button("Analyze Recorded Audio"):
            with st.spinner("Processing audio..."):
                # Process the raw audio data directly
                input_data = process_raw_audio(
                    st.session_state.recorded_audio,
                    22050  # We know this is our recording sample rate
                )
            
            with st.spinner("Making prediction..."):
                is_unhealthy, confidence = predict_heart_sound(input_data)
            
            if is_unhealthy:
                st.error("Prediction: Unhealthy Heart Sound")
            else:
                st.success("Prediction: Healthy Heart Sound")
            st.write(f"Confidence: {confidence:.2f}%")

# Footer
st.write("\n**Note:** This tool is for preliminary detection purposes only and not for diagnostic use.")

# Cleanup temporary files when the app is closed
def cleanup():
    if st.session_state.recorded_file and os.path.exists(st.session_state.recorded_file):
        os.unlink(st.session_state.recorded_file)

# Register the cleanup function
import atexit
atexit.register(cleanup)
