import streamlit as st
import numpy as np
import librosa
import soundfile as sf
from sklearn.cluster import KMeans
import os

st.title("Speaker Gender Classification Based on Voice Features")

# Upload audio file
uploaded_file = st.file_uploader("Upload a short voice sample (wav format)", type=["wav"])

# Directory to save samples
if not os.path.exists("samples"):
    os.makedirs("samples")

def extract_pitch(signal, sr):
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    d = np.diff(autocorr)
    start = np.nonzero(d > 0)[0][0]
    peak = np.argmax(autocorr[start:]) + start
    pitch = sr / peak
    return pitch

def extract_features(file):
    signal, sr = librosa.load(file, sr=None)
    pitch = extract_pitch(signal, sr)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    return np.hstack(([pitch], mfcc_mean))

# Sample data (hardcoded pitch and MFCC values for demo)
sample_features = [
    # Male samples
    [110] + list(np.random.normal(0, 1, 13)),
    [120] + list(np.random.normal(0, 1, 13)),
    [125] + list(np.random.normal(0, 1, 13)),
    # Female samples
    [210] + list(np.random.normal(0, 1, 13)),
    [220] + list(np.random.normal(0, 1, 13)),
    [230] + list(np.random.normal(0, 1, 13))
]

# K-means clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(sample_features)

if uploaded_file is not None:
    file_path = os.path.join("samples", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    features = extract_features(file_path).reshape(1, -1)
    label = kmeans.predict(features)[0]

    # Infer gender based on cluster centers' average pitch
    center_pitches = [center[0] for center in kmeans.cluster_centers_]
    male_label = np.argmin(center_pitches)

    gender = "Male" if label == male_label else "Female"

    st.success(f"Predicted Gender: {gender}")
    st.write(f"Pitch: {features[0][0]:.2f} Hz")
