import streamlit as st
import numpy as np
import librosa
from sklearn.cluster import KMeans
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av
import soundfile as sf
import tempfile

st.title("🎙️ Real-Time Gender Detection from Microphone")

# WebRTC settings
client_settings = ClientSettings(
    media_stream_constraints={"audio": True, "video": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)

# Dummy sample features for clustering
sample_features = [
    [110] + list(np.random.normal(0, 1, 13)),
    [120] + list(np.random.normal(0, 1, 13)),
    [125] + list(np.random.normal(0, 1, 13)),
    [210] + list(np.random.normal(0, 1, 13)),
    [220] + list(np.random.normal(0, 1, 13)),
    [230] + list(np.random.normal(0, 1, 13)),
]

kmeans = KMeans(n_clusters=2, random_state=0).fit(sample_features)

# Feature extraction

def extract_pitch(signal, sr):
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    d = np.diff(autocorr)
    start = np.nonzero(d > 0)[0][0]
    peak = np.argmax(autocorr[start:]) + start
    pitch = sr / peak
    return pitch

def extract_features(audio_np, sr):
    pitch = extract_pitch(audio_np, sr)
    mfcc = librosa.feature.mfcc(y=audio_np, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    return np.hstack(([pitch], mfcc_mean))

# AudioProcessor for streamlit-webrtc
class AudioProcessor:
    def __init__(self):
        self.result = None

    def recv(self, frame: av.AudioFrame):
        samples = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
        sr = frame.sample_rate

        if len(samples) > sr // 2:
            try:
                features = extract_features(samples, sr).reshape(1, -1)
                label = kmeans.predict(features)[0]
                center_pitches = [c[0] for c in kmeans.cluster_centers_]
                male_label = np.argmin(center_pitches)
                gender = "Male" if label == male_label else "Female"
                pitch = features[0][0]
                self.result = (gender, pitch)
            except Exception as e:
                self.result = ("Error", 0)
        return frame

# Stream audio
ctx = webrtc_streamer(
    key="gender-detector",
    mode=WebRtcMode.SENDONLY,
    client_settings=client_settings,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

# Show result
if ctx.audio_processor:
    result = ctx.audio_processor.result
    if result:
        gender, pitch = result
        st.markdown(f"**Predicted Gender:** {gender}")
        st.markdown(f"**Pitch:** {pitch:.2f} Hz")
