import streamlit as st
import numpy as np
import librosa
from sklearn.cluster import KMeans
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import matplotlib.pyplot as plt
from collections import deque
import time

st.set_page_config(layout="centered")
st.title("🎙️ Real-Time Gender Detection")

# KMeans on dummy pitch+MFCCs
sample_features = [
    [110] + list(np.random.normal(0, 1, 13)),
    [120] + list(np.random.normal(0, 1, 13)),
    [125] + list(np.random.normal(0, 1, 13)),
    [210] + list(np.random.normal(0, 1, 13)),
    [220] + list(np.random.normal(0, 1, 13)),
    [230] + list(np.random.normal(0, 1, 13)),
]
kmeans = KMeans(n_clusters=2, random_state=0).fit(sample_features)
past_predictions = deque(maxlen=30)

def extract_pitch(signal, sr):
    autocorr = np.correlate(signal, signal, mode='full')[len(signal):]
    d = np.diff(autocorr)
    start = np.nonzero(d > 0)[0][0]
    peak = np.argmax(autocorr[start:]) + start
    return sr / peak if peak > 0 else 0

def extract_features(audio_np, sr):
    pitch = extract_pitch(audio_np, sr)
    mfcc = librosa.feature.mfcc(y=audio_np, sr=sr, n_mfcc=13)
    return np.hstack(([pitch], np.mean(mfcc, axis=1))), pitch

# Shared state
if "gender_result" not in st.session_state:
    st.session_state["gender_result"] = None
    st.session_state["pitch"] = 0
    st.session_state["energy"] = 0

class AudioProcessor:
    def recv(self, frame: av.AudioFrame):
        samples = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
        sr = frame.sample_rate
        energy = np.mean(samples**2)

        if len(samples) > sr // 2:
            features, pitch = extract_features(samples, sr)
            label = kmeans.predict([features])[0]
            center_pitches = [c[0] for c in kmeans.cluster_centers_]
            gender = "Male" if label == np.argmin(center_pitches) else "Female"
        else:
            gender = "Silent"

        # update state
        st.session_state["gender_result"] = gender
        st.session_state["pitch"] = pitch
        st.session_state["energy"] = energy

        return frame

# Streamlit UI
ctx = webrtc_streamer(
    key="gender",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

gender_box = st.empty()
pitch_box = st.empty()
energy_box = st.empty()
chart_box = st.empty()

# live render loop
while ctx.state.playing:
    gender = st.session_state["gender_result"]
    pitch = st.session_state["pitch"]
    energy = st.session_state["energy"]

    if gender:
        gender_box.markdown(f"**🧑 Gender:** `{gender}`")
        pitch_box.markdown(f"🎵 Pitch: `{pitch:.2f}` Hz")
        energy_box.markdown(f"🎚️ Energy: `{energy:.6f}`")

        past_predictions.append(gender)

        # Chart
        male_count = sum(1 for g in past_predictions if g == "Male")
        female_count = sum(1 for g in past_predictions if g == "Female")
        silent_count = sum(1 for g in past_predictions if g == "Silent")

        fig, ax = plt.subplots()
        ax.bar(["Male", "Female", "Silent"], [male_count, female_count, silent_count])
        ax.set_ylim(0, 30)
        ax.set_title("Gender Prediction History")
        chart_box.pyplot(fig)

    time.sleep(1)
