import streamlit as st
import numpy as np
import librosa
from sklearn.cluster import KMeans
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import matplotlib.pyplot as plt
import librosa.display
from collections import deque
import threading

st.set_page_config(layout="centered")
st.title("🎙️ Real-Time Gender Detection")

# --- Global shared state ---
if 'past_predictions' not in st.session_state:
    st.session_state.past_predictions = deque(maxlen=30)
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'waveform' not in st.session_state:
    st.session_state.waveform = None
if 'mfcc' not in st.session_state:
    st.session_state.mfcc = None
if 'energy' not in st.session_state:
    st.session_state.energy = 0

# --- Dummy KMeans ---
sample_features = [
    [110] + list(np.random.normal(0, 1, 13)),
    [120] + list(np.random.normal(0, 1, 13)),
    [125] + list(np.random.normal(0, 1, 13)),
    [210] + list(np.random.normal(0, 1, 13)),
    [220] + list(np.random.normal(0, 1, 13)),
    [230] + list(np.random.normal(0, 1, 13)),
]
kmeans = KMeans(n_clusters=2, random_state=0).fit(sample_features)

# --- Audio Feature Extraction ---
def extract_pitch(signal, sr):
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    d = np.diff(autocorr)
    try:
        start = np.nonzero(d > 0)[0][0]
        peak = np.argmax(autocorr[start:]) + start
        return sr / peak
    except Exception:
        return 0

def extract_features(audio_np, sr):
    pitch = extract_pitch(audio_np, sr)
    mfcc = librosa.feature.mfcc(y=audio_np, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    return np.hstack(([pitch], mfcc_mean)), pitch, mfcc

# --- Audio Processor ---
class AudioProcessor:
    def recv(self, frame: av.AudioFrame):
        samples = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
        sr = frame.sample_rate
        energy = np.mean(samples**2)

        if len(samples) > sr // 2:
            try:
                features, pitch, mfcc = extract_features(samples, sr)
                label = kmeans.predict(features.reshape(1, -1))[0]
                center_pitches = [c[0] for c in kmeans.cluster_centers_]
                male_label = np.argmin(center_pitches)
                gender = "Male" if label == male_label else "Female"
            except:
                gender = "Unclear"
                pitch = 0

            st.session_state.last_result = (gender, pitch)
            st.session_state.energy = energy
            st.session_state.waveform = samples
            st.session_state.mfcc = mfcc
            st.session_state.past_predictions.append(gender)

        return frame

# --- Start Stream ---
ctx = webrtc_streamer(
    key="gender",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

# --- Visuals ---
if ctx.state.playing:
    st.markdown(f"**🎚️ Mic Energy Level:** `{st.session_state.energy:.6f}`")

    result = st.session_state.last_result
    if result:
        gender, pitch = result
        st.markdown(f"**Pitch:** `{pitch:.2f} Hz`")
        if gender == "Unclear":
            st.warning("🎤 Speak louder or closer to the mic...")
        else:
            st.success(f"🧑 Predicted Gender: **{gender}**")

    # --- Waveform ---
    if st.session_state.waveform is not None:
        fig1, ax1 = plt.subplots(figsize=(6, 2))
        ax1.plot(st.session_state.waveform)
        ax1.set_title("Waveform")
        waveform_box = st.pyplot(fig1)

    # --- MFCCs ---
    if st.session_state.mfcc is not None:
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        librosa.display.specshow(st.session_state.mfcc, sr=44100, x_axis='time', ax=ax2)
        ax2.set_title("MFCCs")
        mfcc_box = st.pyplot(fig2)

    # --- History Chart ---
    male = sum(1 for g in st.session_state.past_predictions if g == "Male")
    female = sum(1 for g in st.session_state.past_predictions if g == "Female")
    unclear = sum(1 for g in st.session_state.past_predictions if g == "Unclear")

    fig3, ax3 = plt.subplots()
    ax3.bar(["Male", "Female", "Unclear"], [male, female, unclear], color=['blue', 'pink', 'gray'])
    ax3.set_ylim(0, 30)
    ax3.set_ylabel("Count (last 30 samples)")
    ax3.set_title("Gender Prediction History")
    chart_box = st.pyplot(fig3)
