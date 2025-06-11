import streamlit as st
import numpy as np
import librosa
from sklearn.cluster import KMeans
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import matplotlib.pyplot as plt
import librosa.display
from collections import deque
import soundfile as sf

# --- Page config ---
st.set_page_config(layout="centered")
st.title("🎙️ In-Browser Gender Detection with Manual Control")

# --- Initialize session state ---
if 'buffer' not in st.session_state:
    st.session_state.buffer = []
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'history' not in st.session_state:
    st.session_state.history = deque(maxlen=30)
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'last_waveform' not in st.session_state:
    st.session_state.last_waveform = None
if 'last_mfcc' not in st.session_state:
    st.session_state.last_mfcc = None

# --- Dummy KMeans model ---
sample_feats = [
    [110] + list(np.random.randn(13)),
    [120] + list(np.random.randn(13)),
    [125] + list(np.random.randn(13)),
    [210] + list(np.random.randn(13)),
    [220] + list(np.random.randn(13)),
    [230] + list(np.random.randn(13)),
]
kmeans = KMeans(n_clusters=2, random_state=0).fit(sample_feats)
centers = [c[0] for c in kmeans.cluster_centers_]
male_label = int(np.argmin(centers))

# --- Feature extraction functions ---
def extract_pitch(signal, sr):
    autoc = np.correlate(signal, signal, mode='full')[len(signal):]
    d = np.diff(autoc)
    idx = np.where(d > 0)[0]
    if not len(idx):
        return 0
    peak = np.argmax(autoc[idx[0]:]) + idx[0]
    return sr/peak if peak > 0 else 0


def extract_mfcc(signal, sr):
    return librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)

# --- Audio processor collecting frames ---
class AudioRecorder:
    def __init__(self):
        self.buffer = []
    def recv(self, frame: av.AudioFrame):
        samples = frame.to_ndarray().flatten().astype(np.float32)/32768.0
        if st.session_state.is_recording:
            self.buffer.append(samples)
        return frame

# --- WebRTC streamer (always running) ---
ctx = webrtc_streamer(
    key='recorder',
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioRecorder,
    media_stream_constraints={"audio": True, "video": False},
)

# --- Control buttons ---
col1, col2 = st.columns(2)
with col1:
    if not st.session_state.is_recording and st.button("Start Recording"):
        st.session_state.buffer = []
        st.session_state.is_recording = True
        st.session_state.last_result = None
with col2:
    if st.session_state.is_recording and st.button("Stop & Analyze"):
        st.session_state.is_recording = False
        # concatenate buffer
        data = np.concatenate(ctx.audio_processor.buffer) if ctx.audio_processor.buffer else np.array([])
        ctx.audio_processor.buffer.clear()
        if data.size > 1000:
            sr = 44100
            pitch = extract_pitch(data, sr)
            mfcc = extract_mfcc(data, sr)
            feat = [pitch] + list(np.mean(mfcc, axis=1))
            label = kmeans.predict([feat])[0]
            gender = "Male" if label == male_label else "Female"
            st.session_state.last_result = (gender, pitch)
            st.session_state.last_waveform = data
            st.session_state.last_mfcc = mfcc
            st.session_state.history.append(gender)
        else:
            st.session_state.last_result = ("Too short", 0)

# --- Display results ---
if st.session_state.last_result:
    gender, pitch = st.session_state.last_result
    if gender in ["Male","Female"]:
        st.success(f"🧑 Predicted Gender: **{gender}** — 🎵 Pitch: **{pitch:.1f} Hz**")
    else:
        st.warning(gender)

    if st.session_state.last_waveform is not None:
        fig1, ax1 = plt.subplots(figsize=(6,2))
        ax1.plot(st.session_state.last_waveform)
        ax1.set(title="Waveform", xlabel="Sample", ylabel="Amplitude")
        st.pyplot(fig1)

    if st.session_state.last_mfcc is not None:
        fig2, ax2 = plt.subplots(figsize=(6,3))
        librosa.display.specshow(st.session_state.last_mfcc, sr=sr, x_axis='time', ax=ax2)
        ax2.set(title="MFCCs")
        st.pyplot(fig2)

# --- History chart ---
hist = list(st.session_state.history)
male_count = hist.count("Male")
female_count = hist.count("Female")
unclear_count = hist.count("Too short")
fig3, ax3 = plt.subplots()
ax3.bar(["Male","Female","Too short"], [male_count, female_count, unclear_count], color=["blue","pink","gray"])
ax3.set_ylim(0,30)
ax3.set(title="Prediction History (last 30)", ylabel="Count")
st.pyplot(fig3)
