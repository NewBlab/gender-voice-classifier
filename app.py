import streamlit as st
import numpy as np
import librosa
from sklearn.cluster import KMeans
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import matplotlib.pyplot as plt
import librosa.display
from collections import deque

# Page setup
st.set_page_config(layout="centered")
st.title("🎙️ In-Browser Gender Detection")

# Initialize session state
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

# Dummy KMeans model for pitch+MFCC
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

# Feature extraction functions
def extract_pitch(y, sr):
    autoc = np.correlate(y, y, mode='full')[len(y):]
    d = np.diff(autoc)
    idx = np.where(d > 0)[0]
    if not len(idx):
        return 0
    peak = np.argmax(autoc[idx[0]:]) + idx[0]
    return sr/peak if peak > 0 else 0

def extract_mfcc(y, sr):
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# AudioRecorder collects frames while recording\class AudioRecorder:
    def __init__(self):
        self.buffer = []
    def recv(self, frame: av.AudioFrame):
        arr = frame.to_ndarray().flatten().astype(np.float32)/32768.0
        if st.session_state.is_recording:
            self.buffer.append(arr)
        return frame

# Start the audio streamer
ctx = webrtc_streamer(
    key='recorder',
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioRecorder,
    media_stream_constraints={"audio": True, "video": False},
)

# Mic status indicator\mic_status = st.empty()
if st.session_state.is_recording:
    mic_status.markdown("🔴 **Recording...** Speak now.")
else:
    mic_status.markdown("⚪ **Idle**")

# Control buttons
if not st.session_state.is_recording:
    if st.button("Start Recording"):
        st.session_state.is_recording = True
        if ctx.audio_processor:
            ctx.audio_processor.buffer.clear()
else:
    if st.button("Stop & Analyze"):
        st.session_state.is_recording = False
        # concatenate and process
        data = np.concatenate(ctx.audio_processor.buffer) if ctx.audio_processor and ctx.audio_processor.buffer else np.array([])
        if ctx.audio_processor:
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
            st.session_state.last_result = ("Too short or no audio", 0)

# Display results
if st.session_state.last_result:
    gender, pitch = st.session_state.last_result
    if gender in ["Male", "Female"]:
        st.success(f"🧑 Predicted Gender: **{gender}** — 🎵 Pitch: **{pitch:.1f} Hz**")
    else:
        st.warning(gender)
    # Waveform
    if st.session_state.last_waveform is not None:
        fig1, ax1 = plt.subplots(figsize=(6,2))
        ax1.plot(st.session_state.last_waveform)
        ax1.set(title="Waveform", xlabel="Sample", ylabel="Amplitude")
        st.pyplot(fig1)
    # MFCC
    if st.session_state.last_mfcc is not None:
        fig2, ax2 = plt.subplots(figsize=(6,3))
        librosa.display.specshow(st.session_state.last_mfcc, sr=sr, x_axis='time', ax=ax2)
        ax2.set(title="MFCCs")
        st.pyplot(fig2)

# History chart
hist = list(st.session_state.history)
male_count = hist.count("Male")
female_count = hist.count("Female")
unclear_count = hist.count("Too short or no audio")
fig3, ax3 = plt.subplots()
ax3.bar(["Male","Female","No Audio"], [male_count, female_count, unclear_count], color=["blue","pink","gray"])
ax3.set_ylim(0,30)
ax3.set(title="Prediction History (last 30)", ylabel="Count")
st.pyplot(fig3)
