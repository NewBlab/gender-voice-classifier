import streamlit as st
import numpy as np
import librosa
from sklearn.cluster import KMeans
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import matplotlib.pyplot as plt
import librosa.display
from collections import deque

st.set_page_config(layout="centered")
st.title("🎙️ In-Browser Gender Detection")

# — KMeans on dummy pitch+MFCC data
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

# — History
if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=30)
# track STOP event
if "was_playing" not in st.session_state:
    st.session_state.was_playing = False

def extract_pitch(y, sr):
    autoc = np.correlate(y, y, mode="full")[len(y):]
    d = np.diff(autoc)
    idx = np.where(d>0)[0]
    if not len(idx):
        return 0
    peak = np.argmax(autoc[idx[0]:]) + idx[0]
    return sr/peak if peak>0 else 0

def extract_features(y, sr):
    pitch = extract_pitch(y, sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.hstack(([pitch], np.mean(mfcc, axis=1))), pitch, mfcc

class AudioRecorder:
    def __init__(self):
        self.buffer = []  # collect numpy arrays

    def recv(self, frame: av.AudioFrame):
        arr = frame.to_ndarray().flatten().astype(np.float32)/32768.0
        self.buffer.append(arr)
        return frame

# start/stop recorder
rec_ctx = webrtc_streamer(
    key="recorder",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioRecorder,
    media_stream_constraints={"audio": True, "video": False},
)

# placeholders
status = st.empty()
waveform_box = st.empty()
mfcc_box = st.empty()
result_box = st.empty()
history_box = st.empty()

# detect STOP
if rec_ctx.audio_processor:
    playing = rec_ctx.state.playing
    # just stopped
    if st.session_state.was_playing and not playing:
        status.info("▶ Processing recording…")
        buf = np.concatenate(rec_ctx.audio_processor.buffer) if rec_ctx.audio_processor.buffer else np.array([])
        rec_ctx.audio_processor.buffer.clear()
        if buf.size and buf.shape[0]>1000:
            # process
            sr = rec_ctx.audio_receiver._config.media_stream_constraints["audio"]["sampleRate"] if False else 44100
            # librosa load expects file; we use default sr
            features, pitch, mfcc = extract_features(buf, sr)
            label = kmeans.predict([features])[0]
            gender = "Male" if label==male_label else "Female"
            st.session_state.history.append(gender)
            # display
            result_box.success(f"🧑 Predicted Gender: **{gender}** — 🎵 Pitch: **{pitch:.1f} Hz**")
            # waveform
            fig1,ax1=plt.subplots(figsize=(6,2))
            ax1.plot(buf); ax1.set(title="Waveform",xlabel="Sample",ylabel="Amplitude")
            waveform_box.pyplot(fig1)
            # MFCC
            fig2,ax2=plt.subplots(figsize=(6,3))
            librosa.display.specshow(mfcc, sr=sr, x_axis="time", ax=ax2)
            ax2.set(title="MFCCs")
            mfcc_box.pyplot(fig2)
        else:
            result_box.warning("⚠️ Recording too short or silent.")
        status.empty()
    st.session_state.was_playing = playing

# history chart
hist = list(st.session_state.history)
counts = [hist.count("Male"), hist.count("Female"), hist.count("Silent/Unclear")]
fig3,ax3=plt.subplots()
ax3.bar(["Male","Female","Silent"], counts, color=["blue","pink","gray"])
ax3.set_ylim(0,30)
ax3.set(title="Last 30 Predictions",ylabel="Count")
history_box.pyplot(fig3)
