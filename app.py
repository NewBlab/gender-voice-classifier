import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import librosa
from sklearn.cluster import KMeans
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import matplotlib.pyplot as plt
import librosa.display
from collections import deque

# ── 1) MIC DIAGNOSTIC HTML ────────────────────────────────────────────────────
components.html("""
<!DOCTYPE html>
<html>
  <body>
    <div id="mic-status">Mic test: <em>pending…</em></div>
    <script>
      navigator.mediaDevices.getUserMedia({audio:true})
        .then(stream => {
          document.getElementById('mic-status').innerText = 'Mic test: OK 🎤';
          stream.getTracks().forEach(t => t.stop());
        })
        .catch(err => {
          document.getElementById('mic-status').innerText = 'Mic test: FAILED – ' + err.message;
        });
    </script>
  </body>
</html>
""", height=60)

# ── 2) PAGE SETUP ─────────────────────────────────────────────────────────────
st.set_page_config(layout="centered")
st.title("🎙️ In-Browser Gender Detection")

ss = st.session_state
if "is_recording" not in ss:
    ss.is_recording = False
if "history" not in ss:
    ss.history = deque(maxlen=30)
if "last_result" not in ss:
    ss.last_result = None
if "last_waveform" not in ss:
    ss.last_waveform = None
if "last_mfcc" not in ss:
    ss.last_mfcc = None

# ── 3) DUMMY KMEANS ───────────────────────────────────────────────────────────
def train_dummy_kmeans():
    feats = []
    for base in [110,120,125,210,220,230]:
        feats.append([base] + list(np.random.randn(13)))
    return KMeans(n_clusters=2, random_state=0).fit(feats)

kmeans = train_dummy_kmeans()
centers = [c[0] for c in kmeans.cluster_centers_]
male_label = int(np.argmin(centers))

# ── 4) FEATURE EXTRACTORS ─────────────────────────────────────────────────────
def extract_pitch(y, sr):
    autoc = np.correlate(y, y, mode="full")[len(y):]
    d = np.diff(autoc)
    idx = np.where(d>0)[0]
    if not idx.size:
        return 0
    peak = np.argmax(autoc[idx[0]:]) + idx[0]
    return sr/peak if peak>0 else 0

def extract_mfcc(y, sr):
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# ── 5) AUDIO RECORDER ─────────────────────────────────────────────────────────
class AudioRecorder:
    def __init__(self):
        self.buffer = []
    def recv(self, frame: av.AudioFrame):
        samples = frame.to_ndarray().flatten().astype(np.float32)/32768.0
        if ss.is_recording:
            self.buffer.append(samples)
        return frame

# ── 6) START WEBRTC ───────────────────────────────────────────────────────────
ctx = webrtc_streamer(
    key="recorder",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioRecorder,
    media_stream_constraints={"audio": True, "video": False},
)

# ── 7) MIC ENERGY METER ───────────────────────────────────────────────────────
playing = ctx.state.playing
st.text(f"WebRTC active? {playing}")
energy = 0.0
if ctx.audio_processor and ctx.audio_processor.buffer:
    last = ctx.audio_processor.buffer[-1]
    energy = float(np.mean(last**2))
st.metric("Mic energy", f"{energy:.6f}")

# ── 8) CONTROL BUTTONS ────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    if not ss.is_recording and st.button("Start Recording"):
        ss.is_recording = True
        if ctx.audio_processor:
            ctx.audio_processor.buffer.clear()
with col2:
    if ss.is_recording and st.button("Stop & Analyze"):
        ss.is_recording = False
        # gather buffer
        rec = ctx.audio_processor
        buf = np.concatenate(rec.buffer) if rec and rec.buffer else np.array([])
        if rec:
            rec.buffer.clear()
        # require ≥0.5s
        if buf.size > 44100*0.5:
            sr = 44100
            pitch = extract_pitch(buf, sr)
            mfcc = extract_mfcc(buf, sr)
            feat = [pitch] + list(np.mean(mfcc, axis=1))
            label = kmeans.predict([feat])[0]
            gender = "Male" if label==male_label else "Female"
            ss.last_result = (gender, pitch)
            ss.last_waveform = buf
            ss.last_mfcc = mfcc
            ss.history.append(gender)
        else:
            ss.last_result = ("Too short", 0)

# ── 9) SHOW RESULTS ───────────────────────────────────────────────────────────
if ss.last_result:
    gender, pitch = ss.last_result
    if gender in ("Male","Female"):
        st.success(f"🧑 Predicted Gender: **{gender}** — 🎵 {pitch:.1f} Hz")
    else:
        st.warning(gender)
    # waveform
    if ss.last_waveform is not None:
        fig1,ax1 = plt.subplots(figsize=(6,2))
        ax1.plot(ss.last_waveform)
        ax1.set(title="Waveform", xlabel="Sample", ylabel="Amplitude")
        st.pyplot(fig1)
    # mfcc
    if ss.last_mfcc is not None:
        fig2,ax2 = plt.subplots(figsize=(6,3))
        librosa.display.specshow(ss.last_mfcc, sr=44100, x_axis="time", ax=ax2)
        ax2.set(title="MFCCs")
        st.pyplot(fig2)

# ── 10) HISTORY CHART ─────────────────────────────────────────────────────────
hist = list(ss.history)
counts = [hist.count("Male"), hist.count("Female"), hist.count("Too short")]
fig3,ax3 = plt.subplots()
ax3.bar(["Male","Female","Short"], counts, color=["blue","pink","gray"])
ax3.set_ylim(0,30)
ax3.set(title="Last 30 Predictions", ylabel="Count")
st.pyplot(fig3)
