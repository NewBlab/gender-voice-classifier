import streamlit as st
import numpy as np
import librosa
from sklearn.cluster import KMeans
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import matplotlib.pyplot as plt
import time
import librosa.display

st.set_page_config(layout="centered")
st.title("🎙️ Real-Time Gender Detection from Microphone")

# Sample training data
sample_features = [
    [110] + list(np.random.normal(0, 1, 13)),
    [120] + list(np.random.normal(0, 1, 13)),
    [125] + list(np.random.normal(0, 1, 13)),
    [210] + list(np.random.normal(0, 1, 13)),
    [220] + list(np.random.normal(0, 1, 13)),
    [230] + list(np.random.normal(0, 1, 13)),
]
kmeans = KMeans(n_clusters=2, random_state=0).fit(sample_features)

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

class AudioProcessor:
    def __init__(self):
        self.result = None
        self.waveform = None
        self.pitch = 0
        self.mfcc = None
        self.energy = 0

    def recv(self, frame: av.AudioFrame):
        samples = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
        sr = frame.sample_rate
        self.energy = np.mean(samples**2)
        self.waveform = samples

        if len(samples) > sr // 2:
            try:
                features, pitch, mfcc = extract_features(samples, sr)
                self.pitch = pitch
                self.mfcc = mfcc

                if pitch > 50:
                    label = kmeans.predict(features.reshape(1, -1))[0]
                    center_pitches = [c[0] for c in kmeans.cluster_centers_]
                    male_label = np.argmin(center_pitches)
                    gender = "Male" if label == male_label else "Female"
                else:
                    gender = "Silent/Unclear"

                self.result = (gender, pitch)
            except Exception as e:
                self.result = ("Error", 0)

        return frame

ctx = webrtc_streamer(
    key="gender-detector",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

# UI placeholders
status = st.empty()
gender_box = st.empty()
pitch_box = st.empty()
energy_box = st.empty()
waveform_box = st.empty()
mfcc_box = st.empty()

if ctx.audio_processor:
    while ctx.state.playing:
        result = ctx.audio_processor.result
        waveform = ctx.audio_processor.waveform
        mfcc = ctx.audio_processor.mfcc
        pitch = ctx.audio_processor.pitch
        energy = ctx.audio_processor.energy

        energy_box.markdown(f"**🎚️ Mic Energy Level:** `{energy:.6f}`")

        if result:
            gender, pitch = result
            gender_box.markdown(f"### 🧑 Predicted Gender: **{gender}**")
            pitch_box.markdown(f"**Pitch:** `{pitch:.2f} Hz`")
        else:
            status.markdown("⏳ Waiting for voice input...")

        if waveform is not None:
            fig1, ax1 = plt.subplots(figsize=(6, 2))
            ax1.plot(waveform)
            ax1.set_title("Waveform")
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Amplitude")
            waveform_box.pyplot(fig1)

        if mfcc is not None:
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            librosa.display.specshow(mfcc, sr=44100, x_axis='time', ax=ax2)
            ax2.set_title("MFCCs")
            mfcc_box.pyplot(fig2)

        time.sleep(1)
