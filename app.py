import streamlit as st
import numpy as np
import librosa
from sklearn.cluster import KMeans
from pydub import AudioSegment
import io
from st_audiorec import st_audiorec  # from streamlit-audio-recorder

st.title("🎙️ Real-Time Gender Detection from Microphone")

# Record audio from mic
wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    st.audio(wav_audio_data, format='audio/wav')

    # Save recorded audio to a file-like object
    audio = AudioSegment.from_file(io.BytesIO(wav_audio_data), format="wav")
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    buffer.seek(0)

    def extract_pitch(signal, sr):
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        d = np.diff(autocorr)
        start = np.nonzero(d > 0)[0][0]
        peak = np.argmax(autocorr[start:]) + start
        pitch = sr / peak
        return pitch

    def extract_features(file_like):
        signal, sr = librosa.load(file_like, sr=None)
        pitch = extract_pitch(signal, sr)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        return np.hstack(([pitch], mfcc_mean))

    features = extract_features(buffer).reshape(1, -1)

    # Dummy training data for clustering
    sample_features = [
        [110] + list(np.random.normal(0, 1, 13)),
        [120] + list(np.random.normal(0, 1, 13)),
        [125] + list(np.random.normal(0, 1, 13)),
        [210] + list(np.random.normal(0, 1, 13)),
        [220] + list(np.random.normal(0, 1, 13)),
        [230] + list(np.random.normal(0, 1, 13)),
    ]

    kmeans = KMeans(n_clusters=2, random_state=0).fit(sample_features)
    label = kmeans.predict(features)[0]
    center_pitches = [center[0] for center in kmeans.cluster_centers_]
    male_label = np.argmin(center_pitches)
    gender = "Male" if label == male_label else "Female"

    st.success(f"Predicted Gender: **{gender}**")
    st.write(f"Pitch: {features[0][0]:.2f} Hz")
