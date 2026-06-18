import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import joblib

model = load_model('emotion_model.h5')
le = joblib.load('label_encoder.pkl')

N_MFCC = 40
MAX_PAD_LEN = 174
SAMPLE_RATE = 22050

def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    if mfcc.shape[1] < MAX_PAD_LEN:
        pad_width = MAX_PAD_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_PAD_LEN]
    return mfcc

st.set_page_config(page_title="Emotion Recognition", layout="centered")
st.title("🎤 Emotion Recognition from Speech")
st.write("Upload a `.wav` file and the model will predict the emotion.")

uploaded_file = st.file_uploader("Choose a WAV file", type=['wav'])
if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    audio, sr = librosa.load(uploaded_file, sr=SAMPLE_RATE)
    mfcc = extract_features(audio, sr)
    mfcc_input = mfcc.T[np.newaxis, ...]   # (1, time_steps, features)
    pred_probs = model.predict(mfcc_input)
    pred_idx = np.argmax(pred_probs)
    emotion = le.inverse_transform([pred_idx])[0]
    confidence = np.max(pred_probs)
    st.success(f"**Predicted Emotion:** {emotion}  (confidence: {confidence:.2%})")
