import streamlit as st

st.title("Microphone Test")

audio = st.audio_input(
    "Say: Hello, this is a microphone test",
    sample_rate=16000
)

if audio:
    st.success("Recording received.")

    st.audio(
        audio,
        format="audio/wav"
    )
