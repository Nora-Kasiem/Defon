import streamlit as st
from outline import *
import pages.camera as cam
from st_audiorec import st_audiorec
from transformers import pipeline



st.set_page_config(page_title="Deafon", page_icon="logo.png", layout="wide")

title()
languages()

# # Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

langs = {"Arabic": "ar", "English": "en", "Hindi": "hi"}

src = langs[st.session_state["from_lang"]]
dest = langs[st.session_state["to_lang"]]


types = ["Text", "Microphone", "Camera"]
t1, t2, t3 = st.tabs(types)

with t1: # Text Input
    st.text_area("Type your text here", key="input_text")
    with st.chat_message("assistant"):
        temp = cam.translating(st.session_state["input_text"], src, dest)
        st.markdown(temp)

with t2: # Audio Input
    wav_audio_data = st_audiorec()
    # st.text_area("Transcribed words goes here.....", disabled=True, key="transcribed_text")

    with st.chat_message("assistant"):
        if wav_audio_data is not None:
            pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
            text = pipe(wav_audio_data, generate_kwargs={"task": "translate", "language": dest})
            st.markdown(text)
            # st.session_state.transcribed_text = text
            # st.audio(wav_audio_data, format='audio/wav')

with t3: # Images Input
    # uploads =  
    st.file_uploader("Upload images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="file_uploader")
    
    with st.chat_message("assistant"):
        if st.session_state["file_uploader"] is not None:
            st.image(st.session_state["file_uploader"])
            cam.processing(st.session_state["file_uploader"], [src, dest])
            # st.markdown(st.session_state["file_uploader"])
            # st.write(translated_text)
        

