import streamlit as st
from st_audiorec import st_audiorec
from transformers import pipeline
from keras.models import load_model
import numpy as np
from skimage import transform
from PIL import Image
from transformers import pipeline




# Page Config
st.set_page_config(page_title="Deafon", page_icon="logo.png", layout="wide")

# Title
st.title("Deafon")
st.text("Sign Language Translator")

# Sidebar
languages = {"Arabic": "ar", "English": "en", "Hindi": "hi"}

st.sidebar.image("logo.png", width=150)
st.sidebar.selectbox("Source Language", languages.keys(), index=0, key="from_lang")
st.sidebar.selectbox("Target Language", languages.keys(), index=1, key="to_lang")

if st.session_state["from_lang"] == st.session_state["to_lang"]:
    st.error("Source and Target Languages cannot be the same")

# Arabic Letter Dict
labels = {0: 'ع',
 1: 'ال',
 2: 'ا',
 3: 'ب',
 4: 'د',
 5: 'ظ',
 6: 'ض',
 7: 'ف',
 8: 'ق',
 9: 'غ',
 10: 'ه',
 11: 'ح',
 12: 'ج',
 13: 'ك',
 14: 'خ',
 15: 'لا',
 16: 'ل',
 17: 'م',
 18: 'ن',
 19: 'ر',
 20: 'ص',
 21: 'س',
 22: 'ش',
 23: 'ط',
 24: 'ت',
 25: 'ث',
 26: 'ذ',
 27: 'ة',
 28: 'و',
 29: 'ئ',
 30: 'ي',
 31: 'ز'}


src = languages[st.session_state["from_lang"]]
dest = languages[st.session_state["to_lang"]]

def translating(text, src, dest):
    translator = pipeline("translation", model=f"Helsinki-NLP/opus-mt-{src}-{dest}")
    output = translator(text)[0]['translation_text']
    return output


types = ["Text", "Microphone", "Camera"]
t1, t2, t3 = st.tabs(types)

with t1: # Text Input
    intext = st.text_area("Type your text here", key="input_text")
    
    if intext is not None:
        with st.chat_message("assistant"):
            temp = translating(st.session_state["input_text"], src, dest)
            st.markdown(temp)


with t2: # Audio Input
    wav_audio_data = st_audiorec()

    with st.chat_message("assistant"):
        
        if wav_audio_data is not None:
            pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
            # autext = pipe(wav_audio_data, generate_kwargs={"task": "transcribe"}) #, "language": dest})
            text = pipe(wav_audio_data, generate_kwargs={"task": "translate", "language": dest})
            st.markdown(text)

with t3: # Images Input
    # uploads =  
    files = st.file_uploader("Upload images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="file_uploader")
    
    with st.chat_message("assistant"):
        if files is not None:
            st.image(st.session_state["file_uploader"])
            
            string, letter = "", ""
            
            model = load_model("arsl_cnn_model.h5")
            
            for i in files:
                img = np.array(Image.open(i))
                resized = transform.resize(img, (64, 64, 3))
                rescaled = transform.rescale(resized, (1./255))
                
                
                yhat = model.predict(rescaled)

                st.markdown(yhat)
            # st.markdown(st.session_state["file_uploader"])
            # st.write(translated_text)
        

