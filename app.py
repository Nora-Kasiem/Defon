import streamlit as st
from st_audiorec import st_audiorec
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from transformers import pipeline
import cv2

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

@st.cache_resource
def translating(text, src, dest):
    translator = pipeline("translation", model=f"Helsinki-NLP/opus-mt-{src}-{dest}")
    output = translator(text)[0]['translation_text']
    return output


def import_and_predict(image_data, model):
        size = (224, 224)
        image = Image.open(image_data)
        image = image.resize(size, Image.LANCZOS) 

        # img = np.expand_dims(img, 0)   
        # image = ImageOps.fit(image_data, size, Image.LANCZOS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # img = image[...,::-1]  
         
        img = np.expand_dims(image, 0)

        prediction = model.predict(img)
        return prediction


@st.cache_data()
def run_model(lang):
    model = tf.keras.models.load_model(f'./{lang}-model.h5')
    return model


@st.cache_resource
def audio_text():
    pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
    return pipe


types = ["Text", "Microphone", "Camera"]
t1, t2, t3 = st.tabs(types)

with t1: # Text Input
    intext = st.text_area("Type your text here", key="input_text")
    
    if intext is not None:
        # with st.chat_message("assistant"):
        temp = translating(intext, src, dest)
        st.markdown(temp)


with t2: # Audio Input
    wav_audio_data = st_audiorec()
    
    if wav_audio_data is not None:
        # with st.chat_message("assistant"):
        pipe = audio_text()
        # autext = pipe(wav_audio_data, generate_kwargs={"task": "transcribe"}) #, "language": dest})
        text = pipe(wav_audio_data, generate_kwargs={"task": "translate", "language": dest})
        st.markdown(text)

with t3: # Images Input  
    files = st.file_uploader("Upload images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="file_uploader")
    
    # st.set_option('deprecation.showfileUploaderEncoding', False)

    if files is not None:
        st.image(st.session_state["file_uploader"])
        
        string, letter = "", ""

        with st.spinner('Loading Model ...'):
            model = run_model(src)
        
            for image in files:
                
                predictions = import_and_predict(image, model)
                
                predicted_class_index = np.argmax(predictions[0])
                predicted_class_name = labels[predicted_class_index]
                
                st.write(f"""## Predicted Letter : {predicted_class_name}""")
    

