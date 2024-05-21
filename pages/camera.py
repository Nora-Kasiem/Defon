import streamlit as st
import numpy as np
from keras.models import load_model
from skimage import transform
from PIL import Image
from transformers import pipeline


# Letters Map to model output
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


# if files is not None:
    # Add user message to chat history
    # st.image(files)
    # st.session_state.messages.append({"role": "user", "content": "images"})
    
    # Display user message in chat message container
    # with st.chat_message("user"):
    #     st.markdown("images")
    # st.markdown(string)
    
    # output_text = translating(string, lang[0], lang[1])
    # st.markdown(output_text)
    # st.session_state.messages.append({"role": "assistant", "content": output_text})
    
def loading():
    # Load your pre-trained model here
    model = load_model("arsl_cnn_model.h5")
    return model


def translating(text, src, dest):
    translator = pipeline("translation", model=f"Helsinki-NLP/opus-mt-{src}-{dest}")
    output = translator(text)[0]['translation_text']
    return output


def processing(files, lang): 
            
    string, letter = "", ""
    model = loading()
    
    for i in files:
        img = np.array(Image.open(i))
        resized = transform.resize(img, (64, 64, 3))
        rescaled = resized.rescale(1./255)
        
        
        yhat = model.predict(rescaled)

        st.markdown(yhat)

    #     result = np.where(yhat[0] == np.amax(yhat[0]))

    #     letter = labels[result[0][0]]

    #     string += letter

    # translated = translating(string, lang[0], lang[1])
    
    # return 