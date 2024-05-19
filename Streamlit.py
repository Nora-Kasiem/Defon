import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from skimage import transform
from PIL import Image
from googletrans import Translator


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

tran = Translator()

# Load your pre-trained model here
model = load_model('ArSLText_GRAY.h5')
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

st.title("Sign Language Translator")

# Accept user input
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png"])

# img = PIL.Image.open(uploaded_file)

if uploaded_file is not None:
    # Add user message to chat history
    st.image(uploaded_file)
    st.session_state.messages.append({"role": "user", "content": uploaded_file.name})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(uploaded_file.name)
    
    with st.chat_message("assistant"):
        # stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        
        # img = np.load(uploaded_file.getvalue())
        # st.write(img)

        img = np.array(Image.open(uploaded_file))
        resized_image = transform.resize(img, (256, 256)) # Resize the input image if needed
        np.expand_dims(resized_image, 0)
        yhat = model(np.expand_dims(resized_image/255, 0))

        result = np.where(yhat[0] == np.amax(yhat[0]))

        letter = labels[result[0][0]]

        st.markdown(letter)

        translated = tran.translate(letter)#, src='ar', dest='en')
        
        # response = f"Letter: {letter}, Translated:{translated}"

    st.markdown(translated)
    # st.session_state.messages.append({"role": "assistant", "content": response})
    