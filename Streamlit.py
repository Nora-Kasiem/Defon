import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from skimage import transform
import PIL
from googletrans import Translator


# Load your pre-trained model here (replace 'path/to/your/model.h5')
model = load_model('models\ArSLText.h5')
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
 31: 'ز'} # Map your model output to English letters or words

st.title("Sign Language Translator")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = np.array(PIL.Image.open(uploaded_file))
    resized_image = transform.resize(image, (256, 256)) # Resize the input image if needed
    np.expand_dims(resized_image, 0)
    yhat = model.predict(np.expand_dims(resized_image/255, 0))

    result = np.where(yhat[0] == np.amax(yhat[0]))

    letter = labels[result[0][0]]

    translated = Translator.translate(letter, src='ar', dest='en')
    
    st.text_area(f"Letter: {letter}, Translated:{translated}")