import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

model = load_model('model.h5')

st.header('Hotdog/Not-Hotdog')

img_file = st.file_uploader('Upload image:', type=['png', 'jpg', 'jpeg'])


if img_file:
    SIZE = (250, 250)
    test_image = Image.open(img_file)
    test_image_resized = ImageOps.fit(test_image, SIZE, Image.ANTIALIAS)
    test_image_prepared = np.asarray(test_image_resized)
    image_reshaped = test_image_prepared.reshape((1, test_image_prepared.shape[0], test_image_prepared.shape[1], test_image_prepared.shape[2]))
    st.image(image_reshaped)

    pred = model.predict(image_reshaped)

    st.write(pred)

    hotdog = 0

    if pred > .5:
        hotdog = 1

    result_dict = {
        1:'hotdog',
        0:'not-hotdog'
    }
    result = f'This image is a {result_dict[hotdog]}'
    st.success(result)





#def predict_image(im, model):
#size = (250, 250)
#image = ImageOps.fit(im, size)
#img = np.asarray(image).flatten()
#prediction = model.predict(img)
#return prediction


#if img_file != None:
    #image = Image.open(img_file)
    #st.image(image)
    #pred = predict_image(image, model)
    #print(pred)
    #prediction = predict_image(image, model)
    #class_names = ['hotdog', 'not-hotdog']
    #result = f'This image is a {class_names[prediction]}'
    #st.success(result)



#img_to_array(img)
#print(type(img))
#plt.imshow(img, cmap='gray')

#model = load_model('first_model_anel')

#model.predict(img)
