import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
temp = pathlib.PosixPath = pathlib.WindowsPath

#Title
st.title('Classification model')


#rasm yuklash
file = st.file_uploader('Upload img', type=['jfif','png','jpg','jpeg','gif','svg'])
if file:
    st.image(file)
    #PIL convert
    img = PILImage.create(file)

    #model
    model = load_learner('Furniture_Model.pkl')

    # predict
    pred, pred_id, probs=model.predict(img)
    st.success(f"Prediction: {pred}")
    st.info(f'Probability: {probs[pred_id]*100:.1f}%')
     
    
    #Plotting 
    fig=px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
    
