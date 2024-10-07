# METEHAN AYHAN

import streamlit as st
import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from sklearn.neighbors import NearestNeighbors
from PIL import Image
from numpy.linalg import norm

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model = tf.keras.Sequential([
    base_model,
    GlobalMaxPool2D()
])
model.trainable = False

#dosyaları yükleyelim
image_features = pkl.load(open('Images_features.pkl', 'rb'))
filenames = pkl.load(open('filenames.pkl', 'rb'))

# KNN modeli
neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
neighbors.fit(image_features)

# Resim özelliklerini çıkaran fonk.
def extract_features_from_images(image_path, model):
    img = image.load_img(image_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()  
    norm_result = result / norm(result)  # Normalizasyon 
    return norm_result

st.title("Fashion Product Recommendation System - Metehan Ayhan")

uploaded_file = st.file_uploader("Lütfen bir moda ürünü resmi yükleyin...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Kullanıcının yüklediği resmi ekranda göserelim
    st.image(uploaded_file, caption="Yüklenen Resim", use_column_width=True)

    # Yüklenen resmi kaydedelim
    img = Image.open(uploaded_file)
    img_path = "uploaded_image.jpg"
    img.save(img_path)

    input_image_features = extract_features_from_images(img_path, model)

    # En yakın 5 resim
    distances, indices = neighbors.kneighbors([input_image_features])

    st.write("Benzer ürünler öneriliyor...")

    col1, col2, col3, col4, col5 = st.columns(5)

    # İlk 5 benzer resim
    with col1:
        st.image(filenames[indices[0][0]], caption="1. Öneri", use_column_width=True)
    with col2:
        st.image(filenames[indices[0][1]], caption="2. Öneri", use_column_width=True)
    with col3:
        st.image(filenames[indices[0][2]], caption="3. Öneri", use_column_width=True)
    with col4:
        st.image(filenames[indices[0][3]], caption="4. Öneri", use_column_width=True)
    with col5:
        st.image(filenames[indices[0][4]], caption="5. Öneri", use_column_width=True)
