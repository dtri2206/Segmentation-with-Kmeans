import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import requests
from io import BytesIO

st.title("Image Segmentation with kMeans")


# Load image from URL
def load_image_from_url(url):
    img_url = url
    img_pil = Image.open(requests.get(img_url, stream=True).raw)
    img = np.array(img_pil)
    return img

# Segment the image using kMeans
def segment_image(img, k=4):
    X = img.reshape(-1, img.shape[-1])
    kmeans = KMeans(n_clusters=4, n_init='auto', random_state=42)
    kmeans.fit(X)
    img_new = kmeans.cluster_centers_[kmeans.labels_]
    img_new = img_new.reshape(img.shape)
    img_new = np.uint8(img_new)
    return img_new

# Process and display images
col1, col2 = st.columns(2)
with col1:
        img_url = st.text_input("Image URL (Press Enter to apply)", value="")
        img = load_image_from_url(img_url)
        st.image(img, caption="Original Image", use_column_width=True)
with col2:
        K = st.slider("K", min_value=2, max_value=10, value=6)
        segmented_img = segment_image(img, K)
        st.image(segmented_img, caption="Segmented Image", use_column_width=True)