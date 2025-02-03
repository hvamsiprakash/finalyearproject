import streamlit as st
import zipfile
import os
import requests
import tensorflow as tf
from PIL import Image
import numpy as np

# Function to download and extract the model ZIP
def download_and_extract_model(github_url, extract_to="image_captioning_model"):
    # Download the ZIP file
    response = requests.get(github_url)
    with open("image_captioning_model.zip", "wb") as f:
        f.write(response.content)

    # Extract the ZIP file
    with zipfile.ZipFile("image_captioning_model.zip", 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Load the model
def load_model_from_directory(model_dir="image_captioning_model"):
    return tf.keras.models.load_model(model_dir)

# Preprocess the image for model input (resize and normalize)
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)  # Resize image to match the model's expected input size
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to generate a caption (this will depend on your model)
def predict_caption(model, preprocessed_image):
    # Implement caption prediction logic based on your model
    caption = "This is a placeholder caption"  # Replace with actual prediction logic
    return caption

# GitHub URL of the model ZIP
github_url = "https://github.com/hvamsiprakash/finalyearproject/raw/main/image_captioning_model.zip"

# Download and extract model
download_and_extract_model(github_url)

# Load the model
model = load_model_from_directory()

# Now you can use the model for predictions in your Streamlit app
st.title("Image Captioning")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    
    # Preprocess the image as needed (resize, normalize, etc.)
    preprocessed_image = preprocess_image(image)
    
    # Predict the caption for the image
    caption = predict_caption(model, preprocessed_image)
    st.write("Predicted Caption: ", caption)
