import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests
import zipfile

# Function to download and unzip the model from GitHub
def download_and_unzip_model():
    model_url = "https://github.com/hvamsiprakash/finalyearproject/raw/main/image_captioning_model.zip"
    model_zip_path = "image_captioning_model.zip"
    model_dir = "image_captioning_model"

    # Download the zip file
    if not os.path.exists(model_zip_path):
        st.write("Downloading model...")
        response = requests.get(model_url)
        with open(model_zip_path, "wb") as f:
            f.write(response.content)
        st.write("Download complete.")

    # Unzip the model
    if not os.path.exists(model_dir):
        st.write("Unzipping model...")
        with zipfile.ZipFile(model_zip_path, "r") as zip_ref:
            zip_ref.extractall(model_dir)
        st.write("Unzip complete.")

# Load the model and vectorization layer
@st.cache_resource  # Use st.cache_resource for loading models
def load_model():
    download_and_unzip_model()  # Download and unzip the model
    model = tf.saved_model.load("image_captioning_model")
    return model


# Function to load the vocabulary
@st.cache_resource  # Use st.cache_resource for caching the loaded vocabulary
def load_vectorization():
    # Load the vocabulary from the JSON file
    with open('vocab.json', 'r') as f:
        vocab = json.load(f)

    # Initialize the TextVectorization layer
    vectorization = TextVectorization(
        max_tokens=13000,
        output_sequence_length=24,
        standardize=custom_standardization
    )

    # Set the vocabulary
    vectorization.set_vocabulary(vocab)
    return vectorization

model = load_model()
vectorization = load_vectorization()

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((299, 299))
    image = np.array(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.expand_dims(image, 0)
    return image

# Function to generate caption
def generate_caption(image):
    image = preprocess_image(image)
    img_embed = model.cnn_model(image)
    encoded_img = model.encoder(img_embed, training=False)
    decoded_caption = "<start> "
    for i in range(23):  # SEQ_LENGTH - 1
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = model.decoder(tokenized_caption, encoded_img, training=False, mask=mask)
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = vectorization.get_vocabulary()[sampled_token_index]
        if sampled_token == "<end>":
            break
        decoded_caption += " " + sampled_token
    decoded_caption = decoded_caption.replace("<start> ", "")
    decoded_caption = decoded_caption.replace(" <end>", "").strip()
    return decoded_caption

# Streamlit app
st.title("Image Captioning App")
st.write("Upload an image and the model will generate a caption for it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Generating caption...")
    caption = generate_caption(image)
    st.write(f"**Generated Caption:** {caption}")
