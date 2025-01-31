# app.py

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests
import zipfile

# Function to download and extract the model zip file from GitHub
def download_and_extract_model():
    github_url = "https://github.com/hvamsiprakash/finalyearproject/raw/main/image_captioning_model.zip"
    model_folder_name = "image_captioning_model"  # Name of the model folder

    # Check if the model folder already exists
    if os.path.exists(model_folder_name):
        st.write("Model folder already exists. Skipping download.")
        return

    # Download the zip file
    st.write("Downloading model from GitHub...")
    response = requests.get(github_url)
    with open("image_captioning_model.zip", "wb") as f:
        f.write(response.content)

    # Extract the zip file
    st.write("Extracting model...")
    with zipfile.ZipFile("image_captioning_model.zip", "r") as zip_ref:
        zip_ref.extractall(".")

    # Clean up: Remove the zip file after extraction
    os.remove("image_captioning_model.zip")
    st.write("Model downloaded and extracted successfully.")

# Load the saved model
@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_model():
    # Ensure the model folder is downloaded
    if not os.path.exists("image_captioning_model"):
        download_and_extract_model()

    # Load the model
    model = tf.saved_model.load("image_captioning_model")
    return model

# Load the vocabulary and create INDEX_TO_WORD mapping
@st.cache_data  # Cache the vocabulary to avoid reloading on every interaction
def load_vocabulary():
    # Assuming you saved the vectorization layer's vocabulary as a text file
    with open("vocab.txt", "r") as f:
        vocab = f.read().splitlines()
    INDEX_TO_WORD = {idx: word for idx, word in enumerate(vocab)}
    return INDEX_TO_WORD

# Preprocess the image
def preprocess_image(image):
    image = image.resize((299, 299))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Generate caption using the model
def generate_caption(model, image, INDEX_TO_WORD, max_length=SEQ_LENGTH - 1):
    # Pass the image through the CNN to get features
    img_features = model.cnn_model(image, training=False)

    # Encode the image features using the encoder
    encoded_img = model.encoder(img_features, training=False)

    # Initialize the decoded caption with the start token
    decoded_caption = "<start> "

    # Loop to generate the caption word by word
    for i in range(max_length):
        # Tokenize the current decoded caption
        tokenized_caption = vectorization([decoded_caption])[:, :-1]

        # Create a mask to ignore padding tokens
        mask = tf.math.not_equal(tokenized_caption, 0)

        # Generate predictions for the next token
        predictions = model.decoder(tokenized_caption, encoded_img, training=False, mask=mask)

        # Select the token with the highest probability
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = INDEX_TO_WORD[sampled_token_index]

        # If the end token is generated, break the loop
        if sampled_token == "<end>":
            break

        # Append the sampled token to the decoded caption
        decoded_caption += " " + sampled_token

    # Remove the start token and trim any trailing spaces or end token
    decoded_caption = decoded_caption.replace("<start> ", "")
    decoded_caption = decoded_caption.replace(" <end>", "").strip()

    return decoded_caption

# Streamlit app
def main():
    st.title("Image Captioning with Streamlit")
    st.write("Upload an image, and the model will generate a caption for it.")

    # Load the model and vocabulary
    model = load_model()
    INDEX_TO_WORD = load_vocabulary()

    # File uploader for the image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        image = preprocess_image(image)

        # Generate caption
        if st.button("Generate Caption"):
            caption = generate_caption(model, image, INDEX_TO_WORD)
            st.write("**Generated Caption:**")
            st.success(caption)

# Run the app
if __name__ == "__main__":
    main()
