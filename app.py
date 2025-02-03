import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import requests
import zipfile

# GitHub URL for the model
github_url = "https://github.com/hvamsiprakash/finalyearproject/raw/main/image_captioning_model.zip"

# Function to download and extract the model
def download_and_extract_model():
    if not os.path.exists("image_captioning_model"):
        # Download the model zip file
        response = requests.get(github_url, stream=True)
        with open("image_captioning_model.zip", "wb") as file:
            for chunk in response.iter_content(chunk_size=128):
                file.write(chunk)

        # Extract the zip file
        with zipfile.ZipFile("image_captioning_model.zip", "r") as zip_ref:
            zip_ref.extractall()

        # Remove the zip file after extraction
        os.remove("image_captioning_model.zip")

# Load the model
def load_model():
    return tf.saved_model.load("image_captioning_model")

# Function to preprocess the image
def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to generate captions using the model
def generate_caption(image, model):
    # Preprocess the image
    img_embed = model.cnn_model(image, training=False)
    encoded_img = model.encoder(img_embed, training=False)

    # Initialize the decoded caption with the start token
    decoded_caption = "<start> "
    for i in range(23):  # Maximum caption length
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = model.decoder(tokenized_caption, encoded_img, training=False, mask=mask)
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = INDEX_TO_WORD[sampled_token_index]
        if sampled_token == "<end>":
            break
        decoded_caption += " " + sampled_token

    # Remove the start and end tokens
    decoded_caption = decoded_caption.replace("<start> ", "")
    decoded_caption = decoded_caption.replace(" <end>", "").strip()
    return decoded_caption

# Streamlit app
def main():
    st.title("Image Captioning with Streamlit")
    st.write("Upload an image, and the model will generate a caption for it.")

    # Download and extract the model
    download_and_extract_model()

    # Load the model
    model = load_model()

    # Load the vocabulary and vectorization layer
    global vectorization, INDEX_TO_WORD
    vectorization = model.vectorization
    vocab = vectorization.get_vocabulary()
    INDEX_TO_WORD = {idx: word for idx, word in enumerate(vocab)}

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Save the uploaded image temporarily
        temp_image_path = "temp_image.jpg"
        image.save(temp_image_path)

        # Preprocess the image
        image_tensor = preprocess_image(temp_image_path)

        # Generate caption
        if st.button("Generate Caption"):
            caption = generate_caption(image_tensor, model)
            st.write("Generated Caption:")
            st.success(caption)

        # Remove the temporary image file
        os.remove(temp_image_path)

# Run the app
if __name__ == "__main__":
    main()
