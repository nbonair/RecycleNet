import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from recnet import RecycleNet
detector = RecycleNet()
# Set the title of your app
st.set_page_config(
    page_title = "Recycle Trash Detector",
    page_icon= ":teacher:",
)
st.title("Recycling Contamination Detector")

# Create a sidebar section for image upload
st.sidebar.title("Upload Image")
image_uploaded_file = st.sidebar.file_uploader("Upload recycle items", type=["jpg", "jpeg", "png"])

# Check if an image file is uploaded
if image_uploaded_file is not None:
    # Read the image file
    image = Image.open(image_uploaded_file).resize((224,224))
    
    # Display the uploaded image
    st.sidebar.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Process the image with the object detector (replace with your own object detection code)
    labels = detector.predict(image)
    if labels == "trash":
        labels = "Non Recyclable"
    else:
        labels = "Recyclable"
    # Display the processed image on the main page
    st.image(image, caption="Processed Image",width=300)
    st.write(f"Detected Items: {labels}")
    

