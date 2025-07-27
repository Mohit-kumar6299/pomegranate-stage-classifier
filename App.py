import streamlit as st
#import timm
#import torch
from PIL import Image
import numpy as np
from torchvision import transforms

# Define class labels from the dataset
CLASSES = ['bud', 'flower', 'early-fruit', 'mid-growth', 'ripe']

# Load the fine-tuned Tiny ViT model
@st.cache_resource
def load_model():
    model = timm.create_model('tiny_vit_5m_224', pretrained=False, num_classes=len(CLASSES))
    # Load the fine-tuned weights (adjust the path to your .pth file)
    state_dict = torch.load('pomegranate_mobilenet_best_model.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Load class labels
@st.cache_data
def load_labels():
    return np.array(CLASSES)

# Image preprocessing function
def preprocess_image(image):
    # Use the same transforms as in the notebook for inference
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Load model and labels
model = load_model()
labels = load_labels()

# Streamlit app interface
st.title("Pomegranate Stage Classifier")
st.write("Upload an image to classify the pomegranate stage using a fine-tuned Tiny ViT (5M, 224x224) model.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("Classifying...")

    # Preprocess and predict
    with torch.no_grad():
        input_tensor = preprocess_image(image)
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0] * 100
        top5_prob, top5_idx = torch.topk(probabilities, 5)

    # Display predictions
    st.subheader("Top Predictions:")
    for i in range(len(top5_prob)):
        st.write(f"{labels[top5_idx[i]]}: {top5_prob[i]:.2f}%")
