import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ---- Class order (alphabetical, matches ImageFolder) ----
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)
    return model

@st.cache_resource  # loads the model ONCE, not on every user interaction
def load_model():
    model = build_model()
    state_dict = torch.load("brain_tumor_model.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

model = load_model()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

def predict(image):
    image = image.convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
    return {CLASS_NAMES[i]: float(probabilities[i]) for i in range(4)}


# ---------------- UI starts here ----------------
st.title("Brain Tumor MRI Classifier")
st.write(
    "Upload an MRI scan to classify it as glioma, meningioma, pituitary tumor, or no tumor. "
    "Built with a fine-tuned ResNet18."
)
st.warning("⚠️ Student portfolio project — NOT a diagnostic tool. Do not use for real medical decisions.")

uploaded_file = st.file_uploader("Upload a brain MRI scan", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded scan", use_container_width=True)

    with st.spinner("Running model..."):
        results = predict(image)

    sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    top_class = next(iter(sorted_results))
    top_confidence = sorted_results[top_class]

    st.subheader(f"Prediction: {top_class} ({top_confidence:.1%} confidence)")
    st.bar_chart(sorted_results)