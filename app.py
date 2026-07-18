import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image

# ---- Exact class order from os.listdir("data/train") at training time ----
CLASS_NAMES = [
    'Acne And Rosacea Photos', 'Atopic Dermatitis Photos', 'Ba  Cellulitis',
    'Ba Impetigo', 'Benign', 'Bullous Disease Photos',
    'Cellulitis Impetigo And Other Bacterial Infections', 'Eczema Photos',
    'Exanthems And Drug Eruptions', 'Fu Athlete Foot', 'Fu Nail Fungus',
    'Fu Ringworm', 'Heathy', 'Systemic Disease', 'Urticaria Hives',
    'Vascular Tumors', 'Vasculitis Photos', 'Vi Chickenpox', 'Vi Shingles',
    'Warts Molluscum And Other Viral Infections'
]
NUM_CLASSES = len(CLASS_NAMES)  # 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    # pretrained=False: we're loading YOUR trained weights, not the original ImageNet ones
    model = timm.create_model("convnext_tiny", pretrained=False, num_classes=NUM_CLASSES)
    state_dict = torch.load("skin_disease_model.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

model = load_model()

# Matches train_loop.py's test_transform exactly (no augmentation at inference time)
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
    return {CLASS_NAMES[i]: float(probabilities[i]) for i in range(NUM_CLASSES)}


# ---------------- UI ----------------
st.title("Skin Disease Classifier")
st.write(
    "Upload a photo of a skin condition to get the model's top predictions "
    "across 20 possible categories."
)
st.warning("⚠️ Student portfolio project — NOT a diagnostic tool. Do not use for real medical decisions.")

uploaded_file = st.file_uploader("Upload a skin photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded photo", use_container_width=True)

    with st.spinner("Running model..."):
        results = predict(image)

    sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    top_class = next(iter(sorted_results))
    top_confidence = sorted_results[top_class]

    st.subheader(f"Top prediction: {top_class} ({top_confidence:.1%} confidence)")

    # Show only the top 5 as a chart — 20 bars at once gets cluttered
    top5 = dict(list(sorted_results.items())[:5])
    st.bar_chart(top5)