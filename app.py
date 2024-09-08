import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch
from torch import nn, optim
from torchvision import transforms, models

class ActionRecognitionClassifier(nn.Module):
    def __init__(self, ntargets):
        super().__init__()
        resnet = models.resnet18(pretrained=True, progress=True)
        modules = list(resnet.children())[:-1] # delete last layer
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(resnet.fc.in_features),
            nn.Dropout(0.2),
            nn.Linear(resnet.fc.in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, ntargets)
        )
    
    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x

# Load your pre-trained model (replace with the path to your model)
@st.cache(allow_output_mutation=True)
def load_model():
    model = ActionRecognitionClassifier(15)
    model.load_state_dict(torch.load('./saved_model/classifier_weights.pth', map_location='cpu'))
    #model = torch.load('./saved_model/classifier_weights.pth', map_location='cpu')
    model.eval()
    return model

# Define the transformation for the uploaded image
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to match input size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalization
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Function to predict the activity
def predict(image, model):
    image_tensor = transform_image(image)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = output.max(1)

    return predicted_class.item()

# Streamlit interface
st.title('Human Activity Recognition')

# Upload an image
uploaded_image = st.file_uploader("Choose an image...", type="jpg")

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    model = load_model()
    
    # Predict the action
    if st.button('Predict'):
        prediction = predict(image, model)
        ss=['calling', 'clapping', 'cycling', 'dancing', 'drinking', 'eating', 'fighting', 'hugging', 'laughing', 'listening_to_music',\
            'running', 'sitting', 'sleeping', 'texting', 'using_laptop']
        targ=ss[prediction]
        st.write(f'Predicted Action: {targ} {prediction}')
