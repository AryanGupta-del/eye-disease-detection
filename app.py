import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import gradio as gr
import numpy as np

LABELS = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
LABEL_NAMES = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 
               'AMD', 'Hypertension', 'Myopia', 'Other']

class EyeModel(nn.Module):
    def __init__(self, num_classes=8):
        super(EyeModel, self).__init__()
        self.model = models.efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    def forward(self, x):
        return self.model(x)

device = torch.device('cpu')
model = EyeModel(num_classes=8).to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image):
    img = Image.fromarray(image).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.sigmoid(outputs).squeeze().numpy()
    
    results = {}
    detected = []
    
    for i, (label, name) in enumerate(zip(LABELS, LABEL_NAMES)):
        prob = float(probs[i])
        results[name] = round(prob, 3)
        if prob > 0.5:
            detected.append(f"{name} ({prob*100:.1f}%)")
    
    if detected:
        diagnosis = "Conditions detected: " + ", ".join(detected)
    else:
        top_idx = np.argmax(probs)
        diagnosis = f"Most likely: {LABEL_NAMES[top_idx]} ({probs[top_idx]*100:.1f}%)"
    
    return diagnosis, results

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload Retinal Fundus Image"),
    outputs=[
        gr.Textbox(label="Diagnosis"),
        gr.Label(label="Disease Probabilities", num_top_classes=8)
    ],
    title="Eye Disease Detection",
    description="Upload a retinal fundus image to detect eye diseases including Diabetes, Glaucoma, Cataract, AMD, Hypertension, and Myopia.",
    examples=[],
)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())