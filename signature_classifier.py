import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class SignatureClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SignatureClassifier, self).__init__()
        self.model = models.resnet101(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Sequential(
            nn.BatchNorm1d(self.model.fc.in_features),
            nn.Dropout(0.4),
            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def load_model(path, num_classes, class_names):
    model = SignatureClassifier(num_classes)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    def predict_signature(image):
        img = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)
        return class_names[predicted.item()]

    return predict_signature
