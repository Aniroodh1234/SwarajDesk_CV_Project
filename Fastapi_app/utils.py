import torch
from PIL import Image
from torchvision import transforms

# Image transform for ViT
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def load_image(image_file):
    """Convert uploaded file to tensor for model"""
    image = Image.open(image_file).convert("RGB")
    return image_transforms(image).unsqueeze(0)  # shape: (1, 3, 224, 224)

def get_top_class(logits):
    """Get class name & confidence from logits"""
    probs = torch.softmax(logits, dim=1)
    confidence, predicted_class = torch.max(probs, dim=1)
    return predicted_class.item(), float(confidence.item())
