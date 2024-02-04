import torch
from PIL import Image
import torchvision.models as models
from PIL import ImageEnhance, ImageFilter
from torchvision import transforms
import torch.nn as nn

class EnhanceContrast:
    def __init__(self, enhancement_factor=1.5):
        self.enhancement_factor = enhancement_factor

    def __call__(self, img):
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(self.enhancement_factor)
        return img


class GaussianBlur:
    def __init__(self, radius=2):
        self.radius = radius

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(self.radius))

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    EnhanceContrast(),
    GaussianBlur(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

resnet = models.resnet50(pretrained=True)
resnet.fc = nn.Linear(resnet.fc.in_features, out_features=2)
model = resnet

device = torch.device("cpu")


model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 1)

state_dict = torch.load("./models/Bilder/resnet50/resnet50_finetuned.pth")
model.load_state_dict(state_dict)

model.to(device)
model.to('cpu')
model.eval()

image_path = './h.jpg'

image = Image.open(image_path)
image = transform(image).unsqueeze(0)

image = image.to('cpu')
with torch.no_grad():
    outputs = model(image)
    preds = torch.sigmoid(outputs) > 0.5

predicted_class = preds.cpu().numpy()[0][0]
#False: NORMAL; True: PNEUMONIA
print(f"Predicted Class: {predicted_class}")