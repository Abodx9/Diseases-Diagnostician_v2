import torch
from PIL import Image
import torchvision.models as models
from PIL import ImageEnhance, ImageFilter
from torchvision import transforms
import torch.nn as nn

# packages for language model
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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

# state_dict = torch.load("./models/Bilder/resnet50/resnet50_finetuned.pth")
state_dict = torch.load("../../../../cv_model/chest_xray/resnet50_finetuned.pth")
model.load_state_dict(state_dict)

model.to(device)
model.to('cpu')
model.eval()

# image_path = './h.jpg'
# image_path = '../../../../cv_model/chest_xray/chestXRay_normal.jfif'
image_path = '../../../../cv_model/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg'

image = Image.open(image_path)
image = transform(image).unsqueeze(0)

image = image.to('cpu')
with torch.no_grad():
    outputs = model(image)
    preds = torch.sigmoid(outputs) > 0.5

predicted_class = preds.cpu().numpy()[0][0]
#False: NORMAL; True: PNEUMONIA
# print(f"Predicted Class: {predicted_class}")

if predicted_class == 0:
    print("The X-ray of your lungs looks normal")
# redirect the prediction to language model if Pneumonia detected
else:
    predicted_disease = 'Pneumonia'
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # disease2diagnosis
    disease2diagnosis_model_path = '../SprachModel/disease2diagnosis_gpt2_fine-tuned.pth'
    disease2diagnosis_model = GPT2LMHeadModel.from_pretrained('gpt2')
    disease2diagnosis_model.load_state_dict(torch.load(disease2diagnosis_model_path, map_location=device))
    disease2diagnosis_model = disease2diagnosis_model.to(device)

    disease_input_ids = tokenizer.encode(predicted_disease, return_tensors='pt').to(device)
    disease_attention_mask = torch.ones(disease_input_ids.shape, dtype=torch.long, device=device)

    diagnosis_output = disease2diagnosis_model.generate(
        disease_input_ids,
        attention_mask=disease_attention_mask,
        max_length=100,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.5,
        top_k=20
    )

    generated_text = tokenizer.decode(diagnosis_output[0], skip_special_tokens=True)
    generated_text = generated_text.replace('[SEP]', '.')
    cut_off_point = generated_text.rfind('.') + 1
    final_text = generated_text[:cut_off_point]

    print(final_text)
