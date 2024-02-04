import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

ft_model_path = 'models/Bilder/vit'

ft_model = ViTForImageClassification.from_pretrained(ft_model_path)
ft_processor = ViTImageProcessor.from_pretrained(ft_model_path)

def process_ft_image(image):
    inputs = ft_processor(image, return_tensors='pt')
    with torch.no_grad():
        outputs = ft_model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class_id = torch.argmax(predictions, dim=-1).item()
    config = ft_model.config
    class_names = config.id2label
    predicted_class_label = class_names[predicted_class_id]
    return predicted_class_id, predicted_class_label