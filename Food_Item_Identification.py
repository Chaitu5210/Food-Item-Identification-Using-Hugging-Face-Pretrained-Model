import os
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch


image_directory = "Images"


model_name = "nateraw/food"
model = AutoModelForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)


for filename in os.listdir(image_directory):
    if filename.endswith(".jpeg") or filename.endswith(".png"):
        image_path = os.path.join(image_directory, filename)


        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt")
        

        with torch.no_grad():
            outputs = model(**inputs)


        predicted_class_idx = outputs.logits.argmax(-1).item()
        predicted_label = model.config.id2label[predicted_class_idx]

        print(predicted_label)
