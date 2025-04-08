import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def generate_caption(image: Image.Image) -> str:
    inputs = blip_processor(image, return_tensors="pt").to(device)
    output = blip_model.generate(**inputs)
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    return caption
