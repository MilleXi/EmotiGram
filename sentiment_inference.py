import torch
from transformers import BertTokenizer
from torchvision import transforms
from models.model import MultiModalSentimentModel
import torch.nn.functional as F
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型加载
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = MultiModalSentimentModel(num_classes=3, fusion_type='moe')
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval().to(device)

label_map = ["positive", "neutral", "negative"]

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def predict_emotion(image: Image.Image, caption: str):
    encoding = tokenizer(caption, return_tensors="pt", padding=True, truncation=True).to(device)
    image_tensor = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, multi_logits, *_ = model(encoding['input_ids'], encoding['attention_mask'], image_tensor)
        probs = F.softmax(multi_logits, dim=1)[0].cpu().numpy()
        label = label_map[probs.argmax()]
    return label, probs
