import pandas as pd
from PIL import Image
from blip_generator import generate_caption
from sentiment_inference import predict_emotion
from sklearn.metrics import classification_report
import os

df = pd.read_csv("data/test.csv")

true_labels = []
pred_labels = []

label_map = ["positive", "neutral", "negative"]
label_to_id = {name: idx for idx, name in enumerate(label_map)}

for _, row in df.iterrows():
    image_path = os.path.join('data/image', row['new_image_id'])
    img = Image.open(image_path).convert("RGB")
    label_index = int(row["multi_label"])
    label_map_ordered = {-1: "negative", 0: "neutral", 1: "positive"}
    true_label = label_map_ordered[label_index]
    
    caption = generate_caption(img)
    pred_label, _ = predict_emotion(img, caption)

    true_labels.append(label_to_id[true_label])
    pred_labels.append(label_to_id[pred_label])

# 评估
print(classification_report(true_labels, pred_labels, target_names=label_map))
