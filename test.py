#使用data/test.py完成图像->多模态情感分类测试
import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import random
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 导入自定义推理工具
from inference import ModelInference

# 路径配置
DATA_ROOT = "data"
IMAGE_ROOT = os.path.join(DATA_ROOT, "image")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
MULTI_MODAL_MODEL_PATH = "best_model.pt"  # 多模态情感模型路径

# 情感标签映射
SENTIMENT_MAP = {
    -1: "negative",
    0: "neutral",
    1: "positive"
}
REVERSE_SENTIMENT_MAP = {
    "negative": -1,
    "neutral": 0,
    "positive": 1
}
TWEET_SENTIMENT_MAP = {
    0: "negative",
    2: "neutral",
    4: "positive"
}

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    plt.close()

def main():
    print("开始评估微调BLIP-2和多模态情感分析模型...")

    try:
        test_df = pd.read_csv(TEST_CSV)
        print(f"成功加载测试数据，共 {len(test_df)} 条记录")
    except Exception as e:
        print(f"加载测试数据失败: {str(e)}")
        return

    inferencer = ModelInference()

    # 加载 BLIP-2 模型（含 adapter）
    blip2_loaded = inferencer.load_blip2_model(
        base_model="ybelkada/blip2-opt-2.7b-fp16-sharded",
        adapter_path="blip2-finetuned"
    )

    if not blip2_loaded:
        print("加载BLIP-2模型失败，请检查模型路径和配置")
        return

    # 加载多模态情感分析模型
    multi_modal_loaded = inferencer.load_multi_modal_model(
        model_path=MULTI_MODAL_MODEL_PATH
    )
    if not multi_modal_loaded:
        print("加载多模态情感分析模型失败，请检查模型路径是否正确")
        return

    results = []
    blip2_predictions = []
    emotion_predictions = []
    true_labels = []

    print("\n开始对测试集进行推理...")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        image_path = os.path.join(IMAGE_ROOT, row['new_image_id'])
        if not os.path.exists(image_path):
            for ext in ['.jpg', '.jpeg', '.png']:
                test_path = image_path + ext
                if os.path.exists(test_path):
                    image_path = test_path
                    break
        if not os.path.exists(image_path):
            print(f"图片不存在: {image_path}，跳过")
            continue

        true_label = row['multi_label']
        true_labels.append(true_label)

        caption = inferencer.generate_caption_blip2(image_path)
        if caption is None:
            print(f"图片描述生成失败: {image_path}，跳过")
            continue
        blip2_predictions.append(caption)

        emotion_label, emotion_probs = inferencer.predict_emotion(image_path, caption)
        emotion_numeric = REVERSE_SENTIMENT_MAP.get(emotion_label)
        emotion_predictions.append(emotion_numeric)

        results.append({
            'image_id': row['new_image_id'],
            'true_label': row['multi_label'],
            'true_sentiment': SENTIMENT_MAP.get(row['multi_label'], "unknown"),
            'blip2_caption': caption,
            'emotion_label': emotion_label,
            'emotion_numeric': emotion_numeric,
            'image_path': image_path
        })

    print("\n===== BLIP-2模型生成描述示例 =====")
    for i, result in enumerate(results[:10]):
        print(f"{i+1}. 图片: {result['image_id']}")
        print(f"   BLIP-2生成文本: {result['blip2_caption']}")
        print("   ---")

    accuracy = accuracy_score(true_labels, emotion_predictions)
    report = classification_report(true_labels, emotion_predictions, target_names=[SENTIMENT_MAP[-1], SENTIMENT_MAP[0], SENTIMENT_MAP[1]])
    cm = confusion_matrix(true_labels, emotion_predictions)

    print("\n===== 多模态情感分析模型评估 =====")
    print(f"准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(report)
    plot_confusion_matrix(cm, [SENTIMENT_MAP[-1], SENTIMENT_MAP[0], SENTIMENT_MAP[1]])
    print("混淆矩阵已保存为 confusion_matrix.png")

    print("\n===== 详细推理结果示例 =====")
    correct_predictions = [r for r in results if r['emotion_numeric'] == r['true_label']]
    incorrect_predictions = [r for r in results if r['emotion_numeric'] != r['true_label']]

    print("\n正确预测的例子:")
    for i, result in enumerate(correct_predictions[:3]):
        print(f"{i+1}. 图片: {result['image_id']}")
        print(f"   BLIP-2生成文本: {result['blip2_caption']}")
        print(f"   真实情感: {result['true_sentiment']} ({result['true_label']})")
        print(f"   预测情感: {result['emotion_label']} ({result['emotion_numeric']})")
        print("   ---")

    print("\n错误预测的例子:")
    for i, result in enumerate(incorrect_predictions[:3]):
        print(f"{i+1}. 图片: {result['image_id']}")
        print(f"   BLIP-2生成文本: {result['blip2_caption']}")
        print(f"   真实情感: {result['true_sentiment']} ({result['true_label']})")
        print(f"   预测情感: {result['emotion_label']} ({result['emotion_numeric']})")
        print("   ---")

    results_df = pd.DataFrame(results)
    results_df.to_csv("evaluation_results.csv", index=False)
    print("\n评估结果已保存到 evaluation_results.csv")

if __name__ == "__main__":
    main()
