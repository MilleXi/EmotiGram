from inference import ModelInference
from PIL import Image

# 路径配置
image_path = "data/image/1313.jpg"

# 初始化推理类
inferencer = ModelInference()

# 1. 首先加载基础模型
base_loaded = inferencer.load_base_model(
    base_model="ybelkada/blip2-opt-2.7b-fp16-sharded"
)

if not base_loaded:
    print("基础模型加载失败，请检查路径和依赖")
    exit()

# 2. 使用原始BLIP-2模型生成描述
raw_caption = inferencer.generate_caption(image_path)
print(f"\n📝 未微调BLIP-2生成原始描述：\n{raw_caption}")

# 3. 应用适配器
adapter_applied = inferencer.apply_adapter(adapter_path="blip2-finetuned")

if not adapter_applied:
    print("适配器应用失败")
    exit()

# 4. 使用微调BLIP-2生成描述
tuned_caption = inferencer.generate_caption(image_path)
print(f"\n📢 微调BLIP-2生成推文描述：\n{tuned_caption}")

# 5. 加载多模态情感分析模型
multi_modal_loaded = inferencer.load_multi_modal_model(
    model_path="best_model.pt"
)

if not multi_modal_loaded:
    print("多模态模型加载失败")
    exit()

# 6. 使用情感模型进行情感分类
label, probs = inferencer.predict_emotion(image_path, tuned_caption)
print(f"\n🎭 多模态情感预测结果：{label}")
print(f"类别概率：{probs}")