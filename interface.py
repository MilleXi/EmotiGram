import gradio as gr
from PIL import Image
from blip_generator import generate_caption
from sentiment_inference import predict_emotion
from ds_wrapper import generate_post

def pipeline(image):
    image = Image.fromarray(image).convert("RGB")
    caption = generate_caption(image)
    emotion, prob = predict_emotion(image, caption)
    post = generate_post(caption, emotion)
    return caption, f"{emotion}（{max(prob):.2f}）", post

iface = gr.Interface(
    fn=pipeline,
    inputs=gr.Image(type="numpy", label="上传图片"),
    outputs=[
        gr.Textbox(label="BLIP 生成文本"),
        gr.Textbox(label="情感预测"),
        gr.Textbox(label="DeepSeek 优化文案")
    ],
    title="📷 情绪感知朋友圈文案生成助手",
    description="上传一张图片，即可生成一段带有情绪的朋友圈文案！"
)

if __name__ == "__main__":
    iface.launch()
