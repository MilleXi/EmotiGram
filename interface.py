import gradio as gr
import os
import tempfile
from PIL import Image
import time
import numpy as np
from inference import ModelInference
from dswrapper import generate_social_content

# 配置模型路径
BASE_MODEL = "ybelkada/blip2-opt-2.7b-fp16-sharded"
ADAPTER_PATH = "blip2-finetuned"
EMOTION_MODEL_PATH = "best_model.pt"

# 创建临时目录用于存储上传的图片
TEMP_DIR = tempfile.mkdtemp()
print(f"临时目录创建在: {TEMP_DIR}")

# 创建offload目录用于模型卸载
OFFLOAD_DIR = os.path.join(TEMP_DIR, "offload")
os.makedirs(OFFLOAD_DIR, exist_ok=True)

class EmotiGramApp:
    def __init__(self):
        self.inferencer = None
        # 直接初始化模型
        self.load_models_on_start()
    
    def load_models_on_start(self):
        """启动时加载所有模型，而不是通过按钮触发"""
        try:
            print("开始加载所有模型...")
            start_time = time.time()
            
            # 初始化推理类
            self.inferencer = ModelInference(offload_dir=OFFLOAD_DIR)
            
            # 1. 先加载多模态情感分析模型
            multi_modal_loaded = self.inferencer.load_multi_modal_model(
                model_path=EMOTION_MODEL_PATH
            )
            if not multi_modal_loaded:
                print("❌ 多模态情感模型加载失败")
                return False
            
            # 2. 加载基础模型
            base_loaded = self.inferencer.load_base_model(
                base_model=BASE_MODEL
            )
            if not base_loaded:
                print("❌ 基础模型加载失败")
                return False
                
            elapsed = time.time() - start_time
            print(f"✅ 所有模型加载成功！耗时: {elapsed:.2f} 秒")
            return True
            
        except Exception as e:
            import traceback
            print(f"模型加载失败: {str(e)}")
            print(traceback.format_exc())
            return False
    
    def process_image(self, image):
        """处理上传的图片并返回结果"""
        if self.inferencer is None:
            return "模型尚未初始化，请刷新页面重试", "", "", "", ""
        
        if image is None:
            return "请上传图片", "", "", "", ""
        
        # 保存上传的图片到临时文件
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = Image.open(image)
            
        temp_image_path = os.path.join(TEMP_DIR, f"upload_{int(time.time())}.jpg")
        pil_image.save(temp_image_path)
        
        try:
            # 使用原始BLIP-2模型生成描述
            start_time = time.time()
            raw_caption = self.inferencer.generate_caption(temp_image_path, use_adapter=False)
            raw_time = time.time() - start_time
            print(f"原始描述生成耗时: {raw_time:.2f}秒")
            
            # 使用微调BLIP-2生成描述
            start_time = time.time()
            tuned_caption = self.inferencer.generate_caption(temp_image_path, use_adapter=True)
            tuned_time = time.time() - start_time
            print(f"风格化描述生成耗时: {tuned_time:.2f}秒")
            
            # 使用情感模型进行情感分类
            start_time = time.time()
            emotion_label, emotion_probs = self.inferencer.predict_emotion(temp_image_path, tuned_caption)
            emotion_time = time.time() - start_time
            print(f"情感分析耗时: {emotion_time:.2f}秒")
            
            # 获取情感概率的格式化字符串
            probs_str = ", ".join([f"{label}: {prob*100:.2f}%" 
                                 for label, prob in zip(["negative", "neutral", "positive"], emotion_probs)])
            
            # 调用DeepSeek生成社交媒体内容
            start_time = time.time()
            social_content = generate_social_content(raw_caption, tuned_caption, emotion_label)
            deepseek_time = time.time() - start_time
            print(f"DeepSeek文案生成耗时: {deepseek_time:.2f}秒")
            
            # 清理临时文件
            try:
                os.remove(temp_image_path)
            except:
                pass
                
            return (
                f"📝 原始描述：\n{raw_caption}",
                f"📢 风格化描述：\n{tuned_caption}",
                f"🎭 情感分析：{emotion_label}",
                f"情感概率分布：{probs_str}",
                f"🌟 社交媒体文案建议：\n{social_content}"
            )
            
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            return f"处理图片时出错：{str(e)}\n\n{error_msg}", "", "", "", ""

    def create_interface(self):
        """创建Gradio界面"""
        with gr.Blocks(title="EmotiGram - 情感图文生成器") as interface:
            gr.Markdown("# 📸 EmotiGram - 智能图文情感生成器")
            gr.Markdown("上传图片，获取AI生成的图像描述、情感分析和社交媒体文案建议。")
            
            with gr.Row():
                with gr.Column(scale=1):
                    upload_image = gr.Image(label="上传图片")
                    process_button = gr.Button("处理图片", variant="primary")
                    
                    # 添加状态指示
                    status = gr.Textbox(label="处理状态", value="模型已加载，可以开始处理图片")
                
                with gr.Column(scale=2):
                    raw_caption = gr.Textbox(label="原始描述", lines=3)
                    tuned_caption = gr.Textbox(label="风格化描述", lines=3)
                    emotion_label = gr.Textbox(label="情感分析结果", lines=1)
                    emotion_probs = gr.Textbox(label="情感概率分布", lines=1)
                    social_content = gr.Textbox(label="社交媒体文案建议", lines=6)
            
            # 设置事件处理
            process_button.click(
                fn=self.process_image, 
                inputs=[upload_image], 
                outputs=[raw_caption, tuned_caption, emotion_label, emotion_probs, social_content]
            )
            
            gr.Markdown("""
            ## 📋 使用说明
            1. 上传一张图片
            2. 点击"处理图片"按钮
            3. 查看生成的描述、情感分析和社交媒体文案建议
            
            ## ⚙️ 技术细节
            - 使用BLIP-2进行图像描述生成
            - 使用微调的BLIP-2生成风格化描述
            - 使用多模态情感分析模型预测图像和文本的情感
            - 使用DeepSeek API生成社交媒体文案建议
            """)
            
        return interface

# 启动应用
if __name__ == "__main__":
    app = EmotiGramApp()
    demo = app.create_interface()
    demo.launch(share=True)