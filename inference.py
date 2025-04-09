import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from loguru import logger
from transformers import Blip2ForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, BertTokenizer
from torchvision import transforms
import torch.nn.functional as F
from peft import PeftModel
import tempfile

# 配置日志
logger.remove()
logger.add(lambda msg: print(msg, end=""), colorize=True)

class ModelInference:
    def __init__(self, offload_dir=None):
        """
        初始化推理类
        
        Args:
            offload_dir: 模型卸载目录，用于大模型的CPU卸载
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")

        # 模型和处理器
        self.base_model = None
        self.adapter_model = None
        self.processor = None
        self.multi_modal_model = None
        self.tokenizer = None

        # 设置CPU卸载目录
        self.offload_dir = offload_dir or os.path.join(tempfile.gettempdir(), "model_offload")
        os.makedirs(self.offload_dir, exist_ok=True)
        
        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def load_base_model(self, base_model="ybelkada/blip2-opt-2.7b-fp16-sharded"):
        """加载基础(未微调)BLIP-2模型"""
        try:
            logger.info(f"加载基础BLIP-2模型...")
            
            # 加载处理器，只需加载一次
            if self.processor is None:
                self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
                logger.info(f"处理器加载完成")
            
            # 加载基础模型
            if self.base_model is None:
                quant_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True
                )
                
                self.base_model = Blip2ForConditionalGeneration.from_pretrained(
                    base_model,
                    device_map="auto",
                    quantization_config=quant_config,
                    offload_folder=self.offload_dir
                )
                self.base_model.eval()
                logger.info(f"基础BLIP-2模型加载完成")
            
            return True
            
        except Exception as e:
            logger.error(f"加载基础BLIP-2模型失败: {str(e)}")
            return False

    def apply_adapter(self, adapter_path="blip2-finetuned"):
        """在基础模型上应用适配器"""
        try:
            if self.base_model is None:
                logger.error("无法应用适配器：基础模型未加载")
                return False
                
            logger.info(f"正在应用适配器 {adapter_path}...")
            
            # 应用适配器
            if self.adapter_model is None:
                self.adapter_model = PeftModel.from_pretrained(
                    self.base_model, 
                    adapter_path,
                    offload_folder=self.offload_dir
                )
                self.adapter_model.eval()
            
            logger.info("适配器应用成功")
            return True
            
        except Exception as e:
            logger.error(f"应用适配器失败: {str(e)}")
            return False

    def generate_caption(self, image_path, use_adapter=False):
        """
        使用BLIP-2模型生成描述
        
        Args:
            image_path: 图片路径
            use_adapter: 是否使用适配器微调模型
        """
        try:
            # 确保模型已加载
            if self.base_model is None:
                success = self.load_base_model()
                if not success:
                    return "基础模型加载失败"
            
            # 如果需要使用适配器且尚未加载
            if use_adapter and self.adapter_model is None:
                success = self.apply_adapter()
                if not success:
                    return "适配器应用失败"
            
            # 选择要使用的模型
            model = self.adapter_model if use_adapter and self.adapter_model is not None else self.base_model
            
            # 加载图像
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # 生成描述
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    num_beams=5,
                    min_length=5,
                    top_p=0.9,
                    repetition_penalty=1.5,
                    length_penalty=1.0,
                    temperature=1.0,
                )
                caption = self.processor.decode(outputs[0], skip_special_tokens=True)
                
            return caption
            
        except Exception as e:
            logger.error(f"生成描述失败: {str(e)}")
            return f"生成描述时出错: {str(e)}"

    def load_multi_modal_model(self, model_path, num_classes=3, fusion_type='moe', bert_model="bert-base-uncased"):
        """加载多模态情感分析模型"""
        try:
            # 导入模型类
            try:
                from models.model import MultiModalSentimentModel
            except ImportError:
                logger.error("未找到MultiModalSentimentModel类，请确保models.model模块可用")
                return False
            
            # 加载tokenizer    
            self.tokenizer = BertTokenizer.from_pretrained(bert_model)
            
            # 加载模型
            self.multi_modal_model = MultiModalSentimentModel(num_classes=num_classes, fusion_type=fusion_type)
            self.multi_modal_model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.multi_modal_model.eval().to(self.device)
            
            logger.info("多模态情感模型加载完成")
            return True
            
        except Exception as e:
            logger.error(f"加载多模态模型失败: {str(e)}")
            return False

    def predict_emotion(self, image_path, caption, label_map=None):
        """预测图像和文本的情感"""
        if self.multi_modal_model is None or self.tokenizer is None:
            logger.error("多模态情感分析模型未加载")
            return "模型未加载", np.zeros(3)
        
        if label_map is None:
            label_map = ["negative", "neutral", "positive"]
        
        try:
            # 处理图像
            if isinstance(image_path, str) or isinstance(image_path, Path):
                image = Image.open(image_path).convert("RGB")
            elif isinstance(image_path, Image.Image):
                image = image_path
            else:
                logger.error(f"不支持的图像类型: {type(image_path)}")
                return "不支持的图像类型", np.zeros(3)
            
            # 文本编码
            encoding = self.tokenizer(
                text=caption, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # 图像转换
            image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
            
            # 情感预测
            with torch.no_grad():
                outputs = self.multi_modal_model(
                    encoding['input_ids'],
                    encoding['attention_mask'],
                    image_tensor
                )
                
                # 根据模型输出结构调整
                _, _, multi_logits, *_ = outputs
                
                probs = F.softmax(multi_logits, dim=1)[0].cpu().numpy()
                label = label_map[probs.argmax()]
                
                return label, probs
                
        except Exception as e:
            logger.error(f"预测情感错误: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return "情感预测失败", np.zeros(3)