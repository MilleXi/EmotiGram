from openai import OpenAI
import os
import json
import time

# 加载配置
def load_config():
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
            return config
    except Exception as e:
        print(f"加载配置文件失败: {str(e)}")
        # 返回默认配置或提示用户设置API密钥
        return {"api_key": os.environ.get("DEEPSEEK_API_KEY", "")}

def generate_social_content(caption, style_caption, emotion):
    """
    使用DeepSeek API生成基于图像描述和情感的社交媒体内容
    
    Args:
        caption: 原始图像描述
        style_caption: 风格化图像描述
        emotion: 情感标签 (positive/neutral/negative)
        
    Returns:
        str: 生成的社交媒体内容
    """
    try:
        # 加载配置
        config = load_config()
        api_key = config.get("api_key", "")
        
        if not api_key:
            return "⚠️ 未找到DeepSeek API密钥，请在config.json中设置api_key或设置环境变量DEEPSEEK_API_KEY"
        
        # 初始化客户端
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        
        # 构建提示词
        prompt = f"""
        你是一个懂社交、懂年轻人、懂情绪的AI文案助手。你的任务是：根据用户上传的一张图片，结合 AI 模型生成的描述、模拟的推文语言、识别到的情绪基调，帮助用户生成一段风格自然、真实感强、情感一致的朋友圈/社交媒体文案。请综合以下三部分信息：
        
        [图像内容描述]这是系统通过视觉模型对图像生成的客观描述（偏理性）："{caption}"
        
        [用户风格推文预测]这是模型基于图像生成的、贴近真实用户社交风格的可能推文（偏感性）："{style_caption}"
        
        [系统识别情绪]这是模型分析图像与文案的多模态情感结果（情绪基调）："{emotion}"（可为 positive / neutral / negative）
        
        请参考以上内容，融合图像内容、语言风格、情绪感受，生成一段适合社交媒体发布的文案，要求如下：
        使用中文；适合年轻人、有梗、风格自然、有情绪、有细节；避免过于客观，也不要浮夸；保持情感基调与分析结果一致；字数在 25~100 字之间；可适当添加 emoji（非必须）；可选 hashtag，但不强求；
        
        最终请只输出生成的文案，不要解释过程。
        """
        
        # 发送请求
        start_time = time.time()
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": prompt},
            ]
        )
        
        # 提取生成的内容
        generated_content = response.choices[0].message.content.strip()
        
        elapsed = time.time() - start_time
        print(f"DeepSeek API请求耗时: {elapsed:.2f}秒")
        
        return generated_content
        
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        return f"调用DeepSeek API时出错：{str(e)}\n\n{error_msg}"

# def generate_social_content_streaming(caption, style_caption, emotion):
#     """使用DeepSeek API生成基于图像描述和情感的社交媒体内容（流式响应）"""
    
#     try:
#         # 加载配置
#         config = load_config()
#         api_key = config.get("api_key", "")
        
#         if not api_key:
#             yield "⚠️ 未找到DeepSeek API密钥，请在config.json中设置api_key或设置环境变量DEEPSEEK_API_KEY"
#             return
        
#         # 初始化客户端
#         client = OpenAI(
#             api_key=api_key,
#             base_url="https://api.deepseek.com"
#         )
        
#         # 构建提示词
#         prompt = f"""
#         你是一个懂社交、懂年轻人、懂情绪的AI文案助手。你的任务是：根据用户上传的一张图片，结合 AI 模型生成的描述、模拟的推文语言、识别到的情绪基调，帮助用户生成一段风格自然、真实感强、情感一致的朋友圈/社交媒体文案。请综合以下三部分信息：
        
#         [图像内容描述]这是系统通过视觉模型对图像生成的客观描述（偏理性）："{caption}"
        
#         [用户风格推文预测]这是模型基于图像生成的、贴近真实用户社交风格的可能推文（偏感性）："{style_caption}"
        
#         [系统识别情绪]这是模型分析图像与文案的多模态情感结果（情绪基调）："{emotion}"（可为 positive / neutral / negative）
        
#         请参考以上内容，融合图像内容、语言风格、情绪感受，生成一段适合社交媒体发布的文案，要求如下：
#         风格自然、有情绪、有细节；避免过于客观，也不要浮夸；保持情感基调与分析结果一致；没有特殊情况的话请使用中文；文案要符合当代年轻人的口味；字数在 25~100 字之间；可适当添加 emoji（非必须）；可选 hashtag，但不强求；
        
#         最终请只输出生成的文案，不要解释过程。
#         """
        
#         # 发送流式请求
#         stream = client.chat.completions.create(
#             model="deepseek-chat",
#             messages=[
#                 {"role": "user", "content": prompt},
#             ],
#             stream=True
#         )
        
#         # 流式输出
#         for chunk in stream:
#             if chunk.choices[0].delta.content:
#                 yield chunk.choices[0].delta.content
                
#     except Exception as e:
#         yield f"调用DeepSeek API时出错：{str(e)}"

# openai.api_key = os.getenv("OPENAI_API_KEY")

# def generate_post(caption: str, emotion: str) -> str:
#     prompt = f"""我有一张图，并生成了如下描述："{caption}"。这张图传达的情绪是：{emotion}。请基于这些内容，生成一句适合年轻人发朋友圈的文案，表达这个情绪，要求自然简洁，不超过50字："""

#     response = openai.ChatCompletion.create(
#         model="gpt-4",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.7,
#     )
#     return response['choices'][0]['message']['content'].strip()
