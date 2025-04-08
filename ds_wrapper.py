from openai import OpenAI
import os
import json
# openai.api_key = os.getenv("OPENAI_API_KEY")

# def generate_post(caption: str, emotion: str) -> str:
#     prompt = f"""我有一张图，并生成了如下描述："{caption}"。这张图传达的情绪是：{emotion}。请基于这些内容，生成一句适合年轻人发朋友圈的文案，表达这个情绪，要求自然简洁，不超过50字："""

#     response = openai.ChatCompletion.create(
#         model="gpt-4",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.7,
#     )
#     return response['choices'][0]['message']['content'].strip()


# 使用 DeepSeek API 进行请求
with open("config.json", "r") as f: 
    os.config = json.load(f)
key = os.config["api_key"]
client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

def generate_post(caption: str, emotion: str) -> str:
    prompt = f"""我有一张图，并生成了如下描述："{caption}"。这张图传达的情绪是：{emotion}。请基于这些内容，用中文生成一句适合年轻人发朋友圈的文案，表达这个情绪，要求自然简洁，不超过50字："""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": prompt},
        ],
        stream=True
        )

    return response['choices'][0]['message']['content'].strip()