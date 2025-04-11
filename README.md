# 📸 EmotiGram: Affective-Captioning-for-Social-Sharing

**EmotiGram** 是一个基于视觉语言模型（VLM）与多模态情绪理解技术的智能图文共创平台。它能够通过用户上传的一张图片，自动生成一段风格真实、情绪一致、贴近人类表达习惯的社交媒体文案。

> 🤖 背后融合了图像内容理解、模拟社交风格文本生成、情绪分析与 LLM 强化表达等关键模块。

---

## 🌟 项目亮点

### 🎯 多模态情绪感知  
结合图像与文本，使用训练好的多模态情绪分类模型，感知图文内容所表达的 **整体情绪基调**（positive / neutral / negative）。

### 🧠 双BLIP-2协同生成  
- 使用 **未微调 BLIP-2** 生成一段对图像的客观描述；  
- 使用 **微调后的 BLIP-2 + Adapter** 生成一段真实社交风格的模拟推文。

### ✍️ LLM文案共创（DeepSeek API）  
将上述两个描述与情绪结果，构造成提示词传入 DeepSeek Chat 模型，生成最终推荐文案。

### 📊 模块化结构与交互式界面  
每个功能模块独立封装，方便二次开发、扩展或部署。使用 Gradio 打造一键式上传 + 输出体验，自动完成图文→情绪→文案三连操作。

---

## 🛠️ 系统架构

```text
       ┌──────────────┐
       │  用户上传图像  │
       └──────┬───────┘
              ▼
     ┌─────────────────────┐
     │ BLIP2 图像理解模块  │
     └──┬────────────┬─────┘
        ▼            ▼
  原始描述        风格推文（微调）
     │              │
     └──┬──────────┘
        ▼
    🎭 多模态情绪分类模型
        │
        ▼
   📤 构建LLM Prompt
        │
        ▼
  ✨ GPT文案生成（DeepSeek）
```

---

## 🚀 项目结构

| 文件/目录              | 功能说明 |
|------------------------|----------|
| `interface.py`         | Gradio 界面主入口，支持上传图像一键生成文案 |
| `inference.py`         | 推理类，封装 BLIP2 / Adapter / 多模态情感模型 |
| `dswrapper.py`         | 调用 DeepSeek API，构造文案生成提示词 |
| `example.py`           | 示例脚本：展示未微调 vs 微调BLIP2 + 情绪分析 + 文案结果 |
| `test.py`              | 批量测试集评估脚本，输出准确率与混淆矩阵 |
| `blip2-finetune.ipynb/`| Blip2微调得到 Adapter 代码 |
| `data/`                | 数据集，Twitter1517，请见 releases |
| `config_template.json` | 配置 DeepSeek API 文件 |
| `best_model.pt`        | 多模态情绪分类模型权重，请见 releases |
| `blip2-finetuned/`     | Blip2微调模型权重，请见 releases |

---

## 💡 使用方式

### 1. 本地运行

确保安装依赖（建议使用 Python 3.12 + CUDA 11.x）：

```bash
pip install -r requirements.txt
```
### 2. 配置 🔐 DeepSeek 接口

请在 config_template.json 中设置你的 API KEY

并将 config_template.json 重命名为 config.json

### 3. 启动界面

```bash
python interface.py
```

> 打开 Gradio 界面，上传图片，查看描述与文案。

---

## 📦 技术细节

| 模块              | 技术细节 |
|-------------------|----------|
| 图像描述生成       | [ybelkada/blip2-opt-2.7b-fp16-sharded](https://huggingface.co/ybelkada/blip2-opt-2.7b-fp16-sharded)，使用 HuggingFace Transformers 接入 |
| 模型微调方法       | LoRA Adapter 微调 + PEFT |
| 情绪分类模型       | 自定义 MultiModalSentimentModel，支持多种融合策略 |
| LLM 调用方式       | DeepSeek Chat API |
| 模型加载优化       | 支持 offload + 8bit 量化加速大模型加载 |

---

## 📈 示例效果（来自 `example.py`）

```text
📝 原始描述：
a couple standing together on a bridge overlooking a lake

📢 微调 BLIP-2 风格推文：
Sunset walks with you never get old 💛

🎭 情绪分析：
positive（概率：positive 84.2%）

🌟 生成文案推荐：
每一次一起看夕阳的时刻，都想永远停在这儿💫 #感恩当下
```

---

## 🤖 创新价值

相比直接使用 GPT，EmotiGram 在以下方面具备显著优势：

| 维度           | GPT | EmotiGram |
|----------------|-----|-----------|
| 图像情绪识别   | ❌   | ✅         |
| 多模态推文融合 | ❌   | ✅         |
| 提示词优化结构 | ❌   | ✅         |
| 真实用户表达风格 | ❌   | ✅（通过 BLIP2 微调） |

---

## 📌 已知限制 & 展望

- 添加多语言支持（中/英/多语自动切换）
- 加入多风格文案模板（文艺、幽默、简约）
- 引入图像风格滤镜与图文调性匹配推荐
- 将模型封装为 API，支持移动端/微信小程序等前端调用
- 当前情感模型在模糊情绪类别上仍有改进空间；
- 未来可支持用户设定风格偏好、支持视频片段摘要、支持小红书/微博等平台风格模板；
- 提供“创作者专属风格记忆”，模拟特定用户文风。

---

## 💬 致谢

本项目参考并使用了如下开源组件：

- [BLIP-2 (ybelkada)](https://huggingface.co/ybelkada/blip2-opt-2.7b-fp16-sharded)
- [Transformers (HuggingFace)](https://github.com/huggingface/transformers)
- [DeepSeek Chat API](https://platform.deepseek.com/)
- Gradio, PEFT, Torch, Sklearn 等

---
## 📮 联系作者

如需部署、调试或演示需求，欢迎随时联系开发者。若你希望扩展为微信小程序、嵌入网页、个性化写作插件，也可以一键适配。如果你对本项目感兴趣或想要合作，欢迎提 Issue 或发起 PR！

## ✨ 让图像开口说话，让情绪更有表达

EmotiGram，给每一次分享，注入真实又温柔的温度。

