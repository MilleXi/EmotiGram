# 📸 EmotiGram：Affective-Captioning-for-Social-Sharing （developing）

> 情绪感知型朋友圈文案生成助手

EmotiGram 是一个将 **视觉图像理解**、**情绪识别** 与 **社交文案生成** 融为一体的多模态智能系统。通过图像字幕生成模型 BLIP、情感分析模型 EmoMoE 以及 DeepSeek 接口，EmotiGram 能够从图片中生成描述文字、识别对应情绪，并产出适合发朋友圈的情绪化短文案。

---

## ✨ 项目亮点

1. **图像自动理解**：基于微调后的 BLIP 模型，从图片生成更贴近人类表达的描述文本。
2. **多模态情感识别**：融合图像和文本特征，通过 ViT + BERT + Transformer/MoE 架构实现高精度情绪分类。
3. **DeepSeek 文案生成**：调用 DeepSeek 接口，结合图文语义与情绪，生成适合社交分享的文艺文案。
4. **交互式界面**：使用 Gradio 打造一键式上传 + 输出体验，自动完成图文→情绪→文案三连操作。
5. **模块化结构清晰**：每个功能模块独立封装，方便二次开发、扩展或部署。

---

## 📁 项目结构说明：

- `blip_generator.py` ：图像字幕生成模块（可加载微调后模型）
- `sentiment_inference.py` ：情绪识别推理模块（调用多模态分类模型）
- `ds_wrapper.py` ：与 DeepSeek 接口通信模块
- `test_evaluation.py` ：模型在 test.csv 上的评估脚本（输出准确率与F1）
- `interface.py` ：基于 Gradio 的图形用户界面
- `blip_finetune_captioning.ipynb` ：BLIP 模型微调训练笔记本（可自定义风格）
- `best_model.pt` ：已训练好的情绪分类模型权重
- `config_template.json` : 设置 DeepSeek API Key
- `data/` ：数据文件夹，包括 train.csv、test.csv 及 image 图像文件夹

---
## ⚙️ 关于我的 多模态情感识别 & 数据集：

- 详情请见 [EmoMoE](https://github.com/MilleXi/EmoMoE) (暂时 private)


---

## 🛠 使用指南：

1. 安装依赖（建议使用虚拟环境）  
   `pip install -r requirements.txt`

2. 微调 BLIP 模型（可选）  
   运行 `blip_finetune_captioning.ipynb`，使用 train.csv 中图文对个性化训练 BLIP 模型

3. 启动界面程序  
   `python interface.py`  
   打开网页界面上传图片，即可获取描述、情绪预测与朋友圈文案

4. 模型评估（可选）  
   `python test_evaluation.py`  
   使用 test.csv 中的图文对对情绪识别模型进行验证评估

---

## 🔐 **DeepSeek 接口使用：**

- 请在 config_template.json 中设置你的 API KEY

- 并将 config_template.json 重命名为 config.json

---

## 📊 **模型亮点技术：**

- 基于 Huggingface Transformers 的 BLIP 图像描述架构
- 自定义多模态情绪分析模型：支持 Transformer/Attention/MoE 融合
- 支持情绪冲突检测模块
- 可扩展的 DeepSeek 文案生成 prompt 模板

---

## 🎯 **未来可拓展方向：**

- 添加多语言支持（中/英/多语自动切换）
- 加入多风格文案模板（文艺、幽默、简约）
- 引入图像风格滤镜与图文调性匹配推荐
- 将模型封装为 API，支持移动端/微信小程序等前端调用

---

## 📮 **联系作者**

如果你对本项目感兴趣或想要合作，欢迎提 Issue 或发起 PR！

---

## ✨ **让图像开口说话，让情绪更有表达**  
EmotiGram，给每一次分享，注入真实又温柔的温度。
