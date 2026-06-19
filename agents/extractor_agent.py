"""
多模态内容提取智能体

负责从视频中提取语音、字幕和画面文字信息。
使用 LangGraph create_react_agent 构建。
"""

from langgraph.prebuilt import create_react_agent

from tools.speech_to_text import speech_to_text
# from tools.ocr_service import ocr_service_video, ocr_service_image  # OCR 暂时注释
from utils.config import get_lightweight_model, settings
from utils.logger import logger

# 聊天模型
_model = get_lightweight_model()

# 提取智能体的系统提示词
EXTRACTOR_SYSTEM_PROMPT = """你是一位多模态内容处理专家，擅长使用先进的AI技术从视频中提取各种形式的文本内容。

你的能力和约束：
1. 从音频中提取语音内容（自动语音识别）
2. 提取视频字幕
3. 将提取结果结构化整理

工作时请遵循：
- 确保提取的内容尽可能准确
- 对提取结果进行基本的清洗和整理
- 标注每种内容的来源（语音/字幕）
"""

extractor_agent = create_react_agent(
    model=_model,
    prompt=EXTRACTOR_SYSTEM_PROMPT,
    tools=[speech_to_text],  # OCR 工具暂时注释
)

if __name__ == "__main__":
    print("ExtractorAgent 初始化完成")
    print("  - speech_to_text: 语音转文字")