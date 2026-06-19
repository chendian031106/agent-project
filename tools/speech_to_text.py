"""
语音转写工具 — 基于 faster-whisper 本地模型
"""

import os
from langchain.tools import tool
from utils.logger import logger

# 延迟导入：首次调用时才加载模型
_whisper_model = None


def _get_whisper_model():
    """延迟加载 Whisper 模型"""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        model_size = os.getenv("WHISPER_MODEL_SIZE", "base")
        logger.info(f"[SpeechToText] 加载 faster-whisper 模型: {model_size}")
        # 使用 CPU, int8 量化以降低内存占用
        _whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
        logger.info("[SpeechToText] 模型加载完成")
    return _whisper_model


class SpeechToText:
    """语音转文字服务（基于 faster-whisper）"""

    def __init__(self, model: str = "base"):
        self.model = _get_whisper_model()
        logger.info("[SpeechToText] 初始化完成")

    def transcribe(self, audio_path: str) -> str:
        """将音频文件转写为文字"""
        logger.info(f"[SpeechToText] 转写音频: {audio_path}")
        if not os.path.exists(audio_path):
            logger.error(f"[SpeechToText] 音频文件不存在: {audio_path}")
            return ""

        segments, info = self.model.transcribe(audio_path, language="zh")
        text = "".join(segment.text for segment in segments)
        logger.info(f"[SpeechToText] 转写完成 | 音频时长: {info.duration:.1f}s | 文字长度: {len(text)}")
        return text


# 全局单例（延迟初始化）
_stt = None


def _get_stt():
    global _stt
    if _stt is None:
        _stt = SpeechToText()
    return _stt


@tool
def speech_to_text(audio_path: str) -> str:
    """将音频文件转写为文字。

    接受本地音频文件路径（MP3/WAV 等格式），返回对应的文字内容。

    Args:
        audio_path: 本地音频文件路径

    Returns:
        转写后的文字内容
    """
    return _get_stt().transcribe(audio_path)