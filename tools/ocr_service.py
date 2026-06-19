# """
# OCR 文字识别工具 — 基于 EasyOCR 本地模型
# """

# import os
# import tempfile
# from langchain.tools import tool
# from utils.logger import logger

# # 延迟导入：首次调用时才加载模型
# _ocr_reader = None


# def _get_ocr_reader():
#     """延迟加载 EasyOCR 模型"""
#     global _ocr_reader
#     if _ocr_reader is None:
#         import easyocr
#         logger.info("[OCRService] 加载 EasyOCR 模型 (ch_sim + en)")
#         _ocr_reader = easyocr.Reader(["ch_sim", "en"], gpu=False)
#         logger.info("[OCRService] 模型加载完成")
#     return _ocr_reader


# class OCRService:
#     """OCR 文字识别服务（基于 EasyOCR）"""

#     def __init__(self, lang: str = "ch_sim+en"):
#         self.reader = _get_ocr_reader()
#         logger.info("[OCRService] 初始化完成")

#     def ocr_image(self, image_path: str) -> str:
#         """从单张图片中提取文字"""
#         logger.info(f"[OCRService] OCR 识别图片: {image_path}")
#         if not os.path.exists(image_path):
#             logger.error(f"[OCRService] 图片文件不存在: {image_path}")
#             return ""

#         try:
#             results = self.reader.readtext(image_path)
#             texts = [text for _, text, confidence in results if confidence > 0.3]
#             result = "\n".join(texts)
#             logger.info(f"[OCRService] 图片识别完成 | 文字长度: {len(result)}")
#             return result
#         except Exception as e:
#             logger.error(f"[OCRService] 图片 OCR 失败: {e}")
#             return ""

#     def ocr_video(self, video_path: str, interval_sec: int = 30) -> str:
#         """从视频中按间隔提取帧并进行 OCR

#         Args:
#             video_path: 本地视频文件路径
#             interval_sec: 每隔多少秒抽取一帧（默认 30 秒）

#         Returns:
#             所有帧识别文字的拼接结果
#         """
#         logger.info(f"[OCRService] OCR 识别视频: {video_path} | 间隔: {interval_sec}s")
#         if not os.path.exists(video_path):
#             logger.error(f"[OCRService] 视频文件不存在: {video_path}")
#             return ""

#         try:
#             import cv2
#             cap = cv2.VideoCapture(video_path)
#             fps = cap.get(cv2.CAP_PROP_FPS)
#             total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             if fps <= 0 or total_frames <= 0:
#                 logger.warning("[OCRService] 无法读取视频信息")
#                 cap.release()
#                 return ""

#             duration = total_frames / fps
#             frame_interval = int(fps * interval_sec)
#             all_texts = []
#             frame_count = 0

#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break

#                 if frame_count % frame_interval == 0:
#                     # 临时保存帧为图片
#                     with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
#                         tmp_path = f.name
#                     cv2.imwrite(tmp_path, frame)
#                     text = self.ocr_image(tmp_path)
#                     if text:
#                         all_texts.append(text)
#                     os.unlink(tmp_path)

#                 frame_count += 1

#             cap.release()
#             result = "\n\n".join(all_texts)
#             logger.info(f"[OCRService] 视频识别完成 | 帧数: {frame_count} | 文字长度: {len(result)}")
#             return result
#         except ImportError:
#             logger.error("[OCRService] 需要安装 opencv-python: pip install opencv-python")
#             return ""
#         except Exception as e:
#             logger.error(f"[OCRService] 视频 OCR 失败: {e}")
#             return ""


# # 全局单例（延迟初始化）
# _ocr = None


# def _get_ocr():
#     global _ocr
#     if _ocr is None:
#         _ocr = OCRService()
#     return _ocr


# @tool
# def ocr_service_video(video_path: str) -> str:
#     """从视频帧画面中提取文字信息（OCR）。

#     接受本地视频文件路径，提取视频画面中的可见文字。

#     Args:
#         video_path: 本地视频文件路径

#     Returns:
#         从画面中识别出的文字内容
#     """
#     return _get_ocr().ocr_video(video_path)


# @tool
# def ocr_service_image(image_path: str) -> str:
#     """从单张图片中提取文字信息（OCR）。

#     Args:
#         image_path: 本地图片文件路径

#     Returns:
#         从图片中识别出的文字内容
#     """
#     return _get_ocr().ocr_image(image_path)