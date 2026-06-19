"""
抖音爬取智能体

负责从抖音平台爬取视频和音频内容。
使用 LangGraph create_react_agent 构建，绑定以下工具：
  - crawl_videos: 爬取博主视频列表
  - get_video_info: 获取视频元数据
  - download_single_video: 下载视频文件
  - download_video_audio: 下载音频文件（供 SpeechToText 使用）
  - delete_video: 删除本地视频
"""

from langgraph.prebuilt import create_react_agent

from tools.douyin_crawler import (
    crawl_videos,
    get_video_info,
    download_single_video,
    download_video_audio,
    delete_video,
)
from utils.config import get_lightweight_model, settings
from utils.logger import logger

# 聊天模型
_model = get_lightweight_model()

# 爬取智能体的系统提示词
CRAWLER_SYSTEM_PROMPT = """你是一位抖音视频爬取专家，负责从抖音平台爬取指定URL的视频内容。

你的能力和约束：
1. 支持通过抖音号或完整主页链接获取视频列表
2. 支持单个视频的文件下载和音频提取
3. 下载视频后可使用 download_video_audio 提取音频，供语音转文字使用
4. 自动跳过已下载的文件（断点续爬）
5. 提取完整元数据：标题、发布时间、点赞量、收藏量、时长等
6. 加入请求延迟、随机UA、重试机制，防止被反爬

工作时请遵循：
- 接收到爬取任务后，先确认 URL 或抖音号有效
- 爬取视频列表后，如需后续语音分析，请使用 download_video_audio 下载音频
- 合理使用工具，按需爬取指定数量
- 及时报告爬取进度和结果
- 遇到错误时，给出清晰的错误说明
"""

crawler_agent = create_react_agent(
    model=_model,
    prompt=CRAWLER_SYSTEM_PROMPT,
    tools=[crawl_videos, get_video_info, download_single_video, download_video_audio, delete_video],
)

if __name__ == "__main__":
    print("CrawlerAgent 初始化完成，工具列表:")
    print("  - crawl_videos: 爬取博主视频列表")
    print("  - get_video_info: 获取视频元数据")
    print("  - download_single_video: 下载视频文件")
    print("  - download_video_audio: 下载音频文件")
    print("  - delete_video: 删除本地视频")