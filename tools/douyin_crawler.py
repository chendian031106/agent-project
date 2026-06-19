"""
抖音视频爬取工具 - 纯工具层

不调用任何大语言模型，只负责视频爬取和元数据提取。
所有对外暴露的方法均使用 @tool 注解，供智能体调用。
"""

import json
import os
import random
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yt_dlp
from langchain.tools import tool
from pydantic import BaseModel, Field

from utils.config import settings
from utils.logger import logger

# ============ Pydantic 数据模型 ============


class VideoInfo(BaseModel):
    """抖音视频完整元数据"""

    video_id: str = Field(default="", description="抖音视频唯一ID")
    author: str = Field(default="", description="博主昵称")
    author_id: str = Field(default="", description="博主抖音号")
    title: str = Field(default="", description="视频标题")
    description: str = Field(default="", description="视频简介")
    publish_time: str = Field(default="", description="发布时间（ISO 8601格式）")
    like_count: int = Field(default=0, description="点赞数")
    comment_count: int = Field(default=0, description="评论数")
    collect_count: int = Field(default=0, description="收藏数")
    share_count: int = Field(default=0, description="分享数")
    duration: int = Field(default=0, description="视频时长（秒）")
    file_path: str = Field(default="", description="本地保存的视频文件绝对路径")


class CrawlResult(BaseModel):
    """爬取操作的统一返回结果"""

    success: bool
    data: Any = None
    error: Optional[str] = None
    count: int = 0


# ============ 常量 ============

USER_AGENTS: List[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Mobile/15E148 Safari/604.1",
]


# ============ DouyinCrawler 主类 ============


class DouyinCrawler:
    """抖音视频爬取工具

    纯工具层实现，负责视频爬取和元数据提取。
    所有配置从 utils.config.Settings 读取，不硬编码任何路径或参数。
    """

    def __init__(self) -> None:
        self.video_dir = Path(settings.VIDEO_STORAGE_PATH)
        self.video_dir.mkdir(parents=True, exist_ok=True)

        self.cookie_string: str = settings.DOUYIN_COOKIE
        self.max_retry: int = settings.MAX_RETRY
        self.timeout: int = settings.TIMEOUT

        self._cookie_file_path: Optional[str] = None
        if self.cookie_string:
            self._cookie_file_path = self._write_cookie_file()

        logger.info(
            f"DouyinCrawler 初始化完成 | "
            f"存储路径: {self.video_dir} | "
            f"Cookie: {'已配置' if self.cookie_string else '未配置'} | "
            f"最大重试: {self.max_retry}"
        )

    # ---------- 内部工具方法 ----------

    def _write_cookie_file(self) -> str:
        """将 cookie 字符串写入临时文件，供 yt-dlp 使用"""
        try:
            cookie_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            )
            cookie_file.write("# Netscape HTTP Cookie File\n")
            # 将分号分隔的 cookie 字符串转换为每行一条
            for item in self.cookie_string.split(";"):
                item = item.strip()
                if item and "=" in item:
                    cookie_file.write(item + "\n")
            cookie_file.close()
            logger.debug(f"Cookie 文件已创建: {cookie_file.name}")
            return cookie_file.name
        except Exception as e:
            logger.warning(f"Cookie 文件创建失败: {e}")
            return ""

    @staticmethod
    def _get_random_ua() -> str:
        """随机选择一个 User-Agent"""
        return random.choice(USER_AGENTS)

    @staticmethod
    def _random_delay(min_seconds: float = 1.0, max_seconds: float = 3.0) -> None:
        """随机延迟，防止触发反爬机制"""
        delay = random.uniform(min_seconds, max_seconds)
        logger.debug(f"随机延迟 {delay:.1f}s")
        time.sleep(delay)

    def _is_video_downloaded(self, video_id: str) -> bool:
        """检查视频是否已下载"""
        return (self.video_dir / f"{video_id}.mp4").exists()

    def _get_downloaded_video_ids(self) -> set:
        """获取所有已下载视频的 ID 集合"""
        return {p.stem for p in self.video_dir.glob("*.mp4")}

    @staticmethod
    def _extract_douyin_id(input_str: str) -> str:
        """从输入中提取抖音号

        支持：
        - 纯抖音号: "douyin123456"
        - 完整 URL: "https://www.douyin.com/user/MS4wLjABAAAA..."
        """
        # 尝试匹配 /user/{id} 模式
        match = re.search(r"/user/([A-Za-z0-9_-]+)", input_str)
        if match:
            return match.group(1)

        # 如果是 URL 但没有匹配到标准模式，提取路径最后一段
        if input_str.startswith("http"):
            parts = input_str.rstrip("/").split("/")
            return parts[-1]

        # 纯抖音号，直接返回
        return input_str.strip()

    @staticmethod
    def _convert_date(yt_date: Optional[str]) -> str:
        """将 yt-dlp 的 YYYYMMDD 格式转换为 ISO 8601"""
        if not yt_date or len(yt_date) != 8:
            return yt_date or ""
        try:
            return f"{yt_date[:4]}-{yt_date[4:6]}-{yt_date[6:8]}T00:00:00"
        except (IndexError, ValueError):
            return yt_date

    def _build_yt_dlp_opts(self, output_path: str = "", download: bool = True) -> dict:
        """构建 yt-dlp 通用选项"""
        opts: Dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": not download,  # 获取列表时允许播放列表
            "extract_flat": False,
            "socket_timeout": self.timeout,
            "retries": self.max_retry,
            "fragment_retries": self.max_retry,
            "user_agent": self._get_random_ua(),
        }

        if self._cookie_file_path:
            opts["cookiefile"] = self._cookie_file_path

        if download:
            opts["outtmpl"] = output_path
            opts["format"] = "best[ext=mp4]/best[height<=1080]/best"

        return opts

    def _parse_video_info(self, info: dict, file_path: str = "") -> VideoInfo:
        """将 yt-dlp 提取的字典转换为 VideoInfo 模型"""
        return VideoInfo(
            video_id=str(info.get("id", "")),
            author=info.get("uploader") or info.get("channel") or "",
            author_id=info.get("uploader_id") or info.get("channel_id") or "",
            title=(info.get("title") or "").strip(),
            description=(info.get("description") or "").strip(),
            publish_time=self._convert_date(info.get("upload_date")),
            like_count=int(info.get("like_count") or 0),
            comment_count=int(info.get("comment_count") or 0),
            collect_count=int(info.get("favorite_count") or info.get("collect_count") or 0),
            share_count=int(info.get("repost_count") or 0),
            duration=int(info.get("duration") or 0),
            file_path=file_path,
        )

    # ---------- 核心下载逻辑 ----------

    def _download_single_video(self, video_url: str) -> dict:
        """下载单个视频，返回包含 success/error 的结构化字典

        包含完整的重试、反爬处理、临时文件管理逻辑。
        """
        result: Dict[str, Any] = {"success": False, "video_info": None, "error": None}
        temp_cookie_file: Optional[str] = None

        # 先提取信息（不下载）
        info_opts = self._build_yt_dlp_opts("", download=False)

        for attempt in range(1, self.max_retry + 1):
            info = None
            try:
                logger.info(f"[下载] 提取信息: {video_url} (第 {attempt}/{self.max_retry} 次)")

                with yt_dlp.YoutubeDL(info_opts) as ydl:
                    info = ydl.extract_info(video_url, download=False)

                video_id = str(info.get("id", ""))
                if not video_id:
                    result["error"] = "无法提取视频ID"
                    logger.error(f"[下载] 无法提取视频ID: {video_url}")
                    return result

                logger.info(f"[下载] 视频ID: {video_id} | 标题: {info.get('title', 'N/A')[:50]}")

                # 检查是否已下载
                if self._is_video_downloaded(video_id):
                    final_path = str(self.video_dir / f"{video_id}.mp4")
                    logger.info(f"[下载] 视频已存在，跳过下载: {video_id}")
                    result["success"] = True
                    result["video_info"] = self._parse_video_info(info, final_path)
                    return result

                # 开始下载
                tmp_path = str(self.video_dir / f"{video_id}.tmp.mp4")
                final_path = str(self.video_dir / f"{video_id}.mp4")

                download_opts = self._build_yt_dlp_opts(tmp_path, download=True)
                download_opts["user_agent"] = self._get_random_ua()  # 每次下载换 UA

                logger.info(f"[下载] 开始下载: {video_id} -> {tmp_path}")

                with yt_dlp.YoutubeDL(download_opts) as ydl:
                    ydl.download([video_url])

                # 下载完成后重命名
                tmp_file = Path(tmp_path)
                if tmp_file.exists():
                    tmp_file.rename(final_path)
                    logger.info(f"[下载] 完成: {final_path}")
                else:
                    # yt-dlp 可能直接以 final_path 保存
                    logger.warning(f"[下载] 临时文件不存在，检查是否直接保存到目标路径")

                result["success"] = True
                result["video_info"] = self._parse_video_info(info, final_path)
                return result

            except yt_dlp.utils.DownloadError as e:
                error_str = str(e)
                logger.warning(f"[下载] 失败 (第 {attempt} 次): {error_str[:200]}")

                # 反爬错误处理
                if any(code in error_str for code in ["403", "429"]):
                    sleep_sec = 10 * attempt
                    logger.warning(f"[下载] 触发反爬 ({error_str[:50]})，休眠 {sleep_sec}s")
                    time.sleep(sleep_sec)
                else:
                    wait = 2**attempt
                    logger.info(f"[下载] 等待 {wait}s 后重试")
                    time.sleep(wait)

                # 清理失败的临时文件
                if info:
                    vid = str(info.get("id", ""))
                    tmp = self.video_dir / f"{vid}.tmp.mp4"
                    if tmp.exists():
                        tmp.unlink()
                        logger.debug(f"[下载] 已清理临时文件: {tmp}")

            except Exception as e:
                logger.error(f"[下载] 未预期错误 (第 {attempt} 次): {type(e).__name__}: {e}")
                time.sleep(2**attempt)

                # 清理可能的临时文件
                if info:
                    vid = str(info.get("id", ""))
                    tmp = self.video_dir / f"{vid}.tmp.mp4"
                    if tmp.exists():
                        tmp.unlink()

        result["error"] = f"下载失败，已重试 {self.max_retry} 次"
        logger.error(f"[下载] 最终失败: {video_url}")
        return result

    # ========== 对外暴露的方法（内部实现） ==========

    def crawl_videos_impl(self, douyin_id: str, count: int = 5) -> List[VideoInfo]:
        """爬取指定博主的最新视频

        Args:
            douyin_id: 抖音号或主页链接
            count: 爬取视频数量（1-20）

        Returns:
            视频元数据列表
        """
        logger.info(f"[爬取] 开始 | douyin_id={douyin_id}, count={count}")

        count = max(1, min(count, 20))

        try:
            clean_id = self._extract_douyin_id(douyin_id)
            user_url = f"https://www.douyin.com/user/{clean_id}"

            logger.info(f"[爬取] 解析URL: {user_url}")

            # 获取用户视频列表
            flat_opts = self._build_yt_dlp_opts("", download=False)
            flat_opts["extract_flat"] = "in_playlist"
            flat_opts["playlistend"] = count

            entries: List[dict] = []
            with yt_dlp.YoutubeDL(flat_opts) as ydl:
                try:
                    info = ydl.extract_info(user_url, download=False)
                except Exception as e:
                    logger.warning(f"[爬取] 扁平提取失败，尝试完整提取: {e}")
                    full_opts = self._build_yt_dlp_opts("", download=False)
                    full_opts["playlistend"] = count
                    with yt_dlp.YoutubeDL(full_opts) as ydl2:
                        info = ydl2.extract_info(user_url, download=False)

            if info:
                if "entries" in info:
                    entries = list(info["entries"])[:count]
                else:
                    entries = [info]

            logger.info(f"[爬取] 获取到 {len(entries)} 个视频条目")

            results: List[VideoInfo] = []
            downloaded_ids = self._get_downloaded_video_ids()
            skipped_count = 0

            for i, entry in enumerate(entries):
                video_id = str(entry.get("id", ""))

                if not video_id:
                    logger.warning(f"[爬取] 条目 {i+1} 无ID，跳过")
                    continue

                # 断点续爬：跳过已下载
                if video_id in downloaded_ids:
                    final_path = str(self.video_dir / f"{video_id}.mp4")
                    info_obj = self._parse_video_info(entry, final_path)
                    results.append(info_obj)
                    skipped_count += 1
                    logger.info(f"[爬取] [{i+1}/{len(entries)}] 已存在，跳过: {video_id}")
                    continue

                # 请求前延迟
                self._random_delay(0.5, 1.5)

                # 构建完整 URL 并下载
                video_url = entry.get("webpage_url") or entry.get("url") or f"https://www.douyin.com/video/{video_id}"
                logger.info(f"[爬取] [{i+1}/{len(entries)}] 下载: {video_id}")

                download_result = self._download_single_video(video_url)

                if download_result["success"] and download_result["video_info"]:
                    results.append(download_result["video_info"])
                    downloaded_ids.add(video_id)
                else:
                    logger.error(f"[爬取] [{i+1}/{len(entries)}] 失败: {download_result.get('error')}")

                # 视频间延迟
                self._random_delay(2.0, 5.0)

            logger.info(
                f"[爬取] 完成 | 成功: {len(results)} | 跳过: {skipped_count} | 失败: {len(entries) - len(results)}"
            )
            return results

        except Exception as e:
            logger.error(f"[爬取] 异常: {type(e).__name__}: {e}")
            return []

    def get_video_info_impl(self, video_url: str) -> VideoInfo:
        """获取单个视频元信息（不下载视频文件）

        Args:
            video_url: 抖音视频链接

        Returns:
            VideoInfo 对象，失败时字段均为默认值
        """
        logger.info(f"[元信息] 获取: {video_url}")

        try:
            opts = self._build_yt_dlp_opts("", download=False)

            for attempt in range(1, self.max_retry + 1):
                try:
                    with yt_dlp.YoutubeDL(opts) as ydl:
                        info = ydl.extract_info(video_url, download=False)

                    video_id = str(info.get("id", ""))
                    logger.info(f"[元信息] 成功: {video_id}")

                    # 检查是否有本地文件
                    local_path = ""
                    if video_id and self._is_video_downloaded(video_id):
                        local_path = str(self.video_dir / f"{video_id}.mp4")

                    return self._parse_video_info(info, local_path)

                except Exception as e:
                    logger.warning(f"[元信息] 失败 (第 {attempt} 次): {e}")
                    if attempt < self.max_retry:
                        time.sleep(2**attempt)

            logger.error(f"[元信息] 最终失败，已重试 {self.max_retry} 次")
            return VideoInfo()

        except Exception as e:
            logger.error(f"[元信息] 异常: {type(e).__name__}: {e}")
            return VideoInfo()

    def delete_video_impl(self, video_id: str) -> bool:
        """删除本地视频文件

        Args:
            video_id: 视频ID

        Returns:
            是否成功删除
        """
        logger.info(f"[删除] 视频: {video_id}")

        try:
            video_path = self.video_dir / f"{video_id}.mp4"
            tmp_path = self.video_dir / f"{video_id}.tmp.mp4"

            deleted = False

            if video_path.exists():
                video_path.unlink()
                logger.info(f"[删除] 已删除: {video_path}")
                deleted = True

            if tmp_path.exists():
                tmp_path.unlink()
                logger.debug(f"[删除] 已清理临时文件: {tmp_path}")
                deleted = True

            if not deleted:
                logger.warning(f"[删除] 文件不存在: {video_id}")

            return deleted

        except Exception as e:
            logger.error(f"[删除] 失败: {type(e).__name__}: {e}")
            return False

    # ========== 兼容旧版 crawler_agent 接口 ==========

    def download_video(self, url: str) -> dict:
        """下载单个视频，返回 dict（兼容 crawler_agent 接口）"""
        result = self._download_single_video(url)
        if result["success"] and result["video_info"]:
            return result["video_info"].model_dump()
        return {"error": result.get("error", "下载失败")}

    def batch_download(self, urls: List[str]) -> List[dict]:
        """批量下载视频（兼容 crawler_agent 接口）"""
        return [self.download_video(url) for url in urls]


# ============ LangChain @tool 工具函数 ============

_crawler_singleton: Optional[DouyinCrawler] = None


def _get_crawler() -> DouyinCrawler:
    """获取 DouyinCrawler 单例"""
    global _crawler_singleton
    if _crawler_singleton is None:
        _crawler_singleton = DouyinCrawler()
    return _crawler_singleton


@tool
def crawl_videos(douyin_id: str, count: int = 5) -> str:
    """爬取指定抖音博主的最新视频列表。

    支持纯抖音号（如 "douyin123456"）或完整主页链接。
    自动跳过已下载的视频，支持断点续爬。

    Args:
        douyin_id: 抖音号或主页链接，如 "MS4wLjABAAAAxxx" 或 "https://www.douyin.com/user/MS4w..."
        count: 爬取数量，默认5，最多20

    Returns:
        JSON字符串，格式: {"success": bool, "data": [...], "count": int, "error": str|null}
    """
    try:
        crawler = _get_crawler()
        results = crawler.crawl_videos_impl(douyin_id, count)
        data = [r.model_dump() for r in results]
        return json.dumps(
            {"success": True, "data": data, "count": len(data), "error": None},
            ensure_ascii=False,
            indent=2,
        )
    except Exception as e:
        logger.error(f"[tool:crawl_videos] 执行失败: {type(e).__name__}: {e}")
        return json.dumps(
            {"success": False, "data": [], "count": 0, "error": str(e)},
            ensure_ascii=False,
        )


@tool
def get_video_info(video_url: str) -> str:
    """获取单个抖音视频的元数据信息，不会下载视频文件。

    Args:
        video_url: 抖音视频完整链接

    Returns:
        JSON字符串，格式: {"success": bool, "data": {...}, "error": str|null}
    """
    try:
        crawler = _get_crawler()
        result = crawler.get_video_info_impl(video_url)
        return json.dumps(
            {"success": True, "data": result.model_dump(), "error": None},
            ensure_ascii=False,
            indent=2,
        )
    except Exception as e:
        logger.error(f"[tool:get_video_info] 执行失败: {type(e).__name__}: {e}")
        return json.dumps(
            {"success": False, "data": {}, "error": str(e)},
            ensure_ascii=False,
        )


@tool
def delete_video(video_id: str) -> str:
    """删除本地已下载的视频文件。

    Args:
        video_id: 要删除的视频唯一ID

    Returns:
        JSON字符串，格式: {"success": bool, "deleted": bool, "video_id": str, "error": str|null}
    """
    try:
        crawler = _get_crawler()
        result = crawler.delete_video_impl(video_id)
        return json.dumps(
            {"success": result, "deleted": result, "video_id": video_id, "error": None},
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"[tool:delete_video] 执行失败: {type(e).__name__}: {e}")
        return json.dumps(
            {"success": False, "deleted": False, "video_id": video_id, "error": str(e)},
            ensure_ascii=False,
        )


@tool
def download_single_video(video_url: str) -> str:
    """下载单个抖音视频文件到本地。

    下载完成后返回本地文件路径和视频元数据。
    如果视频已存在则跳过下载。

    Args:
        video_url: 抖音视频完整链接

    Returns:
        JSON字符串，格式: {"success": bool, "file_path": str, "video_id": str,
                          "title": str, "duration": int, "error": str|null}
    """
    try:
        crawler = _get_crawler()
        result = crawler._download_single_video(video_url)
        if result.get("success"):
            info = result.get("video_info", {})
            return json.dumps({
                "success": True,
                "file_path": getattr(info, "file_path", "") if not isinstance(info, dict) else info.get("local_path", ""),
                "video_id": getattr(info, "video_id", "") if not isinstance(info, dict) else info.get("video_id", ""),
                "title": getattr(info, "title", "") if not isinstance(info, dict) else info.get("title", ""),
                "duration": getattr(info, "duration", 0) if not isinstance(info, dict) else info.get("duration", 0),
                "error": None,
            }, ensure_ascii=False)
        return json.dumps({
            "success": False, "file_path": "", "video_id": "",
            "title": "", "duration": 0, "error": result.get("error", "下载失败"),
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"[tool:download_single_video] 执行失败: {type(e).__name__}: {e}")
        return json.dumps({
            "success": False, "file_path": "", "video_id": "",
            "title": "", "duration": 0, "error": str(e),
        }, ensure_ascii=False)


@tool
def download_video_audio(video_url: str) -> str:
    """下载抖音视频的音频文件（MP3 格式）到本地。

    下载的视频音频文件可用于语音转文字（SpeechToText）处理。
    音频文件保存在 data/videos/ 目录下，文件名为 {video_id}.mp3。

    Args:
        video_url: 抖音视频完整链接

    Returns:
        JSON字符串，格式: {"success": bool, "audio_path": str, "video_id": str,
                          "title": str, "error": str|null}
    """
    import yt_dlp
    from pathlib import Path

    audio_dir = Path("./data/videos")
    audio_dir.mkdir(parents=True, exist_ok=True)

    try:
        crawler = _get_crawler()

        # 先获取视频信息
        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info = ydl.extract_info(video_url, download=False)

        video_id = str(info.get("id", ""))
        title = str(info.get("title", "未知"))

        audio_path = str(audio_dir / f"{video_id}.mp3")

        # 如果已存在则跳过
        if Path(audio_path).exists():
            logger.info(f"[tool:download_audio] 音频已存在: {audio_path}")
            return json.dumps({
                "success": True,
                "audio_path": audio_path,
                "video_id": video_id,
                "title": title,
                "error": None,
            }, ensure_ascii=False)

        # 下载音频
        opts = {
            "format": "bestaudio/best",
            "outtmpl": str(audio_dir / f"{video_id}.%(ext)s"),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
            "quiet": True,
            "user_agent": crawler._get_random_ua(),
        }

        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([video_url])

        logger.info(f"[tool:download_audio] 下载完成: {audio_path}")
        return json.dumps({
            "success": True,
            "audio_path": audio_path,
            "video_id": video_id,
            "title": title,
            "error": None,
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"[tool:download_audio] 执行失败: {type(e).__name__}: {e}")
        return json.dumps({
            "success": False,
            "audio_path": "",
            "video_id": "",
            "title": "",
            "error": str(e),
        }, ensure_ascii=False)