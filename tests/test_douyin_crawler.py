"""
tools/douyin_crawler.py 单元测试

测试覆盖：
- VideoInfo 数据模型
- DouyinCrawler 初始化
- 抖音号提取
- 已下载检测
- 视频元数据解析
- 视频删除
- @tool 工具函数
- 错误处理
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.douyin_crawler import (
    DouyinCrawler,
    VideoInfo,
    USER_AGENTS,
    _get_crawler,
    _crawler_singleton,
    crawl_videos,
    get_video_info,
    delete_video,
)


class TestVideoInfo(unittest.TestCase):
    """VideoInfo 数据模型测试"""

    def test_default_values(self):
        """测试默认值"""
        info = VideoInfo()
        self.assertEqual(info.video_id, "")
        self.assertEqual(info.author, "")
        self.assertEqual(info.like_count, 0)
        self.assertEqual(info.comment_count, 0)
        self.assertEqual(info.collect_count, 0)
        self.assertEqual(info.share_count, 0)
        self.assertEqual(info.duration, 0)
        self.assertEqual(info.file_path, "")

    def test_full_creation(self):
        """测试完整字段赋值"""
        info = VideoInfo(
            video_id="123456",
            author="测试博主",
            author_id="douyin_test",
            title="测试视频标题",
            description="这是一个测试视频",
            publish_time="2024-01-15T10:30:00",
            like_count=1000,
            comment_count=200,
            collect_count=300,
            share_count=50,
            duration=60,
            file_path="/data/videos/123456.mp4",
        )
        self.assertEqual(info.video_id, "123456")
        self.assertEqual(info.author, "测试博主")
        self.assertEqual(info.like_count, 1000)
        self.assertEqual(info.duration, 60)

    def test_model_dump(self):
        """测试序列化"""
        info = VideoInfo(video_id="abc", title="test")
        dumped = info.model_dump()
        self.assertIsInstance(dumped, dict)
        self.assertEqual(dumped["video_id"], "abc")
        self.assertEqual(dumped["title"], "test")

    def test_json_serializable(self):
        """测试 JSON 序列化"""
        info = VideoInfo(video_id="abc", like_count=100)
        json_str = json.dumps(info.model_dump(), ensure_ascii=False)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["video_id"], "abc")
        self.assertEqual(parsed["like_count"], 100)


class TestDouyinCrawlerInit(unittest.TestCase):
    """DouyinCrawler 初始化测试"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_video_path = os.environ.get("VIDEO_STORAGE_PATH", "")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_init_creates_video_dir(self):
        """测试初始化创建视频目录"""
        with patch("tools.douyin_crawler.settings") as mock_settings:
            mock_settings.VIDEO_STORAGE_PATH = Path(self.temp_dir.name) / "videos"
            mock_settings.DOUYIN_COOKIE = ""
            mock_settings.MAX_RETRY = 3
            mock_settings.TIMEOUT = 30

            crawler = DouyinCrawler()
            video_dir = Path(self.temp_dir.name) / "videos"
            self.assertTrue(video_dir.exists())

    def test_init_with_cookie(self):
        """测试带 Cookie 初始化"""
        with patch("tools.douyin_crawler.settings") as mock_settings:
            mock_settings.VIDEO_STORAGE_PATH = Path(self.temp_dir.name) / "videos"
            mock_settings.DOUYIN_COOKIE = "sessionid=abc123; token=xyz789"
            mock_settings.MAX_RETRY = 3
            mock_settings.TIMEOUT = 30

            crawler = DouyinCrawler()
            self.assertEqual(crawler.cookie_string, "sessionid=abc123; token=xyz789")
            self.assertIsNotNone(crawler._cookie_file_path)

            # 清理
            if crawler._cookie_file_path:
                Path(crawler._cookie_file_path).unlink(missing_ok=True)

    def test_init_without_cookie(self):
        """测试无 Cookie 初始化"""
        with patch("tools.douyin_crawler.settings") as mock_settings:
            mock_settings.VIDEO_STORAGE_PATH = Path(self.temp_dir.name) / "videos"
            mock_settings.DOUYIN_COOKIE = ""
            mock_settings.MAX_RETRY = 3
            mock_settings.TIMEOUT = 30

            crawler = DouyinCrawler()
            self.assertEqual(crawler.cookie_string, "")
            self.assertIsNone(crawler._cookie_file_path)


class TestDouyinCrawlerUtilities(unittest.TestCase):
    """DouyinCrawler 工具方法测试"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_crawler(self):
        with patch("tools.douyin_crawler.settings") as mock_settings:
            mock_settings.VIDEO_STORAGE_PATH = Path(self.temp_dir.name) / "videos"
            mock_settings.DOUYIN_COOKIE = ""
            mock_settings.MAX_RETRY = 3
            mock_settings.TIMEOUT = 30
            return DouyinCrawler()

    def test_extract_douyin_id_from_url(self):
        """测试从完整URL提取抖音号"""
        crawler = self._create_crawler()
        result = crawler._extract_douyin_id(
            "https://www.douyin.com/user/MS4wLjABAAAAxxxxx"
        )
        self.assertEqual(result, "MS4wLjABAAAAxxxxx")

    def test_extract_douyin_id_pure(self):
        """测试纯抖音号"""
        crawler = self._create_crawler()
        result = crawler._extract_douyin_id("douyin123456")
        self.assertEqual(result, "douyin123456")

    def test_extract_douyin_id_with_query(self):
        """测试带查询参数的URL"""
        crawler = self._create_crawler()
        result = crawler._extract_douyin_id(
            "https://www.douyin.com/user/TEST_ID?modal=following"
        )
        self.assertEqual(result, "TEST_ID")

    def test_extract_douyin_id_other_url(self):
        """测试其他格式URL"""
        crawler = self._create_crawler()
        result = crawler._extract_douyin_id(
            "https://v.douyin.com/user/abc123/"
        )
        self.assertEqual(result, "abc123")

    def test_random_ua(self):
        """测试随机UA"""
        ua = DouyinCrawler._get_random_ua()
        self.assertIn(ua, USER_AGENTS)

    def test_random_ua_many(self):
        """测试多次随机UA"""
        uas = [DouyinCrawler._get_random_ua() for _ in range(20)]
        # 至少覆盖了多个不同的UA
        unique_uas = set(uas)
        self.assertGreater(len(unique_uas), 1)

    def test_convert_date_valid(self):
        """测试日期格式转换"""
        result = DouyinCrawler._convert_date("20240115")
        self.assertEqual(result, "2024-01-15T00:00:00")

    def test_convert_date_none(self):
        """测试空日期"""
        result = DouyinCrawler._convert_date(None)
        self.assertEqual(result, "")

    def test_convert_date_empty(self):
        """测试空字符串日期"""
        result = DouyinCrawler._convert_date("")
        self.assertEqual(result, "")

    def test_convert_date_invalid(self):
        """测试非法日期"""
        result = DouyinCrawler._convert_date("not_a_date")
        self.assertEqual(result, "not_a_date")

    def test_is_video_downloaded_true(self):
        """测试检测已下载视频"""
        crawler = self._create_crawler()
        video_id = "test_video_123"
        video_path = crawler.video_dir / f"{video_id}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.touch()

        self.assertTrue(crawler._is_video_downloaded(video_id))

    def test_is_video_downloaded_false(self):
        """测试检测未下载视频"""
        crawler = self._create_crawler()
        self.assertFalse(crawler._is_video_downloaded("nonexistent_video"))

    def test_get_downloaded_video_ids(self):
        """测试获取已下载视频ID列表"""
        crawler = self._create_crawler()
        crawler.video_dir.mkdir(parents=True, exist_ok=True)

        (crawler.video_dir / "vid1.mp4").touch()
        (crawler.video_dir / "vid2.mp4").touch()
        (crawler.video_dir / "not_a_video.txt").touch()

        ids = crawler._get_downloaded_video_ids()
        self.assertIn("vid1", ids)
        self.assertIn("vid2", ids)
        self.assertNotIn("not_a_video", ids)

    def test_parse_video_info_complete(self):
        """测试完整元数据解析"""
        crawler = self._create_crawler()
        raw = {
            "id": "7123456789",
            "uploader": "测试博主",
            "uploader_id": "test_douyin_id",
            "title": "你好世界",
            "description": "视频描述内容",
            "upload_date": "20240115",
            "like_count": 5000,
            "comment_count": 300,
            "favorite_count": 200,
            "repost_count": 100,
            "duration": 45,
        }
        info = crawler._parse_video_info(raw, "/data/videos/7123456789.mp4")

        self.assertEqual(info.video_id, "7123456789")
        self.assertEqual(info.author, "测试博主")
        self.assertEqual(info.author_id, "test_douyin_id")
        self.assertEqual(info.title, "你好世界")
        self.assertEqual(info.like_count, 5000)
        self.assertEqual(info.comment_count, 300)
        self.assertEqual(info.collect_count, 200)
        self.assertEqual(info.share_count, 100)
        self.assertEqual(info.duration, 45)
        self.assertEqual(info.publish_time, "2024-01-15T00:00:00")
        self.assertEqual(info.file_path, "/data/videos/7123456789.mp4")

    def test_parse_video_info_incomplete(self):
        """测试不完整元数据解析（缺失字段用默认值）"""
        crawler = self._create_crawler()
        raw = {"id": "minimal_id", "title": "最小视频"}
        info = crawler._parse_video_info(raw)

        self.assertEqual(info.video_id, "minimal_id")
        self.assertEqual(info.title, "最小视频")
        self.assertEqual(info.author, "")
        self.assertEqual(info.like_count, 0)
        self.assertEqual(info.comment_count, 0)


class TestDeleteVideo(unittest.TestCase):
    """删除视频功能测试"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_crawler(self):
        with patch("tools.douyin_crawler.settings") as mock_settings:
            mock_settings.VIDEO_STORAGE_PATH = Path(self.temp_dir.name) / "videos"
            mock_settings.DOUYIN_COOKIE = ""
            mock_settings.MAX_RETRY = 3
            mock_settings.TIMEOUT = 30
            return DouyinCrawler()

    def test_delete_existing_video(self):
        """测试删除存在的视频"""
        crawler = self._create_crawler()
        video_id = "delete_test_123"
        video_path = crawler.video_dir / f"{video_id}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.touch()

        self.assertTrue(video_path.exists())
        result = crawler.delete_video_impl(video_id)
        self.assertTrue(result)
        self.assertFalse(video_path.exists())

    def test_delete_nonexistent_video(self):
        """测试删除不存在的视频"""
        crawler = self._create_crawler()
        result = crawler.delete_video_impl("nonexistent_999")
        self.assertFalse(result)

    def test_delete_tmp_file(self):
        """测试删除临时文件"""
        crawler = self._create_crawler()
        video_id = "tmp_test_456"
        tmp_path = crawler.video_dir / f"{video_id}.tmp.mp4"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.touch()

        result = crawler.delete_video_impl(video_id)
        self.assertTrue(result)
        self.assertFalse(tmp_path.exists())

    def test_delete_both_files(self):
        """测试同时删除正式文件和临时文件"""
        crawler = self._create_crawler()
        video_id = "both_test_789"
        crawler.video_dir.mkdir(parents=True, exist_ok=True)
        (crawler.video_dir / f"{video_id}.mp4").touch()
        (crawler.video_dir / f"{video_id}.tmp.mp4").touch()

        result = crawler.delete_video_impl(video_id)
        self.assertTrue(result)
        self.assertFalse((crawler.video_dir / f"{video_id}.mp4").exists())
        self.assertFalse((crawler.video_dir / f"{video_id}.tmp.mp4").exists())


class TestToolFunctions(unittest.TestCase):
    """@tool 工具函数测试"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()
        # 重置单例
        import tools.douyin_crawler as crawler_module
        crawler_module._crawler_singleton = None

    def _patch_crawler(self):
        """创建 mock crawler 并 patch 单例"""
        with patch("tools.douyin_crawler.settings") as mock_settings:
            mock_settings.VIDEO_STORAGE_PATH = Path(self.temp_dir.name) / "videos"
            mock_settings.DOUYIN_COOKIE = ""
            mock_settings.MAX_RETRY = 3
            mock_settings.TIMEOUT = 30
            crawler = DouyinCrawler()
            return crawler

    def test_crawl_videos_tool_success(self):
        """测试 crawl_videos @tool 成功场景"""
        crawler = self._patch_crawler()
        mock_results = [
            VideoInfo(video_id="vid1", title="视频1", author="作者1"),
            VideoInfo(video_id="vid2", title="视频2", author="作者2"),
        ]
        crawler.crawl_videos_impl = MagicMock(return_value=mock_results)

        import tools.douyin_crawler as crawler_module
        crawler_module._crawler_singleton = crawler

        result_str = crawl_videos("douyin_test", count=3)
        result = json.loads(result_str)

        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 2)
        self.assertEqual(len(result["data"]), 2)
        self.assertEqual(result["data"][0]["video_id"], "vid1")
        self.assertIsNone(result["error"])

    def test_crawl_videos_tool_exception(self):
        """测试 crawl_videos @tool 异常场景"""
        crawler = self._patch_crawler()
        crawler.crawl_videos_impl = MagicMock(side_effect=RuntimeError("网络错误"))

        import tools.douyin_crawler as crawler_module
        crawler_module._crawler_singleton = crawler

        result_str = crawl_videos("test_id", count=5)
        result = json.loads(result_str)

        self.assertFalse(result["success"])
        self.assertEqual(result["count"], 0)
        self.assertEqual(len(result["data"]), 0)
        self.assertIsNotNone(result["error"])
        self.assertIn("网络错误", result["error"])

    def test_get_video_info_tool_success(self):
        """测试 get_video_info @tool 成功场景"""
        crawler = self._patch_crawler()
        mock_info = VideoInfo(
            video_id="single_vid",
            title="单个视频",
            author="作者",
            like_count=100,
        )
        crawler.get_video_info_impl = MagicMock(return_value=mock_info)

        import tools.douyin_crawler as crawler_module
        crawler_module._crawler_singleton = crawler

        result_str = get_video_info("https://www.douyin.com/video/123")
        result = json.loads(result_str)

        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["video_id"], "single_vid")
        self.assertEqual(result["data"]["title"], "单个视频")
        self.assertEqual(result["data"]["like_count"], 100)

    def test_get_video_info_tool_exception(self):
        """测试 get_video_info @tool 异常场景"""
        crawler = self._patch_crawler()
        crawler.get_video_info_impl = MagicMock(side_effect=ValueError("无效URL"))

        import tools.douyin_crawler as crawler_module
        crawler_module._crawler_singleton = crawler

        result_str = get_video_info("invalid_url")
        result = json.loads(result_str)

        self.assertFalse(result["success"])
        self.assertIsNotNone(result["error"])

    def test_delete_video_tool_success(self):
        """测试 delete_video @tool 成功场景"""
        crawler = self._patch_crawler()
        crawler.delete_video_impl = MagicMock(return_value=True)

        import tools.douyin_crawler as crawler_module
        crawler_module._crawler_singleton = crawler

        result_str = delete_video("vid_to_delete")
        result = json.loads(result_str)

        self.assertTrue(result["success"])
        self.assertTrue(result["deleted"])
        self.assertEqual(result["video_id"], "vid_to_delete")

    def test_delete_video_tool_failure(self):
        """测试 delete_video @tool 文件不存在场景"""
        crawler = self._patch_crawler()
        crawler.delete_video_impl = MagicMock(return_value=False)

        import tools.douyin_crawler as crawler_module
        crawler_module._crawler_singleton = crawler

        result_str = delete_video("nonexistent")
        result = json.loads(result_str)

        self.assertFalse(result["deleted"])
        self.assertEqual(result["video_id"], "nonexistent")

    def test_delete_video_tool_exception(self):
        """测试 delete_video @tool 异常场景"""
        crawler = self._patch_crawler()
        crawler.delete_video_impl = MagicMock(side_effect=OSError("权限不足"))

        import tools.douyin_crawler as crawler_module
        crawler_module._crawler_singleton = crawler

        result_str = delete_video("protected_vid")
        result = json.loads(result_str)

        self.assertFalse(result["success"])
        self.assertFalse(result["deleted"])
        self.assertIn("权限不足", result["error"])


class TestDownloadVideoCompat(unittest.TestCase):
    """兼容 crawler_agent 的 download_video / batch_download 接口测试"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_crawler(self):
        with patch("tools.douyin_crawler.settings") as mock_settings:
            mock_settings.VIDEO_STORAGE_PATH = Path(self.temp_dir.name) / "videos"
            mock_settings.DOUYIN_COOKIE = ""
            mock_settings.MAX_RETRY = 3
            mock_settings.TIMEOUT = 30
            return DouyinCrawler()

    def test_download_video_returns_dict(self):
        """测试 download_video 返回字典格式"""
        crawler = self._create_crawler()
        mock_info = VideoInfo(
            video_id="v123", title="测试", author="作者", like_count=10
        )
        crawler._download_single_video = MagicMock(
            return_value={"success": True, "video_info": mock_info, "error": None}
        )

        result = crawler.download_video("https://example.com/video")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["video_id"], "v123")
        self.assertEqual(result["title"], "测试")

    def test_download_video_error_returns_dict(self):
        """测试下载失败时返回错误字典"""
        crawler = self._create_crawler()
        crawler._download_single_video = MagicMock(
            return_value={"success": False, "video_info": None, "error": "下载失败"}
        )

        result = crawler.download_video("https://example.com/video")
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)
        self.assertEqual(result["error"], "下载失败")

    def test_batch_download(self):
        """测试批量下载"""
        crawler = self._create_crawler()
        mock_info = VideoInfo(video_id="v1", title="视频1")
        crawler.download_video = MagicMock(return_value=mock_info.model_dump())

        urls = ["https://a.com/1", "https://a.com/2", "https://a.com/3"]
        results = crawler.batch_download(urls)

        self.assertEqual(len(results), 3)
        self.assertEqual(crawler.download_video.call_count, 3)


class TestCountValidation(unittest.TestCase):
    """count 参数边界值测试"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_crawler(self):
        with patch("tools.douyin_crawler.settings") as mock_settings:
            mock_settings.VIDEO_STORAGE_PATH = Path(self.temp_dir.name) / "videos"
            mock_settings.DOUYIN_COOKIE = ""
            mock_settings.MAX_RETRY = 3
            mock_settings.TIMEOUT = 30
            return DouyinCrawler()

    def test_count_default(self):
        """测试默认 count=5"""
        crawler = self._create_crawler()
        crawler._download_single_video = MagicMock(
            return_value={"success": False, "video_info": None, "error": "模拟"}
        )
        with patch.object(crawler, "_build_yt_dlp_opts", return_value={"quiet": True}):
            with patch("yt_dlp.YoutubeDL") as mock_ydl:
                mock_ydl_instance = MagicMock()
                mock_ydl_instance.extract_info.return_value = {
                    "entries": [{"id": f"vid_{i}", "title": f"视频{i}"} for i in range(10)]
                }
                mock_ydl.return_value.__enter__.return_value = mock_ydl_instance

                results = crawler.crawl_videos_impl("test_id", count=5)
                # 最多取5个
                self.assertLessEqual(len(results), 5)

    def test_count_minimum(self):
        """测试 count 最小值1"""
        crawler = self._create_crawler()
        crawler._download_single_video = MagicMock(
            return_value={"success": False, "video_info": None, "error": "模拟"}
        )
        with patch.object(crawler, "_build_yt_dlp_opts", return_value={"quiet": True}):
            with patch("yt_dlp.YoutubeDL") as mock_ydl:
                mock_ydl_instance = MagicMock()
                mock_ydl_instance.extract_info.return_value = {
                    "entries": [{"id": f"vid_{i}"} for i in range(3)]
                }
                mock_ydl.return_value.__enter__.return_value = mock_ydl_instance

                results = crawler.crawl_videos_impl("test_id", count=0)
                self.assertLessEqual(len(results), 1)

    def test_count_maximum(self):
        """测试 count 最大值20"""
        crawler = self._create_crawler()
        crawler._download_single_video = MagicMock(
            return_value={"success": False, "video_info": None, "error": "模拟"}
        )
        with patch.object(crawler, "_build_yt_dlp_opts", return_value={"quiet": True}):
            with patch("yt_dlp.YoutubeDL") as mock_ydl:
                mock_ydl_instance = MagicMock()
                mock_ydl_instance.extract_info.return_value = {
                    "entries": [{"id": f"vid_{i}"} for i in range(30)]
                }
                mock_ydl.return_value.__enter__.return_value = mock_ydl_instance

                results = crawler.crawl_videos_impl("test_id", count=50)
                self.assertLessEqual(len(results), 20)


class TestErrorHandling(unittest.TestCase):
    """错误处理与日志记录测试"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_crawler(self):
        with patch("tools.douyin_crawler.settings") as mock_settings:
            mock_settings.VIDEO_STORAGE_PATH = Path(self.temp_dir.name) / "videos"
            mock_settings.DOUYIN_COOKIE = ""
            mock_settings.MAX_RETRY = 3
            mock_settings.TIMEOUT = 30
            return DouyinCrawler()

    def test_crawl_videos_impl_handles_exception(self):
        """测试 crawl_videos_impl 异常时返回空列表"""
        crawler = self._create_crawler()
        crawler._extract_douyin_id = MagicMock(side_effect=RuntimeError("解析失败"))

        with patch("tools.douyin_crawler.logger") as mock_logger:
            results = crawler.crawl_videos_impl("bad_input", count=5)
            self.assertEqual(results, [])
            mock_logger.error.assert_called()

    def test_get_video_info_impl_handles_exception(self):
        """测试 get_video_info_impl 异常时返回空 VideoInfo"""
        crawler = self._create_crawler()
        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_ydl.side_effect = RuntimeError("连接超时")

            result = crawler.get_video_info_impl("https://example.com")
            self.assertIsInstance(result, VideoInfo)
            self.assertEqual(result.video_id, "")

    def test_delete_video_impl_handles_exception(self):
        """测试 delete_video_impl 异常时返回 False"""
        crawler = self._create_crawler()
        with patch("pathlib.Path.exists", side_effect=PermissionError("无权限")):
            result = crawler.delete_video_impl("some_id")
            self.assertFalse(result)

    def test_download_single_video_no_id(self):
        """测试无法提取视频ID时返回错误"""
        crawler = self._create_crawler()
        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_ydl_instance = MagicMock()
            mock_ydl_instance.extract_info.return_value = {"id": "", "title": ""}
            mock_ydl.return_value.__enter__.return_value = mock_ydl_instance

            result = crawler._download_single_video("https://example.com")
            self.assertFalse(result["success"])
            self.assertEqual(result["error"], "无法提取视频ID")


class TestDuplicateSkip(unittest.TestCase):
    """断点续爬/重复跳过测试"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_crawler(self):
        with patch("tools.douyin_crawler.settings") as mock_settings:
            mock_settings.VIDEO_STORAGE_PATH = Path(self.temp_dir.name) / "videos"
            mock_settings.DOUYIN_COOKIE = ""
            mock_settings.MAX_RETRY = 3
            mock_settings.TIMEOUT = 30
            return DouyinCrawler()

    def test_already_downloaded_skips(self):
        """测试已下载视频被跳过"""
        crawler = self._create_crawler()
        crawler.video_dir.mkdir(parents=True, exist_ok=True)

        # 预先创建 vid1 的文件
        (crawler.video_dir / "vid1.mp4").touch()

        entries = [
            {"id": "vid1", "title": "已下载视频", "uploader": "作者1"},
            {"id": "vid2", "title": "新视频", "uploader": "作者2"},
        ]

        mock_download_result = {
            "success": True,
            "video_info": VideoInfo(video_id="vid2", title="新视频", author="作者2"),
            "error": None,
        }
        crawler._download_single_video = MagicMock(return_value=mock_download_result)

        with patch.object(crawler, "_build_yt_dlp_opts", return_value={"quiet": True}):
            with patch("yt_dlp.YoutubeDL") as mock_ydl:
                mock_ydl_instance = MagicMock()
                mock_ydl_instance.extract_info.return_value = {"entries": entries}
                mock_ydl.return_value.__enter__.return_value = mock_ydl_instance

                results = crawler.crawl_videos_impl("test_id", count=2)

                # vid1 应该被跳过（已存在），vid2 应该下载
                vids = [r.video_id for r in results]
                self.assertIn("vid1", vids)
                self.assertIn("vid2", vids)
                # vid1 的 file_path 应该存在
                vid1 = [r for r in results if r.video_id == "vid1"][0]
                self.assertNotEqual(vid1.file_path, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)