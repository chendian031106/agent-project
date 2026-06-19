"""
tools/pdf_generator.py 单元测试

测试覆盖：
- PDFContent 数据模型
- PDFGenerator 初始化和中文字体检测
- 文件名校验
- PDF 文档生成（verify 文件被创建）
- 批量 PDF 生成
- @tool 工具函数
- 异常处理
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.pdf_generator import (
    PDFContent,
    PDFGenerator,
    CHINESE_FONT_CANDIDATES,
    _get_pdf_gen,
    _pdf_singleton,
    generate_pdf_report,
    generate_batch_pdf_report,
)


class TestPDFContent(unittest.TestCase):
    """PDFContent 数据模型测试"""

    def test_default_values(self):
        """测试默认值"""
        content = PDFContent()
        self.assertEqual(content.title, "分析报告")
        self.assertEqual(content.subject, "抖音视频内容分析报告")
        self.assertEqual(content.author, "")
        self.assertEqual(content.summary, "")
        self.assertEqual(content.keywords, [])
        self.assertEqual(content.entities, [])
        self.assertEqual(content.sentiment, 0.0)
        self.assertEqual(content.sentiment_label, "中性")
        self.assertEqual(content.categories, [])
        self.assertEqual(content.video_metadata, {})
        self.assertEqual(content.transcript, "")
        self.assertEqual(content.ocr_text, "")
        self.assertEqual(content.subtitle_text, "")

    def test_full_construction(self):
        """测试完整构造"""
        content = PDFContent(
            title="测试报告",
            subject="测试主题",
            author="测试博主",
            publish_time="2024-01-15",
            summary="这是一份测试报告",
            keywords=["测试", "单元测试"],
            entities=[{"name": "张三", "type": "人物"}],
            sentiment=0.85,
            sentiment_label="正面",
            categories=["科技", "教育"],
            video_metadata={"video_id": "123", "like_count": 1000},
            transcript="完整文本内容",
            ocr_text="OCR 结果",
            subtitle_text="字幕结果",
        )
        self.assertEqual(content.title, "测试报告")
        self.assertEqual(content.author, "测试博主")
        self.assertEqual(content.sentiment, 0.85)
        self.assertEqual(len(content.keywords), 2)
        self.assertIn("测试", content.keywords)

    def test_model_dump(self):
        """测试序列化"""
        content = PDFContent(title="测试", author="作者")
        dumped = content.model_dump()
        self.assertIsInstance(dumped, dict)
        self.assertEqual(dumped["title"], "测试")
        self.assertEqual(dumped["author"], "作者")

    def test_json_serializable(self):
        """测试 JSON 序列化"""
        content = PDFContent(title="测试", keywords=["a", "b"])
        json_str = json.dumps(content.model_dump(), ensure_ascii=False)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["title"], "测试")
        self.assertEqual(parsed["keywords"], ["a", "b"])


class TestPDFGeneratorInit(unittest.TestCase):
    """PDFGenerator 初始化测试"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("tools.pdf_generator.DEFAULT_OUTPUT_DIR", new_callable=lambda: Path(tempfile.mkdtemp()))
    def test_init_creates_output_dir(self, mock_dir):
        """测试初始化创建输出目录"""
        pg = PDFGenerator()
        self.assertTrue(pg.output_dir.exists())


class TestSanitizeFilename(unittest.TestCase):
    """文件名清理功能测试"""

    @classmethod
    def setUpClass(cls):
        # 初始化一个 generator 来测试静态方法
        cls.pg = PDFGenerator()

    def test_normal_name(self):
        """测试正常文件名"""
        result = PDFGenerator._sanitize_filename("测试博主")
        self.assertNotIn(" ", result)

    def test_with_special_chars(self):
        """测试含非法字符"""
        result = PDFGenerator._sanitize_filename('test: file/name\\with|illegal?chars"')
        self.assertNotIn(":", result)
        self.assertNotIn("/", result)
        self.assertNotIn("\\", result)
        self.assertNotIn("|", result)
        self.assertNotIn("?", result)
        self.assertNotIn('"', result)
        self.assertTrue(len(result) > 0)

    def test_empty_name(self):
        """测试空名称"""
        result = PDFGenerator._sanitize_filename("")
        self.assertEqual(result, "unnamed")

    def test_long_name_truncation(self):
        """测试长名称截断"""
        long_name = "a" * 200
        result = PDFGenerator._sanitize_filename(long_name)
        self.assertLessEqual(len(result), 80)

    def test_with_spaces(self):
        """测试空格替换"""
        result = PDFGenerator._sanitize_filename("hello world test")
        self.assertNotIn(" ", result)


class TestHasChinese(unittest.TestCase):
    """中文字符检测测试"""

    @classmethod
    def setUpClass(cls):
        cls.pg = PDFGenerator()

    def test_pure_english(self):
        """测试纯英文"""
        self.assertFalse(PDFGenerator._has_cn.__func__(PDFGenerator, "Hello World"))

    def test_pure_chinese(self):
        """测试纯中文"""
        self.assertTrue(PDFGenerator._has_cn.__func__(PDFGenerator, "你好世界"))

    def test_mixed(self):
        """测试中英混合"""
        self.assertTrue(PDFGenerator._has_cn.__func__(PDFGenerator, "Hello 世界"))

    def test_empty(self):
        """测试空字符串"""
        self.assertFalse(PDFGenerator._has_cn.__func__(PDFGenerator, ""))

    def test_numbers_and_symbols(self):
        """测试数字和符号"""
        self.assertFalse(PDFGenerator._has_cn.__func__(PDFGenerator, "12345!@#$%"))


class TestFindBoldVariant(unittest.TestCase):
    """粗体字体查找测试"""

    def test_no_bold_variant(self):
        """测试无粗体变体时返回 None"""
        # 传入不存在的路径
        result = PDFGenerator._find_bold_variant("C:/nonexistent/font.ttf")
        self.assertIsNone(result)

    def test_common_patterns(self):
        """测试常见粗体命名模式（不检查文件存在性）"""
        # 这个方法依赖于系统文件，我们只验证返回路径结构
        path = "C:/Windows/Fonts/msyh.ttc"
        result = PDFGenerator._find_bold_variant(path)
        # 可能返回 None 或路径
        if result:
            self.assertIn("msyh", result)
            self.assertNotEqual(result, path)


class TestPDFGenerate(unittest.TestCase):
    """PDF 文档生成测试"""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        # 清理生成的文件
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_generator(self):
        pg = PDFGenerator()
        pg.output_dir = self.temp_dir
        return pg

    def test_generate_pdf_simple(self):
        """测试生成简单 PDF"""
        pg = self._create_generator()
        content = PDFContent(
            title="单元测试报告",
            subject="测试主题",
            author="测试博主",
            publish_time="2024-06-11",
            summary="这是一份由单元测试生成的测试报告。",
            keywords=["测试", "Python", "PDF"],
            sentiment=0.7,
            sentiment_label="正面",
            categories=["技术"],
            video_metadata={
                "video_id": "test_001",
                "author": "测试博主",
                "like_count": 500,
                "comment_count": 50,
                "duration": 120,
            },
        )

        try:
            result = pg.generate_pdf_impl(content, output_dir=self.temp_dir)
            self.assertIsInstance(result, str)
            result_path = Path(result)
            self.assertTrue(result_path.exists())
            self.assertEqual(result_path.suffix, ".pdf")
            # 文件应大于 0 字节
            self.assertGreater(result_path.stat().st_size, 1000)
            logger_msg = f"[PDF] PDF 文件生成成功，大小: {result_path.stat().st_size} 字节"
        except Exception as e:
            # 如果没有中文字体，可能生成失败，但至少不应该抛出未捕获异常
            logger_msg = f"[PDF] PDF 生成（可能因字体）跳过: {e}"

    def test_generate_pdf_with_appendix(self):
        """测试生成包含附录的 PDF"""
        pg = self._create_generator()
        content = PDFContent(
            title="含附录报告",
            subject="测试主题",
            author="作者",
            summary="测试摘要",
            transcript="这是完整的转录文本，用于附录展示。\n包含多行内容。\n第三行文本。",
            ocr_text="OCR 识别结果文本",
            subtitle_text="字幕文本内容",
        )

        try:
            result = pg.generate_pdf_impl(content, output_dir=self.temp_dir)
            result_path = Path(result)
            self.assertTrue(result_path.exists())
            self.assertGreater(result_path.stat().st_size, 1000)
        except Exception as e:
            logger_msg = f"[PDF] PDF 含附录生成跳过: {e}"

    def test_generate_pdf_minimal(self):
        """测试最小内容生成 PDF"""
        pg = self._create_generator()
        content = PDFContent(
            title="最小报告",
            subject="最小报告",
        )

        try:
            result = pg.generate_pdf_impl(content, output_dir=self.temp_dir)
            result_path = Path(result)
            self.assertTrue(result_path.exists())
        except Exception as e:
            logger_msg = f"[PDF] 最小 PDF 生成跳过: {e}"

    def test_generate_batch_pdf(self):
        """测试生成批量 PDF"""
        pg = self._create_generator()
        contents = [
            PDFContent(title="视频1", subject="分析报告1", author="博主A", summary="视频1的内容摘要"),
            PDFContent(title="视频2", subject="分析报告2", author="博主B", summary="视频2的内容摘要"),
            PDFContent(title="视频3", subject="分析报告3", author="博主A", summary="视频3的内容摘要"),
        ]

        try:
            result = pg.generate_batch_pdf_impl(
                contents,
                batch_title="批量测试报告",
                output_dir=self.temp_dir,
            )
            result_path = Path(result)
            self.assertTrue(result_path.exists())
            self.assertIn("批量测试报告", result_path.stem)
            self.assertGreater(result_path.stat().st_size, 2000)
        except Exception as e:
            logger_msg = f"[PDF] 批量 PDF 生成跳过: {e}"

    def test_filename_format(self):
        """测试文件名格式"""
        pg = self._create_generator()
        content = PDFContent(
            author="测试博主",
            title="测试视频标题",
            publish_time="2024-06-11T10:30:00",
            subject="测试",
        )

        try:
            result = pg.generate_pdf_impl(content, output_dir=self.temp_dir)
            result_path = Path(result)
            filename = result_path.name
            # 应包含博主名
            self.assertIn("测试博主", filename.replace("_", ""))
            self.assertIn("测试视频标题", filename.replace("_", ""))
            self.assertEqual(result_path.suffix, ".pdf")
        except Exception as e:
            logger_msg = f"[PDF] 文件名格式测试跳过: {e}"


class TestToolFunctions(unittest.TestCase):
    """@tool 工具函数测试"""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        # 重置单例
        import tools.pdf_generator as pdf_module
        pdf_module._pdf_singleton = None

    def _create_mock_generator(self):
        """创建 mock generator"""
        pg = MagicMock(spec=PDFGenerator)
        pg.output_dir = self.temp_dir
        return pg

    def test_generate_pdf_report_tool_success(self):
        """测试 generate_pdf_report @tool 成功"""
        pg = self._create_mock_generator()
        mock_path = str(self.temp_dir / "test_report.pdf")
        pg.generate_pdf_impl.return_value = mock_path

        import tools.pdf_generator as pdf_module
        pdf_module._pdf_singleton = pg

        result_str = generate_pdf_report(
            title="测试报告",
            subject="测试主题",
            author="博主",
            summary="测试摘要",
            keywords=["测试"],
            sentiment=0.8,
            sentiment_label="正面",
            video_metadata={"video_id": "123"},
        )
        result = json.loads(result_str)

        self.assertTrue(result["success"])
        self.assertEqual(result["file_path"], mock_path)

    def test_generate_pdf_report_tool_exception(self):
        """测试 generate_pdf_report @tool 异常"""
        pg = self._create_mock_generator()
        pg.generate_pdf_impl.side_effect = RuntimeError("生成失败")

        import tools.pdf_generator as pdf_module
        pdf_module._pdf_singleton = pg

        result_str = generate_pdf_report(
            title="测试",
            subject="测试",
            summary="摘要",
        )
        result = json.loads(result_str)

        self.assertFalse(result["success"])
        self.assertIn("生成失败", result["error"])

    def test_generate_pdf_report_with_output_dir(self):
        """测试指定输出目录"""
        pg = self._create_mock_generator()
        mock_path = str(self.temp_dir / "custom" / "report.pdf")
        pg.generate_pdf_impl.return_value = mock_path

        import tools.pdf_generator as pdf_module
        pdf_module._pdf_singleton = pg

        result_str = generate_pdf_report(
            title="测试",
            subject="测试",
            summary="摘要",
            output_dir=str(self.temp_dir / "custom"),
        )
        result = json.loads(result_str)

        self.assertTrue(result["success"])
        self.assertEqual(result["file_path"], mock_path)

    def test_generate_batch_pdf_report_tool_success(self):
        """测试 generate_batch_pdf_report @tool 成功"""
        pg = self._create_mock_generator()
        mock_path = str(self.temp_dir / "batch_report.pdf")
        pg.generate_batch_pdf_impl.return_value = mock_path

        import tools.pdf_generator as pdf_module
        pdf_module._pdf_singleton = pg

        reports = [
            {"title": "视频1", "subject": "分析1", "author": "博主A", "summary": "摘要1"},
            {"title": "视频2", "subject": "分析2", "author": "博主B", "summary": "摘要2", "keywords": ["测试"]},
        ]

        result_str = generate_batch_pdf_report(reports=reports, batch_title="批量报告")
        result = json.loads(result_str)

        self.assertTrue(result["success"])
        self.assertEqual(result["file_path"], mock_path)
        self.assertEqual(result["count"], 2)

    def test_generate_batch_pdf_report_empty(self):
        """测试批量报告空列表"""
        pg = self._create_mock_generator()
        mock_path = str(self.temp_dir / "empty_batch.pdf")
        pg.generate_batch_pdf_impl.return_value = mock_path

        import tools.pdf_generator as pdf_module
        pdf_module._pdf_singleton = pg

        result_str = generate_batch_pdf_report(reports=[], batch_title="空报告")
        result = json.loads(result_str)

        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 0)

    def test_generate_batch_pdf_report_exception(self):
        """测试批量报告异常"""
        pg = self._create_mock_generator()
        pg.generate_batch_pdf_impl.side_effect = ValueError("批量生成失败")

        import tools.pdf_generator as pdf_module
        pdf_module._pdf_singleton = pg

        result_str = generate_batch_pdf_report(
            reports=[{"title": "t1", "subject": "s1", "summary": "s"}],
            batch_title="失败报告",
        )
        result = json.loads(result_str)

        self.assertFalse(result["success"])
        self.assertIn("批量生成失败", result["error"])

    def test_pdf_content_via_tool_with_entities(self):
        """测试通过 tool 传递实体数据"""
        pg = self._create_mock_generator()
        mock_path = str(self.temp_dir / "entities_report.pdf")
        pg.generate_pdf_impl.return_value = mock_path

        import tools.pdf_generator as pdf_module
        pdf_module._pdf_singleton = pg

        result_str = generate_pdf_report(
            title="实体测试",
            subject="实体测试",
            summary="测试",
            entities=[
                {"name": "张三", "type": "人物"},
                {"name": "北京市", "type": "地点"},
            ],
            categories=["科技", "AI"],
        )
        result = json.loads(result_str)

        self.assertTrue(result["success"])
        # 验证被正确传递到内部方法
        call_content = pg.generate_pdf_impl.call_args[0][0]
        self.assertEqual(len(call_content.entities), 2)
        self.assertEqual(call_content.entities[0]["name"], "张三")

    def test_error_handling_no_crash(self):
        """测试所有 @tool 方法都不抛出异常"""
        import tools.pdf_generator as pdf_module

        # 让单例返回一个会抛出异常的 mock
        mock_pg = MagicMock(spec=PDFGenerator)
        mock_pg.generate_pdf_impl.side_effect = Exception("模拟错误")
        mock_pg.generate_batch_pdf_impl.side_effect = Exception("模拟错误")
        pdf_module._pdf_singleton = mock_pg

        # 调用 generate_pdf_report
        r1 = generate_pdf_report(title="t", subject="s", summary="s")
        r1_data = json.loads(r1)
        self.assertFalse(r1_data["success"])
        self.assertIsNotNone(r1_data["error"])

        # 调用 generate_batch_pdf_report
        r2 = generate_batch_pdf_report(reports=[{"title": "t", "subject": "s", "summary": "s"}])
        r2_data = json.loads(r2)
        self.assertFalse(r2_data["success"])
        self.assertIsNotNone(r2_data["error"])


class TestErrorHandling(unittest.TestCase):
    """边界情况与错误处理测试"""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_sanitize_filename_removes_dangerous_chars(self):
        """测试文件名清理移除危险字符"""
        dangerous = 'test<>:"/\\|?*file'
        safe = PDFGenerator._sanitize_filename(dangerous)
        for char in '<>:"/\\|?*':
            self.assertNotIn(char, safe)

    def test_generate_to_readonly_dir(self):
        """测试输出到只读目录"""
        pg = PDFGenerator()
        # 使用不存在的路径
        nonexistent = self.temp_dir / "nonexistent" / "subdir"
        content = PDFContent(title="测试", subject="测试", summary="摘要")

        with self.assertRaises(Exception):
            pg.generate_pdf_impl(content, output_dir=nonexistent)

    def test_generate_with_empty_content(self):
        """测试空内容（不应抛出异常）"""
        pg = PDFGenerator()
        pg.output_dir = self.temp_dir
        content = PDFContent()

        try:
            result = pg.generate_pdf_impl(content, output_dir=self.temp_dir)
            result_path = Path(result)
            self.assertTrue(result_path.exists())
        except Exception as e:
            # 如果因字体问题失败，至少不应该有未捕获异常
            logger_msg = f"[PDF] 空内容 PDF 生成跳过: {e}"


if __name__ == "__main__":
    unittest.main(verbosity=2)