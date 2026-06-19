"""
PDF 文档自动生成工具

基于 reportlab 生成专业的分析报告 PDF。
支持封面、正文、附录的完整文档模板，自动处理中文排版和页码。

所有对外暴露的方法均使用 @tool 注解，供智能体调用。
"""

import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.tools import tool
from pydantic import BaseModel, Field
from reportlab.lib import pagesizes
from reportlab.lib.colors import HexColor, black, grey, white
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    Frame,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    BaseDocTemplate,
)

from utils.config import settings
from utils.logger import logger

# ============ 常量 ============

DEFAULT_OUTPUT_DIR = Path("./data/output")

# ReportLab 预定义颜色
COLOR_PRIMARY = HexColor("#1a1a2e")       # 深蓝黑
COLOR_SECONDARY = HexColor("#16213e")     # 藏蓝
COLOR_ACCENT = HexColor("#0f3460")        # 普鲁士蓝
COLOR_HIGHLIGHT = HexColor("#e94560")     # 珊瑚红
COLOR_LIGHT_BG = HexColor("#f0f2f5")      # 浅灰背景
COLOR_BORDER = HexColor("#d0d5dd")        # 边框灰
COLOR_TEXT_DARK = HexColor("#1d2939")     # 正文深色
COLOR_TEXT_MEDIUM = HexColor("#475467")   # 辅助文字
COLOR_TEXT_LIGHT = HexColor("#98a2b3")    # 浅色文字

# 常见中文字体路径（按优先级）
CHINESE_FONT_CANDIDATES: List[str] = [
    # Windows
    "C:/Windows/Fonts/msyh.ttc",           # Microsoft YaHei
    "C:/Windows/Fonts/msyhbd.ttc",         # Microsoft YaHei Bold
    "C:/Windows/Fonts/simsun.ttc",         # SimSun
    "C:/Windows/Fonts/simhei.ttf",         # SimHei
    "C:/Windows/Fonts/STSONG.TTF",         # STSong
    "C:/Windows/Fonts/fangsong.ttf",       # FangSong
    "C:/Windows/Fonts/kaiti.ttf",          # KaiTi
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/STSong.ttf",
    # Linux
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
]

# ============ Pydantic 数据模型 ============


class PDFContent(BaseModel):
    """PDF 文档内容模型"""

    title: str = Field(default="分析报告", description="文档标题")
    subject: str = Field(default="抖音视频内容分析报告", description="文档主题")
    author: str = Field(default="", description="博主名称")
    publish_time: str = Field(default="", description="发布时间")
    summary: str = Field(default="", description="内容摘要")
    keywords: List[str] = Field(default_factory=list, description="关键词列表")
    entities: List[Dict[str, str]] = Field(default_factory=list, description="命名实体")
    sentiment: float = Field(default=0.0, description="情感分值")
    sentiment_label: str = Field(default="中性", description="情感标签")
    categories: List[str] = Field(default_factory=list, description="分类标签")
    video_metadata: Dict[str, Any] = Field(default_factory=dict, description="视频元数据")
    transcript: str = Field(default="", description="完整文本内容（附录）")
    ocr_text: str = Field(default="", description="OCR 文本（附录）")
    subtitle_text: str = Field(default="", description="字幕文本（附录）")


class PDFGenerateInput(BaseModel):
    """generate_pdf_report 参数"""

    title: str = Field(default="分析报告", description="文档标题")
    subject: str = Field(default="抖音视频内容分析报告", description="文档主题")
    author: str = Field(default="", description="博主名称")
    summary: str = Field(default="", description="内容摘要")
    keywords: List[str] = Field(default_factory=list, description="关键词")
    entities: List[Dict[str, str]] = Field(default_factory=list, description="命名实体")
    sentiment: float = Field(default=0.0, description="情感分值 0-1")
    sentiment_label: str = Field(default="中性", description="情感标签")
    categories: List[str] = Field(default_factory=list, description="分类")
    video_metadata: Dict[str, Any] = Field(default_factory=dict, description="视频元数据")
    transcript: str = Field(default="", description="完整转录文本（可选）")
    ocr_text: str = Field(default="", description="OCR 文本（可选）")
    subtitle_text: str = Field(default="", description="字幕文本（可选）")
    output_dir: str = Field(default="", description="输出目录（默认 ./data/output）")


# ============ PDF 模板（基于 BaseDocTemplate 的多页模板） ============


class ReportDocTemplate(BaseDocTemplate):
    """自定义文档模板，支持封面/正文/附录不同页面布局"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.cover_frame: Optional[Frame] = None
        self.body_frame: Optional[Frame] = None
        super().__init__(*args, **kwargs)

    def _setup_pages(self) -> None:
        """设置页面模板"""
        margin = 2 * cm
        page_w, page_h = pagesizes.A4

        # 封面：居中布局
        cover_frame = Frame(
            margin, margin, page_w - 2 * margin, page_h - 2 * margin,
            id="cover_frame", showBoundary=0,
        )
        cover_template = PageTemplate(
            id="cover",
            frames=[cover_frame],
            onPage=self._cover_page_decorator,
            pagesize=pagesizes.A4,
        )

        # 正文：带页眉页脚
        body_frame = Frame(
            margin, 2.5 * cm, page_w - 2 * margin, page_h - 4.5 * cm,
            id="body_frame", showBoundary=0,
        )
        body_template = PageTemplate(
            id="body",
            frames=[body_frame],
            onPage=self._body_page_decorator,
            pagesize=pagesizes.A4,
        )

        self.addPageTemplates([cover_template, body_template])

    @staticmethod
    def _cover_page_decorator(canvas: Any, doc: Any) -> None:
        """封面页装饰"""
        canvas.saveState()
        page_w, page_h = pagesizes.A4

        # 顶部色块
        canvas.setFillColor(COLOR_PRIMARY)
        canvas.rect(0, page_h - 6 * cm, page_w, 6 * cm, fill=1, stroke=0)

        # 底部色块
        canvas.setFillColor(COLOR_SECONDARY)
        canvas.rect(0, 0, page_w, 3 * cm, fill=1, stroke=0)

        canvas.restoreState()

    @staticmethod
    def _body_page_decorator(canvas: Any, doc: Any) -> None:
        """正文页装饰（页眉页脚 + 页码）"""
        canvas.saveState()
        page_w, page_h = pagesizes.A4

        # 页眉线
        canvas.setStrokeColor(COLOR_ACCENT)
        canvas.setLineWidth(0.5)
        canvas.line(2 * cm, page_h - 2 * cm, page_w - 2 * cm, page_h - 2 * cm)

        # 页眉文字
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(COLOR_TEXT_LIGHT)
        canvas.drawString(2 * cm, page_h - 1.5 * cm, "DouyinCrew - AI 内容分析报告")

        # 页码（底部居中）
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(COLOR_TEXT_MEDIUM)
        page_num = f"- {doc.page} -"
        canvas.drawCentredString(page_w / 2, 1.2 * cm, page_num)

        # 页脚线
        canvas.setStrokeColor(COLOR_BORDER)
        canvas.setLineWidth(0.3)
        canvas.line(2 * cm, 1.8 * cm, page_w - 2 * cm, 1.8 * cm)

        canvas.restoreState()


# ============ PDFGenerator 主类 ============


class PDFGenerator:
    """PDF 文档生成器

    使用 reportlab 生成专业的分析报告 PDF。
    自动检测系统中文可用字体，解决中文乱码。
    支持封面、正文、附录的完整文档模板。
    """

    def __init__(self) -> None:
        self.output_dir = DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 中文支持
        self._font_name = "Helvetica"  # 默认西文
        self._font_name_bold = "Helvetica-Bold"
        self._cn_font_name: Optional[str] = None
        self._cn_font_name_bold: Optional[str] = None
        self._setup_chinese_font()

        # 初始化样式
        self.styles = self._init_styles()

        logger.info(
            f"PDFGenerator 初始化完成 | "
            f"输出目录: {self.output_dir} | "
            f"中文字体: {self._cn_font_name or '未找到'}"
        )

    # ---------- 字体管理 ----------

    def _setup_chinese_font(self) -> None:
        """检测并注册中文字体"""
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        for font_path in CHINESE_FONT_CANDIDATES:
            path = Path(font_path)
            if path.exists():
                try:
                    font_name = f"CN_{path.stem}"
                    pdfmetrics.registerFont(TTFont(font_name, str(path)))

                    # 尝试注册粗体
                    bold_path = self._find_bold_variant(font_path)
                    bold_name = f"CN_{path.stem}Bold"
                    if bold_path:
                        pdfmetrics.registerFont(TTFont(bold_name, str(bold_path)))
                        self._cn_font_name_bold = bold_name
                    else:
                        self._cn_font_name_bold = font_name

                    self._cn_font_name = font_name
                    logger.debug(f"已注册中文字体: {font_name} ({font_path})")
                    return

                except Exception as e:
                    logger.warning(f"注册字体失败 {font_path}: {e}")
                    continue

        logger.warning("未找到系统中文字体，PDF 中文可能无法正常显示")

    @staticmethod
    def _find_bold_variant(font_path: str) -> Optional[str]:
        """查找对应粗体字体"""
        base = Path(font_path)
        candidates = [
            base.parent / f"{base.stem}bd{base.suffix}",
            base.parent / f"{base.stem}Bold{base.suffix}",
            base.parent / f"{base.stem}-Bold{base.suffix}",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return None

    def _resolve_font(self, cn: bool = True) -> str:
        """根据是否有中文字体返回合适的字体名"""
        if cn and self._cn_font_name:
            return self._cn_font_name
        return self._font_name

    def _resolve_font_bold(self, cn: bool = True) -> str:
        """返回粗体字体名"""
        if cn and self._cn_font_name_bold:
            return self._cn_font_name_bold
        return self._font_name_bold

    def _has_cn(self, text: str) -> bool:
        """检测文本是否包含中文字符"""
        return bool(re.search(r"[\u4e00-\u9fff]", text))

    def _pick_font(self, bold: bool = False) -> str:
        """智能选择字体：有中文时用中文字体"""
        return self._resolve_font_bold() if bold else self._resolve_font()

    # ---------- 样式初始化 ----------

    def _init_styles(self) -> Dict[str, ParagraphStyle]:
        """初始化段落样式"""
        base_style = ParagraphStyle(
            "base_style",
            fontName=self._pick_font(),
            fontSize=10,
            leading=16,
            textColor=COLOR_TEXT_DARK,
            spaceAfter=6,
        )

        return {
            "cover_title": ParagraphStyle(
                "cover_title",
                fontName=self._pick_font(bold=True),
                fontSize=28,
                leading=38,
                textColor=white,
                alignment=TA_CENTER,
                spaceAfter=12,
            ),
            "cover_subtitle": ParagraphStyle(
                "cover_subtitle",
                fontName=self._pick_font(),
                fontSize=14,
                leading=22,
                textColor=HexColor("#cbd5e1"),
                alignment=TA_CENTER,
                spaceAfter=40,
            ),
            "cover_info": ParagraphStyle(
                "cover_info",
                fontName=self._pick_font(),
                fontSize=11,
                leading=18,
                textColor=COLOR_TEXT_MEDIUM,
                alignment=TA_CENTER,
                spaceAfter=4,
            ),
            "h1": ParagraphStyle(
                "h1",
                fontName=self._pick_font(bold=True),
                fontSize=18,
                leading=26,
                textColor=COLOR_PRIMARY,
                spaceBefore=20,
                spaceAfter=12,
            ),
            "h2": ParagraphStyle(
                "h2",
                fontName=self._pick_font(bold=True),
                fontSize=14,
                leading=20,
                textColor=COLOR_ACCENT,
                spaceBefore=16,
                spaceAfter=8,
            ),
            "h3": ParagraphStyle(
                "h3",
                fontName=self._pick_font(bold=True),
                fontSize=12,
                leading=18,
                textColor=COLOR_TEXT_DARK,
                spaceBefore=12,
                spaceAfter=6,
            ),
            "body": ParagraphStyle(
                "body",
                fontName=self._pick_font(),
                fontSize=10,
                leading=18,
                textColor=COLOR_TEXT_DARK,
                spaceAfter=6,
                firstLineIndent=20,
                alignment=TA_LEFT,
            ),
            "body_indent": ParagraphStyle(
                "body_indent",
                fontName=self._pick_font(),
                fontSize=10,
                leading=18,
                textColor=COLOR_TEXT_DARK,
                spaceAfter=4,
                leftIndent=20,
            ),
            "caption": ParagraphStyle(
                "caption",
                fontName=self._pick_font(),
                fontSize=9,
                leading=14,
                textColor=COLOR_TEXT_LIGHT,
                spaceAfter=10,
                alignment=TA_CENTER,
            ),
            "code": ParagraphStyle(
                "code",
                fontName="Courier",
                fontSize=8,
                leading=12,
                textColor=COLOR_TEXT_DARK,
                spaceAfter=4,
                leftIndent=10,
                backColor=COLOR_LIGHT_BG,
            ),
            "footer": ParagraphStyle(
                "footer",
                fontName=self._pick_font(),
                fontSize=8,
                leading=12,
                textColor=COLOR_TEXT_LIGHT,
                alignment=TA_CENTER,
            ),
        }

    # ---------- 文档构建 ----------

    def _build_cover_story(self, content: PDFContent) -> List[Any]:
        """构建封面内容"""
        story: List[Any] = []

        # 空行将标题推到中部
        story.append(Spacer(1, 8 * cm))

        # 标题
        story.append(Paragraph(content.subject, self.styles["cover_title"]))

        # 副标题 / 具体标题
        if content.title and content.title != content.subject:
            story.append(Paragraph(content.title, self.styles["cover_subtitle"]))

        story.append(Spacer(1, 3 * cm))

        # 元信息
        if content.author:
            story.append(Paragraph(f"博主：{content.author}", self.styles["cover_info"]))
        if content.publish_time:
            story.append(Paragraph(f"发布时间：{content.publish_time}", self.styles["cover_info"]))
        story.append(Paragraph(f"生成日期：{datetime.now().strftime('%Y-%m-%d %H:%M')}", self.styles["cover_info"]))

        if content.keywords:
            kw_text = " | ".join(content.keywords[:8])
            story.append(Spacer(1, 1.5 * cm))
            story.append(Paragraph(f"关键词：{kw_text}", self.styles["cover_info"]))

        return story

    def _build_body_story(self, content: PDFContent) -> List[Any]:
        """构建正文内容"""
        story: List[Any] = []

        # ---------- 第一章：概览 ----------
        story.append(Paragraph("一、内容概览", self.styles["h1"]))

        # 关键词标签
        if content.keywords:
            story.append(Paragraph("关键词", self.styles["h2"]))
            kw_cells = []
            row = []
            for i, kw in enumerate(content.keywords[:12]):
                row.append(kw)
                if len(row) >= 4:
                    kw_cells.append(row)
                    row = []
            if row:
                kw_cells.append(row)

            if kw_cells:
                kw_table = Table(kw_cells, colWidths=[4 * cm] * len(kw_cells[0]))
                kw_table.setStyle(TableStyle([
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("FONTNAME", (0, 0), (-1, -1), self._pick_font()),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("TEXTCOLOR", (0, 0), (-1, -1), COLOR_ACCENT),
                    ("BACKGROUND", (0, 0), (-1, -1), COLOR_LIGHT_BG),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("GRID", (0, 0), (-1, -1), 0.3, COLOR_BORDER),
                ]))
                story.append(kw_table)
                story.append(Spacer(1, 0.3 * cm))

        # 分类
        if content.categories:
            story.append(Paragraph("分类标签", self.styles["h2"]))
            cat_text = "  |  ".join(content.categories)
            story.append(Paragraph(cat_text, self.styles["body_indent"]))

        # ---------- 第二章：摘要 ----------
        story.append(Paragraph("二、内容摘要", self.styles["h1"]))
        if content.summary:
            story.append(Paragraph(content.summary, self.styles["body"]))
        else:
            story.append(Paragraph("（无摘要内容）", self.styles["body"]))

        # ---------- 第三章：情感分析 ----------
        story.append(Paragraph("三、情感分析", self.styles["h1"]))

        # 情感指示条
        sentiment_value = max(0.0, min(1.0, content.sentiment))
        story.extend(self._build_sentiment_chart(sentiment_value, content.sentiment_label))

        # 实体
        if content.entities:
            story.append(Paragraph("提取实体", self.styles["h2"]))
            entity_data = [["实体", "类型"]]
            for ent in content.entities[:15]:
                name = ent.get("name", ent.get("entity", ""))
                etype = ent.get("type", "")
                entity_data.append([str(name), str(etype)])

            if len(entity_data) > 1:
                ent_table = Table(entity_data, colWidths=[8 * cm, 6 * cm])
                ent_table.setStyle(TableStyle([
                    ("FONTNAME", (0, 0), (-1, 0), self._pick_font(bold=True)),
                    ("FONTNAME", (0, 1), (-1, -1), self._pick_font()),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("BACKGROUND", (0, 0), (-1, 0), COLOR_ACCENT),
                    ("TEXTCOLOR", (0, 0), (-1, 0), white),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("GRID", (0, 0), (-1, -1), 0.3, COLOR_BORDER),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]))
                story.append(ent_table)

        # ---------- 第四章：视频元数据 ----------
        if content.video_metadata:
            story.append(Paragraph("四、视频元数据", self.styles["h1"]))
            story.extend(self._build_metadata_table(content.video_metadata))

        # ---------- 第五章：附录 ----------
        appendix_texts = []
        if content.transcript:
            appendix_texts.append(("完整转录文本", content.transcript))
        if content.subtitle_text:
            appendix_texts.append(("字幕文本", content.subtitle_text))
        if content.ocr_text:
            appendix_texts.append(("OCR 文本", content.ocr_text))

        if appendix_texts:
            story.append(NextPageTemplate("body"))
            story.append(PageBreak())
            story.append(Paragraph("五、附录", self.styles["h1"]))

            for section_title, section_content in appendix_texts:
                story.append(Paragraph(section_title, self.styles["h2"]))
                # 截断过长内容（最多3000字）
                display_text = section_content[:3000]
                if len(section_content) > 3000:
                    display_text += "\n\n...（内容过长，已截断）"
                story.append(Paragraph(display_text.replace("\n", "<br/>"), self.styles["body"]))

        return story

    @staticmethod
    def _build_sentiment_chart(value: float, label: str) -> List[Any]:
        """构建情感指示条"""
        from reportlab.platypus import Table, TableStyle
        from reportlab.lib.colors import HexColor

        story: List[Any] = []
        bar_width = 14 * cm
        bar_height = 0.8 * cm

        # 颜色渐变：红（负面）- 黄（中性）- 绿（正面）
        if value < 0.33:
            bar_color = HexColor("#f04438")
            desc = "负面"
        elif value < 0.66:
            bar_color = HexColor("#f79009")
            desc = "中性"
        else:
            bar_color = HexColor("#12b76a")
            desc = "正面"

        filled_width = value * bar_width

        # 简单条状图使用颜色块
        filled_part = [[Paragraph(
            f"<font color='white' size={10}>{desc} ({label})</font>",
            ParagraphStyle("sentiment_text", fontName="Helvetica", fontSize=10, alignment=TA_CENTER)
        )]]

        fill_table = Table(filled_part, colWidths=[filled_width])
        fill_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), bar_color),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))

        empty_part = [[Paragraph(
            f"<font color='#98a2b3' size=8>情感强度: {value:.2f}</font>",
            ParagraphStyle("sentiment_label", fontName="Helvetica", fontSize=8, alignment=TA_CENTER)
        )]]

        empty_table = Table(empty_part, colWidths=[bar_width - filled_width])
        empty_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), COLOR_LIGHT_BG),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))

        bar_table = Table(
            [[fill_table, empty_table]],
            colWidths=[filled_width, bar_width - filled_width],
        )
        bar_table.setStyle(TableStyle([
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))

        story.append(bar_table)
        return story

    @staticmethod
    def _build_metadata_table(metadata: Dict[str, Any]) -> List[Any]:
        """构建视频元数据表格"""
        from reportlab.platypus import Table, TableStyle
        from reportlab.lib.colors import HexColor

        # 映射显示名称
        key_labels = {
            "video_id": "视频ID",
            "title": "视频标题",
            "author": "博主名称",
            "author_id": "博主抖音号",
            "duration": "视频时长（秒）",
            "like_count": "点赞数",
            "comment_count": "评论数",
            "collect_count": "收藏数",
            "share_count": "分享数",
            "publish_time": "发布时间",
            "view_count": "播放量",
            "file_path": "本地文件路径",
        }

        rows = [["字段", "值"]]
        for key, label in key_labels.items():
            if key in metadata and metadata[key]:
                value = str(metadata[key])
                if len(value) > 60:
                    value = value[:57] + "..."
                rows.append([label, value])

        if len(rows) <= 1:
            # 无已知字段，直接展示所有
            rows = [["字段", "值"]]
            for k, v in metadata.items():
                rows.append([str(k), str(v)[:60]])

        col_widths = [4.5 * cm, 10 * cm]
        md_table = Table(rows, colWidths=col_widths, repeatRows=1)
        md_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (-1, 0), COLOR_ACCENT),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("ALIGN", (0, 1), (0, -1), "LEFT"),
            ("ALIGN", (1, 1), (1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.3, COLOR_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, COLOR_LIGHT_BG]),
        ]))
        return [md_table]

    # ========== 对外暴露的方法（内部实现） ==========

    def generate_pdf_impl(
        self,
        content: PDFContent,
        output_dir: Optional[Path] = None,
    ) -> str:
        """生成单个 PDF 报告

        Args:
            content: PDF 内容
            output_dir: 输出目录，默认 ./data/output

        Returns:
            生成的 PDF 文件绝对路径
        """
        output_dir = output_dir or self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # 文件名：{博主名}-{视频标题}-{发布时间}.pdf
        safe_author = self._sanitize_filename(content.author or "未知博主")
        safe_title = self._sanitize_filename(content.title or "分析报告")
        safe_time = self._sanitize_filename(content.publish_time or datetime.now().strftime("%Y%m%d"))
        filename = f"{safe_author}-{safe_title}-{safe_time}.pdf"
        filepath = output_dir / filename

        logger.info(f"[PDF] 开始生成: {filepath}")

        try:
            # 使用自定义模板
            doc = ReportDocTemplate(
                str(filepath),
                pagesize=pagesizes.A4,
                title=content.subject,
                author=content.author or "DouyinCrew",
                subject=content.subject,
                showBoundary=0,
                allowSplitting=1,
            )
            doc._setup_pages()

            # 构建内容
            story: List[Any] = []
            story.extend(self._build_cover_story(content))
            story.append(NextPageTemplate("body"))
            story.append(PageBreak())
            story.extend(self._build_body_story(content))

            doc.build(story)
            logger.info(f"[PDF] 生成成功: {filepath}")
            return str(filepath.resolve())

        except Exception as e:
            logger.error(f"[PDF] 生成失败: {type(e).__name__}: {e}")
            raise

    def generate_batch_pdf_impl(
        self,
        contents: List[PDFContent],
        batch_title: str = "批量分析报告",
        output_dir: Optional[Path] = None,
    ) -> str:
        """生成批量 PDF 报告

        将多个视频的分析报告合并为一个 PDF。

        Args:
            contents: PDF 内容列表
            batch_title: 批量报告总标题
            output_dir: 输出目录

        Returns:
            生成的 PDF 文件绝对路径
        """
        output_dir = output_dir or self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        safe_title = self._sanitize_filename(batch_title)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_title}-{timestamp}.pdf"
        filepath = output_dir / filename

        logger.info(f"[PDF] 开始生成批量报告: {filepath} ({len(contents)} 个视频)")

        try:
            doc = ReportDocTemplate(
                str(filepath),
                pagesize=pagesizes.A4,
                title=batch_title,
                author="DouyinCrew",
                subject=batch_title,
                showBoundary=0,
                allowSplitting=1,
            )
            doc._setup_pages()

            story: List[Any] = []
            # 总封面
            combined = PDFContent(
                title=batch_title,
                subject=batch_title,
                author="",
                publish_time="",
                summary=f"共包含 {len(contents)} 个视频的分析报告",
            )
            story.extend(self._build_cover_story(combined))

            for idx, content in enumerate(contents):
                story.append(NextPageTemplate("body"))
                story.append(PageBreak())

                # 每个视频作为一个独立章节
                story.append(Paragraph(f"视频 {idx + 1}: {content.title or '无标题'}", self.styles["h1"]))
                story.extend(self._build_body_story(content))

            doc.build(story)
            logger.info(f"[PDF] 批量报告生成成功: {filepath}")
            return str(filepath.resolve())

        except Exception as e:
            logger.error(f"[PDF] 批量报告生成失败: {type(e).__name__}: {e}")
            raise

    def generate_pdf(self, video_data: dict, analysis_data: dict) -> str:
        """生成单个 PDF 报告（路由层接口）

        将 video_data 和 analysis_data 两个 dict 转换为 PDFContent 后调用 _impl。
        """
        content = PDFContent(
            title=video_data.get("title", "分析报告"),
            subject=f"抖音视频分析报告 - {video_data.get('title', '')}",
            author=video_data.get("author", ""),
            publish_time=video_data.get("publish_time", ""),
            summary=analysis_data.get("summary", ""),
            keywords=analysis_data.get("keywords", []),
            entities=analysis_data.get("entities", []),
            sentiment=analysis_data.get("sentiment", 0.5),
            categories=analysis_data.get("categories", []),
            video_metadata=video_data,
        )
        return self.generate_pdf_impl(content)

    def generate_batch_pdf(self, video_data_list: list, analysis_results: list) -> str:
        """生成批量 PDF 报告（路由层接口）"""
        contents = []
        for vd, ar in zip(video_data_list, analysis_results):
            contents.append(PDFContent(
                title=vd.get("title", "分析报告"),
                subject=f"抖音视频分析报告 - {vd.get('title', '')}",
                author=vd.get("author", ""),
                summary=ar.get("summary", ""),
                keywords=ar.get("keywords", []),
                entities=ar.get("entities", []),
                sentiment=ar.get("sentiment", 0.5),
                categories=ar.get("categories", []),
                video_metadata=vd,
            ))
        batch_title = f"批量分析报告_{datetime.now().strftime('%Y%m%d')}"
        return self.generate_batch_pdf_impl(contents, batch_title=batch_title)

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """清理文件名中非法字符"""
        if not name:
            return "unnamed"
        # 替换文件系统非法字符
        sanitized = re.sub(r'[<>:"/\\|?*]', "_", name)
        sanitized = re.sub(r"\s+", "_", sanitized)
        # 限制长度
        if len(sanitized) > 80:
            sanitized = sanitized[:80]
        return sanitized.strip("._ ")


# ============ LangChain @tool 工具函数 ============

_pdf_singleton: Optional[PDFGenerator] = None


def _get_pdf_gen() -> PDFGenerator:
    """获取 PDFGenerator 单例"""
    global _pdf_singleton
    if _pdf_singleton is None:
        _pdf_singleton = PDFGenerator()
    return _pdf_singleton


@tool
def generate_pdf_report(
    title: str = "分析报告",
    subject: str = "抖音视频内容分析报告",
    author: str = "",
    summary: str = "",
    keywords: List[str] = None,
    entities: List[Dict[str, str]] = None,
    sentiment: float = 0.5,
    sentiment_label: str = "中性",
    categories: List[str] = None,
    video_metadata: Dict[str, Any] = None,
    transcript: str = "",
    ocr_text: str = "",
    subtitle_text: str = "",
    output_dir: str = "",
) -> str:
    """生成抖音视频分析报告的 PDF 文档。

    生成包含封面、内容概览、摘要、情感分析、视频元数据和附录的专业报告。

    Args:
        title: 报告标题
        subject: 报告主题（封面大字）
        author: 博主名称
        summary: 内容摘要文本
        keywords: 关键词列表，如 ["美食", "教程"]
        entities: 命名实体列表，如 [{"name": "张三", "type": "人物"}]
        sentiment: 情感分值 0.0~1.0，默认0.5
        sentiment_label: 情感标签，如 "正面"、"负面"、"中性"
        categories: 分类标签列表
        video_metadata: 视频元数据字典
        transcript: 完整转录文本（可选，会放入附录）
        ocr_text: OCR 识别文本（可选，会放入附录）
        subtitle_text: 字幕文本（可选，会放入附录）
        output_dir: 输出目录路径（可选，默认 ./data/output）

    Returns:
        JSON字符串，格式: {"success": bool, "file_path": str, "error": str|null}
    """
    try:
        if keywords is None:
            keywords = []
        if entities is None:
            entities = []
        if categories is None:
            categories = []
        if video_metadata is None:
            video_metadata = {}

        pg = _get_pdf_gen()
        out_path = Path(output_dir) if output_dir else None

        content = PDFContent(
            title=title,
            subject=subject,
            author=author,
            publish_time=video_metadata.get("publish_time", ""),
            summary=summary,
            keywords=keywords,
            entities=entities,
            sentiment=sentiment,
            sentiment_label=sentiment_label,
            categories=categories,
            video_metadata=video_metadata,
            transcript=transcript,
            ocr_text=ocr_text,
            subtitle_text=subtitle_text,
        )

        filepath = pg.generate_pdf_impl(content, output_dir=out_path)

        return _json_result(True, filepath=filepath)

    except Exception as e:
        logger.error(f"[tool:generate_pdf_report] 执行失败: {type(e).__name__}: {e}")
        return _json_result(False, error=str(e))


@tool
def generate_batch_pdf_report(
    reports: List[Dict[str, Any]],
    batch_title: str = "批量分析报告",
    output_dir: str = "",
) -> str:
    """批量生成多个视频分析报告的合并 PDF。

    将所有视频的分析报告合并到同一个 PDF 文件中，每个视频独立成章。

    Args:
        reports: 报告数据列表，每项结构同 generate_pdf_report 的参数
        batch_title: 批量报告总标题
        output_dir: 输出目录路径（可选，默认 ./data/output）

    Returns:
        JSON字符串，格式: {"success": bool, "file_path": str, "count": int, "error": str|null}
    """
    try:
        pg = _get_pdf_gen()
        out_path = Path(output_dir) if output_dir else None

        contents: List[PDFContent] = []
        for r in reports:
            vm = r.get("video_metadata", {}) or {}
            contents.append(PDFContent(
                title=r.get("title", ""),
                subject=r.get("subject", "抖音视频内容分析报告"),
                author=r.get("author", ""),
                publish_time=vm.get("publish_time", ""),
                summary=r.get("summary", ""),
                keywords=r.get("keywords", []),
                entities=r.get("entities", []),
                sentiment=r.get("sentiment", 0.5),
                sentiment_label=r.get("sentiment_label", "中性"),
                categories=r.get("categories", []),
                video_metadata=vm,
                transcript=r.get("transcript", ""),
                ocr_text=r.get("ocr_text", ""),
                subtitle_text=r.get("subtitle_text", ""),
            ))

        filepath = pg.generate_batch_pdf_impl(contents, batch_title, output_dir=out_path)

        return _json_result(True, filepath=filepath, count=len(contents))

    except Exception as e:
        logger.error(f"[tool:generate_batch_pdf_report] 执行失败: {type(e).__name__}: {e}")
        return _json_result(False, error=str(e))


def _json_result(success: bool, **kwargs: Any) -> str:
    """构建统一的 JSON 返回结果"""
    import json
    result: Dict[str, Any] = {"success": success}
    for k, v in kwargs.items():
        if v is not None:
            result[k] = v
    if "error" not in result:
        result["error"] = None
    return json.dumps(result, ensure_ascii=False, indent=2)