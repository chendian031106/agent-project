"""
视频处理工作流 — LangGraph StateGraph

完整流水线：下载 → 语音提取 → OCR提取 → 内容分析 →  PDF报告 → 知识入库
使用条件边做错误处理：任一节点失败则跳过后续处理，直接结束。
"""

import json
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from tools.douyin_crawler import crawl_videos, get_video_info
from utils.logger import logger


# ============ 状态定义 ============


class VideoProcessingState(TypedDict):
    """视频处理工作流状态"""

    # 输入
    url: str
    douyin_id: str  # 抖音号（可选，如不指定则从 url 提取）

    # 下载阶段
    video_info: Optional[dict]
    video_path: Optional[str]
    download_error: Optional[str]

    # 提取阶段
    audio_text: Optional[str]
    ocr_text: Optional[str]
    subtitle_text: Optional[str]
    extract_error: Optional[str]

    # 分析阶段
    analysis_result: Optional[dict]
    analysis_error: Optional[str]

    # 报告阶段
    report_path: Optional[str]
    report_error: Optional[str]

    # PDF 预处理阶段（为 RAG 做准备）
    pdf_chunks: Optional[list]
    pdf_process_error: Optional[str]

    # 知识库阶段（基于 PDF 分块入库）
    knowledge_ingested: bool
    ingest_error: Optional[str]


# ============ 节点函数 ============


def _download_video(state: VideoProcessingState) -> dict:
    """节点1：下载视频"""
    logger.info(f"[workflow] 开始下载: url={state['url']}")

    try:
        douyin_id = state.get("douyin_id") or ""
        raw = crawl_videos.invoke({"douyin_id": douyin_id, "count": 1})

        try:
            result = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            result = {"success": False, "error": "结果解析失败"}

        if result.get("success") and result.get("data"):
            videos = result["data"]
            if isinstance(videos, list) and len(videos) > 0:
                video = videos[0] if isinstance(videos[0], dict) else {}
                return {
                    "video_info": video,
                    "video_path": video.get("video_path", ""),
                    "download_error": None,
                }

        return {
            "video_info": None,
            "video_path": None,
            "download_error": result.get("error", "下载失败：未获取到视频数据"),
        }

    except Exception as e:
        logger.error(f"[workflow] 下载失败: {type(e).__name__}: {e}")
        return {
            "video_info": None,
            "video_path": None,
            "download_error": str(e),
        }


def _extract_audio(state: VideoProcessingState) -> dict:
    """节点2：语音转文本（预留节点，需接入 STT 工具）"""
    video_path = state.get("video_path")
    if not video_path:
        return {"audio_text": None, "extract_error": "视频路径为空"}

    logger.info(f"[workflow] 开始语音提取: {video_path}")

    # TODO: 接入 SpeechToText 工具
    # from tools.speech_to_text import transcribe_audio
    # audio_text = transcribe_audio.invoke({"video_path": video_path})
    logger.warning("[workflow] STT 工具尚未接入，跳过语音提取")

    return {
        "audio_text": None,
        "extract_error": None,  # 设为 None 让流程继续（可选步骤）
    }


def _extract_ocr(state: VideoProcessingState) -> dict:
    """节点3：OCR 文字识别（预留节点，需接入 OCR 工具）"""
    video_path = state.get("video_path")
    if not video_path:
        return {"ocr_text": None, "subtitle_text": None}

    logger.info(f"[workflow] 开始 OCR 提取: {video_path}")

    # TODO: 接入 OCR 工具
    # from tools.ocr_service import ocr_video
    # ocr_text = ocr_video.invoke({"video_path": video_path})
    logger.warning("[workflow] OCR 工具尚未接入，跳过画面文字提取")

    return {
        "ocr_text": None,
        "subtitle_text": None,
    }


def _analyze_content(state: VideoProcessingState) -> dict:
    """节点4：内容深度分析（使用 AnalyzerAgent）"""
    # 合并所有文本内容
    texts = [
        state.get("audio_text"),
        state.get("ocr_text"),
        state.get("subtitle_text"),
        # 如果有视频标题/描述，也加入分析
        state.get("video_info", {}).get("title", ""),
        state.get("video_info", {}).get("description", ""),
    ]
    all_content = "\n".join([t for t in texts if t])

    if not all_content.strip():
        logger.warning("[workflow] 无文本内容可分析，跳过分析步骤")
        return {"analysis_result": None, "analysis_error": "无文本内容"}

    logger.info(f"[workflow] 开始内容分析（文本长度: {len(all_content)} 字符）")

    try:
        # 延迟导入，避免 analyzer_agent.py 中 crewai 未安装的问题
        from agents.analyzer_agent import AnalyzerAgent

        agent = AnalyzerAgent()
        result = agent.analyze_content(all_content)

        return {
            "analysis_result": result if isinstance(result, dict) else {},
            "analysis_error": result.get("error") if isinstance(result, dict) else None,
        }

    except Exception as e:
        logger.error(f"[workflow] 内容分析失败: {type(e).__name__}: {e}")
        return {"analysis_result": None, "analysis_error": str(e)}


def _generate_report(state: VideoProcessingState) -> dict:
    """节点5：生成 PDF 分析报告（移至分析后立即执行）"""
    analysis = state.get("analysis_result")
    video_info = state.get("video_info")

    if not analysis:
        return {"report_path": None, "report_error": "无分析结果，无法生成报告"}

    logger.info("[workflow] 开始生成 PDF 报告（步骤 1/3：生成文档）")

    try:
        # 延迟导入，避免 pdf_generator 的依赖问题
        from tools.pdf_generator import generate_pdf_report
        title = video_info.get("title", "分析报告") if video_info else "分析报告"
        author = video_info.get("author", "") if video_info else ""

        raw = generate_pdf_report.invoke({
            "title": title,
            "subject": "抖音视频内容分析报告",
            "author": author,
            "summary": analysis.get("summary", "") if isinstance(analysis, dict) else "",
            "keywords": analysis.get("keywords", []) if isinstance(analysis, dict) else [],
            "entities": analysis.get("entities", []) if isinstance(analysis, dict) else [],
            "sentiment": float(analysis.get("sentiment", 0.5)) if isinstance(analysis, dict) else 0.5,
            "sentiment_label": analysis.get("sentiment_label", "中性") if isinstance(analysis, dict) else "中性",
            "categories": analysis.get("categories", []) if isinstance(analysis, dict) else [],
            "video_metadata": video_info or {},
            "transcript": state.get("audio_text") or "",
            "ocr_text": state.get("ocr_text") or "",
            "subtitle_text": state.get("subtitle_text") or "",
        })

        try:
            result = json.loads(raw) if isinstance(raw, str) else raw
            file_path = result.get("filepath") or result.get("file_path", "")
        except (json.JSONDecodeError, TypeError):
            file_path = ""

        return {
            "report_path": file_path,
            "report_error": None if file_path else "报告生成失败",
        }

    except Exception as e:
        logger.error(f"[workflow] 报告生成失败: {type(e).__name__}: {e}")
        return {"report_path": None, "report_error": str(e)}


def _process_pdf(state: VideoProcessingState) -> dict:
    """节点6：PDF 文档预处理 — 文本提取 + 语义分块（为 RAG 做准备）

    将上一步生成的 PDF 报告进行：
    1. 逐页解析文本
    2. 按语义边界（章节/段落/句子）进行智能分块
    3. 附带页码元数据，便于追溯

    经过 PDF 结构化处理和精细分块的数据，检索召回率更高。
    """
    report_path = state.get("report_path")
    if not report_path:
        return {"pdf_chunks": None, "pdf_process_error": "无 PDF 文件可处理"}

    logger.info(f"[workflow] 开始 PDF 文档预处理（步骤 2/3：分块处理）: {report_path}")

    try:
        # 延迟导入，避免 pgvector/依赖问题
        from tools.pdf_processor import chunk_pdf_content

        video_info = state.get("video_info", {}) or {}
        categories = []
        if state.get("analysis_result"):
            categories = state["analysis_result"].get("categories", []) if isinstance(state["analysis_result"], dict) else []

        raw = chunk_pdf_content.invoke({
            "pdf_path": report_path,
            "chunk_size": 500,
            "chunk_overlap": 80,
        })

        result = json.loads(raw) if isinstance(raw, str) else raw
        success = result.get("success", False) if isinstance(result, dict) else False
        chunks = result.get("chunks", []) if isinstance(result, dict) else []

        if not success or not chunks:
            return {
                "pdf_chunks": None,
                "pdf_process_error": result.get("error", "分块处理失败") if isinstance(result, dict) else "未知错误",
            }

        logger.info(f"[workflow] PDF 预处理完成 | 共 {len(chunks)} 个分块")
        return {
            "pdf_chunks": chunks,
            "pdf_process_error": None,
        }

    except Exception as e:
        logger.error(f"[workflow] PDF 预处理失败: {type(e).__name__}: {e}")
        return {"pdf_chunks": None, "pdf_process_error": str(e)}


def _ingest_knowledge(state: VideoProcessingState) -> dict:
    """节点7：基于 PDF 分块的知识入库（为 RAG 提供高质量语料）

    将经 PDF 预处理后生成的语义分块存入向量知识库。
    相比直接存储原始分析文本，PDF 报告经过结构化排版和章节组织，
    分块后的语义更完整，可大幅提高后续 RAG 检索的召回率。
    """
    pdf_chunks = state.get("pdf_chunks")
    report_path = state.get("report_path")

    if not pdf_chunks:
        # 降级策略：如果没有 PDF 分块但有分析结果，尝试直接入库
        analysis = state.get("analysis_result")
        if analysis:
            logger.warning("[workflow] 无 PDF 分块，降级为直接存储分析结果")
            return _ingest_analysis_direct(state)
        return {"knowledge_ingested": False, "ingest_error": "无 PDF 分块且无分析结果"}

    logger.info(f"[workflow] 开始基于 PDF 分块的知识入库（步骤 3/3：向量存储） | 分块数={len(pdf_chunks)}")

    try:
        # 延迟导入，避免 rag_engine.py 中 pgvector 未安装的问题
        from tools.rag_engine import add_documents

        video_info = state.get("video_info", {}) or {}
        video_id = video_info.get("video_id", "") if isinstance(video_info, dict) else ""
        author = video_info.get("author", "") if isinstance(video_info, dict) else ""
        title = video_info.get("title", "") if isinstance(video_info, dict) else ""

        categories = []
        if state.get("analysis_result"):
            categories = state["analysis_result"].get("categories", []) if isinstance(state["analysis_result"], dict) else []

        # 将 PDF 分块文本逐条入库（每块已按语义边界切分好）
        chunk_texts = [c["text"] for c in pdf_chunks]

        raw = add_documents.invoke({
            "contents": chunk_texts,
            "video_id": video_id,
            "title": f"{title} - PDF 报告分块",
            "author": author,
            "tags": categories if isinstance(categories, list) else [],
            "source_url": state.get("url", ""),
            "metadata": {
                "source": "pdf_report_chunks",
                "pdf_path": report_path or "",
                "total_chunks": len(pdf_chunks),
                "chunk_strategy": "recursive_500_80",
            },
        })

        try:
            result = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(result, dict):
                success = result.get("success", False)
                inserted = result.get("inserted", 0)
            else:
                success = False
                inserted = 0
        except (json.JSONDecodeError, TypeError):
            success = False
            inserted = 0

        logger.info(f"[workflow] 知识入库完成 | 入库={inserted} 条 | 成功={success}")
        return {
            "knowledge_ingested": success,
            "ingest_error": None if success else "入库操作返回失败",
        }

    except Exception as e:
        logger.error(f"[workflow] 知识入库失败: {type(e).__name__}: {e}")
        return {"knowledge_ingested": False, "ingest_error": str(e)}


def _ingest_analysis_direct(state: VideoProcessingState) -> dict:
    """降级方案：将原始分析结果直接入库（不经过 PDF 处理）"""
    analysis = state.get("analysis_result")
    video_info = state.get("video_info")

    logger.warning("[workflow] 使用降级策略：直接存储分析结果")

    try:
        from tools.rag_engine import add_documents

        summary = analysis.get("summary", "") if isinstance(analysis, dict) else ""
        keywords = analysis.get("keywords", []) if isinstance(analysis, dict) else []
        entities = analysis.get("entities", []) if isinstance(analysis, dict) else []
        categories = analysis.get("categories", []) if isinstance(analysis, dict) else []

        contents = [
            f"【摘要】{summary}",
            f"【关键词】{', '.join(keywords) if isinstance(keywords, list) else keywords}",
            f"【实体】{json.dumps(entities, ensure_ascii=False) if isinstance(entities, list) else entities}",
            f"【分类】{', '.join(categories) if isinstance(categories, list) else categories}",
        ]

        title = video_info.get("title", "") if video_info else ""
        author = video_info.get("author", "") if video_info else ""

        raw = add_documents.invoke({
            "contents": contents,
            "video_id": video_info.get("video_id", "") if video_info else "",
            "title": title,
            "author": author,
            "tags": categories if isinstance(categories, list) else [],
            "source_url": state.get("url", ""),
        })

        result = json.loads(raw) if isinstance(raw, str) else raw
        success = result.get("success", False) if isinstance(result, dict) else False

        return {
            "knowledge_ingested": success,
            "ingest_error": None if success else "入库操作返回失败",
        }

    except Exception as e:
        logger.error(f"[workflow] 降级入库失败: {type(e).__name__}: {e}")
        return {"knowledge_ingested": False, "ingest_error": str(e)}


# ============ 条件路由函数 ============


def _route_after_download(state: VideoProcessingState) -> str:
    """下载后路由：成功 → extract_audio，失败 → END"""
    if state.get("download_error"):
        logger.warning(f"[workflow] 下载失败，终止流程: {state['download_error']}")
        return "end"
    return "extract_audio"


def _route_after_extract(state: VideoProcessingState) -> str:
    """提取后路由：有内容 → analyze，无内容但无错误 → analyze（允许空内容），有错误 → END"""
    if state.get("extract_error"):
        # 如果有文本内容但提取部分失败，继续分析
        if state.get("audio_text") or state.get("ocr_text"):
            return "analyze"
        return "end"
    return "analyze"


def _route_after_analyze(state: VideoProcessingState) -> str:
    """分析后路由：成功 → generate（先出 PDF 报告），失败 → END"""
    if state.get("analysis_error") and not state.get("analysis_result"):
        return "end"
    return "generate"


def _route_after_generate(state: VideoProcessingState) -> str:
    """PDF 生成后路由：成功 → process_pdf（预处理），失败 → END"""
    if state.get("report_error"):
        return "end"
    return "process_pdf"


def _route_after_process_pdf(state: VideoProcessingState) -> str:
    """PDF 预处理后路由：有分块 → ingest（入库），无分块但有分析结果 → ingest（降级），否则 → END"""
    if state.get("pdf_process_error") and not state.get("pdf_chunks"):
        # 无分块但有分析结果时 ingest 内部会做降级
        if state.get("analysis_result"):
            return "ingest"
        return "end"
    return "ingest"


def _route_after_ingest(state: VideoProcessingState) -> str:
    """入库后路由：总转到 END"""
    return "end"


# ============ 工作流类 ============


class VideoProcessingWorkflow:
    """视频处理工作流

    使用 LangGraph StateGraph 构造视频处理流水线：
    下载 → 提取 → 分析 → 生成 PDF → PDF 预处理（分块）→ 入库到 RAG 向量库
    """

    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(VideoProcessingState)

        # 注册节点
        workflow.add_node("download", _download_video)
        workflow.add_node("extract_audio", _extract_audio)
        workflow.add_node("extract_ocr", _extract_ocr)
        workflow.add_node("analyze", _analyze_content)
        workflow.add_node("generate", _generate_report)
        workflow.add_node("process_pdf", _process_pdf)
        workflow.add_node("ingest", _ingest_knowledge)

        # 定义边：START → download
        workflow.add_edge(START, "download")

        # 条件边：download → extract_audio / END
        workflow.add_conditional_edges(
            "download",
            _route_after_download,
            {"extract_audio": "extract_audio", "end": END},
        )

        # 顺序边：extract_audio → extract_ocr
        workflow.add_edge("extract_audio", "extract_ocr")

        # 条件边：extract_ocr → analyze / END
        workflow.add_conditional_edges(
            "extract_ocr",
            _route_after_extract,
            {"analyze": "analyze", "end": END},
        )

        # 条件边：analyze → generate（先出 PDF 报告）/ END
        workflow.add_conditional_edges(
            "analyze",
            _route_after_analyze,
            {"generate": "generate", "end": END},
        )

        # 条件边：generate → process_pdf（预处理 PDF 用于 RAG）/ END
        workflow.add_conditional_edges(
            "generate",
            _route_after_generate,
            {"process_pdf": "process_pdf", "end": END},
        )

        # 条件边：process_pdf → ingest（入库到向量库）/ END
        workflow.add_conditional_edges(
            "process_pdf",
            _route_after_process_pdf,
            {"ingest": "ingest", "end": END},
        )

        # ingest → END
        workflow.add_edge("ingest", END)

        return workflow.compile()

    # ---------- 公开接口 ----------

    def run(
        self,
        url: str = "",
        douyin_id: str = "",
    ) -> dict:
        """运行完整视频处理流水线

        Args:
            url: 抖音视频或主页 URL
            douyin_id: 抖音号（可选）

        Returns:
            最终的 VideoProcessingState 字典
        """
        initial_state: VideoProcessingState = {
            "url": url,
            "douyin_id": douyin_id,
            "video_info": None,
            "video_path": None,
            "download_error": None,
            "audio_text": None,
            "ocr_text": None,
            "subtitle_text": None,
            "extract_error": None,
            "analysis_result": None,
            "analysis_error": None,
            "report_path": None,
            "report_error": None,
            "pdf_chunks": None,
            "pdf_process_error": None,
            "knowledge_ingested": False,
            "ingest_error": None,
        }

        logger.info(f"[workflow] VideoProcessingWorkflow 开始运行 | url={url} | douyin_id={douyin_id}")
        result = self.graph.invoke(initial_state)
        logger.info(f"[workflow] VideoProcessingWorkflow 运行完成")

        return dict(result)