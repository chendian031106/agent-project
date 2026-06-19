"""
PDF 文档预处理工具

负责将已生成的 PDF 报告进行：
1. 文本提取（逐页解析）
2. 语义分块（按章节/段落/句子边界）
3. 元数据标注（页码、章节标题）

输出结构化分块数据供 RAG 引擎进行向量化存储，
提高后续检索的召回率。
"""

import json
import os
from typing import Any, Dict, List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool

from utils.logger import logger


@tool
def extract_pdf_text(pdf_path: str) -> str:
    """从 PDF 文件中提取文本内容（逐页）。

    使用 pypdf 逐页解析，保留页面顺序，每页内容附带页码元数据。

    Args:
        pdf_path: PDF 文件的完整路径

    Returns:
        JSON 字符串，格式:
        {"success": bool, "pages": [{"page_num": int, "text": str}], "total_pages": int, "error": str|null}
    """
    try:
        if not pdf_path or not os.path.isfile(pdf_path):
            return json.dumps(
                {"success": False, "pages": [], "total_pages": 0, "error": f"文件不存在: {pdf_path}"},
                ensure_ascii=False,
            )

        from pypdf import PdfReader

        reader = PdfReader(pdf_path)
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append({"page_num": i, "text": text.strip()})

        logger.info(f"[tool:extract_pdf_text] PDF 解析完成 | 路径={pdf_path} | 页数={len(pages)}")
        return json.dumps(
            {"success": True, "pages": pages, "total_pages": len(pages), "error": None},
            ensure_ascii=False,
        )

    except Exception as e:
        logger.error(f"[tool:extract_pdf_text] PDF 解析失败: {type(e).__name__}: {e}")
        return json.dumps(
            {"success": False, "pages": [], "total_pages": 0, "error": str(e)},
            ensure_ascii=False,
        )


@tool
def chunk_pdf_content(
    pdf_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> str:
    """对 PDF 文档进行语义分块处理。

    先用 pypdf 提取全文，再用 langchain 的 RecursiveCharacterTextSplitter
    按语义边界（章节标题 → 段落 → 句子）进行智能分块。
    每个分块附带来源页码，便于召回后追溯原文位置。

    Args:
        pdf_path: PDF 文件的完整路径
        chunk_size: 每块最大字符数，默认 500
        chunk_overlap: 块间重叠字符数，默认 80（提高上下文连贯性）

    Returns:
        JSON 字符串，格式:
        {"success": bool, "chunks": [{"chunk_id": int, "text": str, "page_num": int, "chars": int}], "total_chunks": int, "error": str|null}
    """
    try:
        if not pdf_path or not os.path.isfile(pdf_path):
            return json.dumps(
                {"success": False, "chunks": [], "total_chunks": 0, "error": f"文件不存在: {pdf_path}"},
                ensure_ascii=False,
            )

        # 1. 提取全文
        from pypdf import PdfReader

        reader = PdfReader(pdf_path)
        all_text = ""
        page_map: List[int] = []  # 每个字符对应的页码
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                all_text += text + "\n"
                page_map.extend([i] * (len(text) + 1))  # +1 为换行符

        if not all_text.strip():
            return json.dumps(
                {"success": False, "chunks": [], "total_chunks": 0, "error": "PDF 中无可提取的文本"},
                ensure_ascii=False,
            )

        # 2. 语义分块 — 按 markdown 章节 → 段落 → 句子
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", "。", "!", "?", ". ", "\uff01", "\uff1f"],
            keep_separator=True,
        )
        raw_chunks = splitter.split_text(all_text)

        # 3. 为每个块找到最可能的页码
        chunks = []
        char_cursor = 0
        for idx, chunk in enumerate(raw_chunks):
            chunk_len = len(chunk)
            # 找到块对应的起始页码
            if char_cursor < len(page_map):
                page_num = page_map[char_cursor]
            else:
                page_num = page_map[-1] if page_map else 1

            chunks.append({
                "chunk_id": idx + 1,
                "text": chunk.strip(),
                "page_num": page_num,
                "chars": chunk_len,
            })
            char_cursor += chunk_len

        logger.info(
            f"[tool:chunk_pdf_content] PDF 分块完成 | 路径={pdf_path} | "
            f"总块数={len(chunks)} | 平均每块={sum(c['chars'] for c in chunks)//max(len(chunks),1)} 字符"
        )
        return json.dumps(
            {"success": True, "chunks": chunks, "total_chunks": len(chunks), "error": None},
            ensure_ascii=False,
        )

    except Exception as e:
        logger.error(f"[tool:chunk_pdf_content] PDF 分块失败: {type(e).__name__}: {e}")
        return json.dumps(
            {"success": False, "chunks": [], "total_chunks": 0, "error": str(e)},
            ensure_ascii=False,
        )


@tool
def process_pdf_for_rag(
    pdf_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    video_id: str = "",
    title: str = "",
    author: str = "",
    tags: Optional[List[str]] = None,
    source_url: str = "",
) -> str:
    """PDF 文档 → RAG 向量库 一站式处理。

    将 PDF 报告经过「文本提取 → 语义分块 → 向量化存储」全流程处理，
    直接写入知识库，供后续智能问答检索使用。
    相比直接存储原始分析结果，经过 PDF 结构化处理和精细分块的数据，
    检索召回率更高、上下文更完整。

    Args:
        pdf_path: PDF 文件的完整路径
        chunk_size: 每块最大字符数，默认 500
        chunk_overlap: 块间重叠字符数，默认 80
        video_id: 关联的抖音视频 ID（可选）
        title: 文档标题（可选）
        author: 博主名称（可选）
        tags: 标签列表（可选）
        source_url: 来源 URL（可选）

    Returns:
        JSON 字符串，格式:
        {"success": bool, "inserted": int, "total_chunks": int, "file_path": str, "error": str|null}
    """
    if tags is None:
        tags = []

    try:
        # 1. 分块
        raw = chunk_pdf_content.invoke({
            "pdf_path": pdf_path,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        })
        result = json.loads(raw) if isinstance(raw, str) else raw

        if not result.get("success"):
            return json.dumps(
                {"success": False, "inserted": 0, "total_chunks": 0, "file_path": pdf_path, "error": result.get("error", "PDF 分块失败")},
                ensure_ascii=False,
            )

        chunks = result.get("chunks", [])
        if not chunks:
            return json.dumps(
                {"success": False, "inserted": 0, "total_chunks": 0, "file_path": pdf_path, "error": "分块结果为空"},
                ensure_ascii=False,
            )

        # 2. 逐块入库 — 每块作为一个独立文档存入向量库
        # 使用 rag_engine 的 add_documents，每块单独传入
        from tools.rag_engine import add_documents

        chunk_texts = [chunk["text"] for chunk in chunks]
        total_chars = sum(chunk["chars"] for chunk in chunks)

        raw_insert = add_documents.invoke({
            "contents": chunk_texts,
            "video_id": video_id,
            "title": title,
            "author": author,
            "tags": tags,
            "source_url": source_url,
            "metadata": {
                "source": "pdf_report",
                "pdf_path": pdf_path,
                "total_chunks": len(chunks),
                "total_chars": total_chars,
                "chunk_strategy": f"recursive_{chunk_size}_{chunk_overlap}",
            },
        })
        insert_result = json.loads(raw_insert) if isinstance(raw_insert, str) else raw_insert
        success = insert_result.get("success", False) if isinstance(insert_result, dict) else False
        inserted = insert_result.get("inserted", 0) if isinstance(insert_result, dict) else 0

        logger.info(
            f"[tool:process_pdf_for_rag] 完成 | 路径={pdf_path} | "
            f"分块={len(chunks)} | 入库={inserted} | 成功={success}"
        )
        return json.dumps(
            {
                "success": success,
                "inserted": inserted,
                "total_chunks": len(chunks),
                "file_path": pdf_path,
                "error": None if success else (insert_result.get("error") if isinstance(insert_result, dict) else "写入失败"),
            },
            ensure_ascii=False,
        )

    except Exception as e:
        logger.error(f"[tool:process_pdf_for_rag] 失败: {type(e).__name__}: {e}")
        return json.dumps(
            {"success": False, "inserted": 0, "total_chunks": 0, "file_path": pdf_path, "error": str(e)},
            ensure_ascii=False,
        )