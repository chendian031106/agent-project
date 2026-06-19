"""
抖音多智能体内容聚合系统 — 会话式智能交互界面

架构:
  - 左栏: 会话列表（新建/切换/删除）
  - 右区: 智能体对话窗口（可滚动查看上下文）
  - 底栏: 输入框 + 发送
  - 后端: Qwen 3.7-Max (DashScope) 驱动意图识别与内容生成

使用方式:
  streamlit run ui/main.py
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests
import streamlit as st

# ── 页面配置 ──────────────────────────────────────────────
st.set_page_config(
    page_title="Douyin Crew — 智能助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 常量 ──────────────────────────────────────────────────
from utils.config import settings

API_BASE = "http://localhost:8000/api"
CONVERSATIONS_FILE = Path("./data/conversations.json")
DASHSCOPE_API_KEY = settings.DASHSCOPE_API_KEY
DASHSCOPE_BASE = settings.DEEPSEEK_API_BASE or "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen-max"  # Qwen 3.7-Max

# 确保 data 目录存在
Path("./data").mkdir(parents=True, exist_ok=True)

# ── 会话管理 ──────────────────────────────────────────────


def load_conversations() -> List[Dict[str, Any]]:
    if CONVERSATIONS_FILE.exists():
        try:
            data = json.loads(CONVERSATIONS_FILE.read_text("utf-8"))
            # 确保每条会话都有 id
            for c in data:
                if "id" not in c:
                    c["id"] = str(datetime.now().timestamp())
            return data
        except Exception:
            pass
    return []


def save_conversations(convs: List[Dict[str, Any]]) -> None:
    CONVERSATIONS_FILE.write_text(
        json.dumps(convs, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def get_current_conversation() -> Dict[str, Any]:
    convs = st.session_state.conversations
    cid = st.session_state.current_cid
    for c in convs:
        if c["id"] == cid:
            return c
    # fallback: 创建默认会话
    new_c = {"id": cid, "title": "新会话", "messages": []}
    convs.append(new_c)
    save_conversations(convs)
    return new_c


def update_conversation_title(cid: str, first_msg: str) -> None:
    """根据第一条用户消息自动生成会话标题（取前 20 字）"""
    for c in st.session_state.conversations:
        if c["id"] == cid:
            title = first_msg.strip()[:20]
            if len(first_msg) > 20:
                title += "…"
            c["title"] = title
            save_conversations(st.session_state.conversations)
            break


# ── Qwen 3.7-Max 调用 ─────────────────────────────────


def call_qwen(
    messages: List[Dict[str, str]],
    system_prompt: str = "",
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """调用 Qwen 3.7-Max（通过 DashScope OpenAI 兼容接口）

    约束: system_prompt 控制输出简洁性，max_tokens 限制长度。
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=DASHSCOPE_API_KEY or "not_set",
        base_url=DASHSCOPE_BASE,
    )
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.extend(messages)

    try:
        resp = client.chat.completions.create(
            model=QWEN_MODEL,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"【LLM 调用失败】{type(e).__name__}: {e}"


# ── 意图识别与动作路由 ──────────────────────────────


# 系统提示词 — 强制输出简洁
COMPACT_SYSTEM = """你是一个专业的抖音视频分析助手。
回答要求：
1. 只输出核心信息，不要废话、不要客套、不要总结
2. 用简洁的中文，关键词优先
3. 列表形式呈现，不要大段文字
4. 不知道就说"无法获取"，不要编造
5. 输出控制在 200 字以内
"""


def detect_intent(user_input: str) -> str:
    """判断用户意图类型: crawl / query / chat"""
    # 爬取意图：包含爬取/下载等词，或包含抖音链接
    if re.search(r"(爬取|下载|抓取|采集)", user_input):
        return "crawl"
    if re.search(r"douyin\.com|v\.douyin|iesdouyin", user_input, re.I):
        return "crawl"

    # 查询已爬取内容
    if re.search(r"(刚[刚才].*视频|之前.*视频|视频.*讲|什么内容|说了什么|主要讲)", user_input):
        return "query"
    if re.search(r"(总结|摘要|分析|关键词|分类|情感)", user_input):
        return "query"
    if re.search(r"(检索|搜索|查找|知识库|问答)", user_input):
        return "query"

    return "chat"


def execute_crawl(url: str) -> str:
    """执行爬取并通过工作流进行分析"""
    messages = [
        {"role": "user", "content": f"请爬取并分析抖音视频: {url}"}
    ]
    system = COMPACT_SYSTEM + "\n当前任务：用户要求爬取抖音视频。请告知正在执行爬取。"

    notice = call_qwen(messages, system, max_tokens=100)

    # 调用爬虫 API
    try:
        resp = requests.post(
            f"{API_BASE}/crawler/start",
            json={"urls": [url], "auto_monitor": False},
            timeout=60,
        )
        data = resp.json()
        if data.get("success"):
            results = data.get("data", {}).get("results", [])
            parts = [notice, ""]
            for r in results:
                if r.get("status") == "success":
                    vid = r.get("video_id", "?")
                    parts.append(f"✅ 爬取成功 | ID: {vid}")
                    # 触发工作流分析
                    try:
                        requests.post(f"{API_BASE}/analysis/video/{vid}", timeout=120)
                        parts.append(f"  分析完成 ✓")
                    except Exception:
                        parts.append(f"  分析排队中…")
                else:
                    parts.append(f"❌ 失败: {r.get('error', '未知')}")
            return "\n".join(parts)
        else:
            return f"{notice}\n\n❌ 爬取请求失败: {data.get('message', '')}"
    except Exception as e:
        return f"{notice}\n\n❌ 请求异常: {type(e).__name__}: {e}"


def execute_query(question: str) -> str:
    """查询知识库获取答案"""
    # 先尝试知识库检索
    try:
        resp = requests.post(
            f"{API_BASE}/knowledge/query",
            json={"question": question, "top_k": 3, "similarity_threshold": 0.6},
            timeout=30,
        )
        data = resp.json()
        answer = data.get("answer", "")
        confidence = data.get("confidence", 0)

        if answer and confidence > 0.3:
            # 精简回答
            brief = call_qwen(
                [
                    {"role": "user", "content": f"请用一句话极简回答（20字内）：{question}"},
                    {"role": "assistant", "content": answer[:500]},
                ],
                COMPACT_SYSTEM,
                max_tokens=200,
            )
            lines = [brief, ""]
            for src in data.get("sources", [])[:2]:
                lines.append(f"📎 来源: {src.get('video_id', '?')} (相似度 {src.get('similarity', 0):.2f})")
            return "\n".join(lines)
        else:
            # 知识库无结果，用 Qwen 直接回答
            return call_qwen(
                [{"role": "user", "content": f"回答以下问题，不知道就说不知道，不要编造：{question}"}],
                COMPACT_SYSTEM,
                max_tokens=300,
            )
    except Exception as e:
        return f"【查询异常】{type(e).__name__}: {e}"


def execute_chat(user_input: str) -> str:
    """通用对话"""
    return call_qwen(
        [{"role": "user", "content": user_input}],
        COMPACT_SYSTEM,
        temperature=0.5,
        max_tokens=512,
    )


# ── Streamlit UI ─────────────────────────────────────


def init_session_state():
    if "conversations" not in st.session_state:
        st.session_state.conversations = load_conversations()
        if not st.session_state.conversations:
            st.session_state.conversations = [
                {"id": "default", "title": "新会话", "messages": []}
            ]
            save_conversations(st.session_state.conversations)
    if "current_cid" not in st.session_state:
        st.session_state.current_cid = st.session_state.conversations[0]["id"]


def render_sidebar():
    """左侧会话导航栏"""
    with st.sidebar:
        st.markdown("## 🤖 Douyin Crew")
        st.markdown("---")

        # 新建会话
        if st.button("＋ 新建会话", use_container_width=True, type="primary"):
            cid = str(datetime.now().timestamp())
            st.session_state.conversations.insert(0, {
                "id": cid, "title": "新会话", "messages": []
            })
            st.session_state.current_cid = cid
            save_conversations(st.session_state.conversations)
            st.rerun()

        st.markdown("---")
        st.markdown("#### 会话列表")

        # 会话列表
        convs = st.session_state.conversations
        to_delete = None
        for c in convs:
            col1, col2 = st.columns([5, 1])
            with col1:
                is_active = c["id"] == st.session_state.current_cid
                btn_type = "primary" if is_active else "secondary"
                if st.button(c["title"], key=f"c_{c['id']}", use_container_width=True, type=btn_type):
                    st.session_state.current_cid = c["id"]
                    st.rerun()
            with col2:
                if len(convs) > 1:
                    if st.button("✕", key=f"del_{c['id']}"):
                        to_delete = c["id"]

        # 删除会话（延迟执行避免迭代冲突）
        if to_delete:
            st.session_state.conversations = [
                c for c in st.session_state.conversations if c["id"] != to_delete
            ]
            if st.session_state.current_cid == to_delete:
                st.session_state.current_cid = st.session_state.conversations[0]["id"]
            save_conversations(st.session_state.conversations)
            st.rerun()

        st.markdown("---")
        st.caption(f"模型: {QWEN_MODEL}")
        st.caption(f"会话数: {len(convs)}")


def render_chat():
    """右侧对话主区域"""
    conv = get_current_conversation()
    messages = conv["messages"]

    st.markdown(
        """
        <style>
        .stChatFloatingInputContainer {
            bottom: 0;
            padding: 1rem 2rem;
            background: white;
            border-top: 1px solid #eee;
        }
        .chat-container {
            max-width: 900px;
            margin: 0 auto;
            padding-bottom: 80px;
        }
        .msg-user {
            background: #e8f4fd;
            padding: 10px 16px;
            border-radius: 12px;
            margin: 8px 0;
            max-width: 75%;
            margin-left: auto;
        }
        .msg-assistant {
            background: #f0f0f0;
            padding: 10px 16px;
            border-radius: 12px;
            margin: 8px 0;
            max-width: 85%;
        }
        .msg-label {
            font-size: 12px;
            color: #888;
            margin-bottom: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 标题
    st.markdown(f"## 💬 {conv['title']}")
    st.markdown("---")

    # 对话历史（可滚动）
    chat_container = st.container()
    with chat_container:
        if not messages:
            st.info("👋 输入指令开始使用。\n\n例如：\n- `爬取 https://www.douyin.com/video/xxx`\n- `这个视频讲了什么？`\n- `帮我分析之前爬取的视频`")

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                st.markdown(f"**🧑 你**\n{content}")
            else:
                st.markdown(f"**🤖 助手**\n{content}")
            st.markdown("---")

    # 底部输入
    prompt = st.chat_input("输入指令，例如「爬取抖音视频...」或「这个视频讲了什么？」")
    if prompt:
        # 1. 添加用户消息
        messages.append({"role": "user", "content": prompt, "time": datetime.now().isoformat()})
        if len(messages) == 1:
            update_conversation_title(conv["id"], prompt)

        # 2. 显示用户消息
        with chat_container:
            st.markdown(f"**🧑 你**\n{prompt}")
            st.markdown("---")

        # 3. 处理消息
        with st.spinner("🤖 思考中…"):
            intent = detect_intent(prompt)
            if intent == "crawl":
                # 提取 URL
                urls = re.findall(r"https?://[^\s]+", prompt)
                url = urls[0] if urls else prompt
                reply = execute_crawl(url)
            elif intent == "query":
                reply = execute_query(prompt)
            else:
                reply = execute_chat(prompt)

        # 4. 添加助手回复
        messages.append({"role": "assistant", "content": reply, "time": datetime.now().isoformat()})
        save_conversations(st.session_state.conversations)

        # 5. 显示回复并强制刷新
        with chat_container:
            st.markdown(f"**🤖 助手**\n{reply}")
            st.markdown("---")
        st.rerun()


# ── 入口 ──────────────────────────────────────────────


def main():
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()