"""
B站视频总结工具 - Flask 后端主程序
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import time
import queue
import re
import logging
from datetime import datetime
import threading
from pathlib import Path
from flask import Flask, request, jsonify, Response, send_from_directory, send_file, make_response
from flask import stream_with_context

from downloader import (
    get_video_info, download_audio,
    check_cached_audio, check_cached_transcript, check_cached_summary,
    extract_subtitles,
    save_metadata, load_metadata, save_transcript, save_summary,
    list_history, delete_video, InvalidPathError,
    OUTPUT_DIR,
)
from transcriber import transcribe, load_recognizer_for_status
from summarizer import summarize, chat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
KEY_FILE = Path.home() / ".bilibili-summary-key"  # 用户目录，不随项目分享

# SSE 消息队列
sse_queues: dict = {}
sse_lock = threading.Lock()

# 任务并发控制
MAX_CONCURRENT_TASKS = 3
active_tasks = 0
active_tasks_lock = threading.Lock()

# API 限流
RATE_LIMIT_WINDOW = 60  # 秒
RATE_LIMIT_MAX_REQUESTS = 10
rate_limit_store: dict = {}
rate_limit_lock = threading.Lock()

_SAFE_AUDIO_RE = re.compile(r"^[A-Za-z0-9_.-]{1,160}\.mp3$")


def _check_rate_limit(client_ip: str) -> bool:
    now = time.time()
    with rate_limit_lock:
        if client_ip not in rate_limit_store:
            rate_limit_store[client_ip] = []
        requests = rate_limit_store[client_ip]
        requests[:] = [t for t in requests if now - t < RATE_LIMIT_WINDOW]
        if len(requests) >= RATE_LIMIT_MAX_REQUESTS:
            return False
        requests.append(now)
        return True


@app.errorhandler(InvalidPathError)
def handle_invalid_path(_):
    return jsonify({"status": "error", "error": "路径参数不合法"}), 400


def _safe_audio_filename(filename: str) -> str:
    """Limit audio serving to simple cached mp3 file names."""
    if not isinstance(filename, str) or not _SAFE_AUDIO_RE.fullmatch(filename):
        raise InvalidPathError("音频文件名不合法")
    return filename


def _load_key() -> str:
    """从文件读取 DeepSeek Key"""
    try:
        if KEY_FILE.exists():
            return KEY_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return ""


def _save_key(key: str):
    """保存 DeepSeek Key 到文件"""
    try:
        KEY_FILE.write_text(key.strip(), encoding="utf-8")
        if os.name == "posix":
            os.chmod(KEY_FILE, 0o600)
    except Exception:
        pass


def send_sse(task_id: str, event: str, data: dict):
    """向指定任务发送 SSE 消息"""
    with sse_lock:
        if task_id in sse_queues:
            sse_queues[task_id].put(json.dumps({"event": event, "data": data}, ensure_ascii=False))


# ─── 路由 ─────────────────────────────────────────────

@app.route("/")
def index():
    """首页"""
    resp = make_response(send_from_directory(str(STATIC_DIR), "index.html"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return resp


@app.route("/static/<path:filename>")
def static_files(filename):
    """静态文件"""
    return send_from_directory(str(STATIC_DIR), filename)


@app.route("/api/parse", methods=["POST"])
def api_parse():
    """解析视频信息"""
    data = request.get_json()
    url = data.get("url", "").strip()
    
    if not url:
        return jsonify({"status": "error", "error": "请输入视频链接"})
    
    info = get_video_info(url)
    return jsonify({"status": "success", "data": info})


@app.route("/api/process", methods=["POST"])
def api_process():
    """完整流程：解析 → 下载 → 转写 → 总结（SSE 流式推送）"""
    client_ip = request.remote_addr or "unknown"
    if not _check_rate_limit(client_ip):
        return jsonify({"status": "error", "error": "请求过于频繁，请稍后再试"}), 429

    global active_tasks
    with active_tasks_lock:
        if active_tasks >= MAX_CONCURRENT_TASKS:
            return jsonify({"status": "error", "error": f"系统繁忙，最多支持 {MAX_CONCURRENT_TASKS} 个任务同时处理"}), 503
        active_tasks += 1

    data = request.get_json()
    url = data.get("url", "").strip()
    deepseek_key = data.get("deepseek_key", "").strip()
    
    # 如果前端没传 key，从缓存文件中读取
    if not deepseek_key:
        deepseek_key = _load_key()
    elif deepseek_key != _load_key():
        _save_key(deepseek_key)
    
    if not url:
        with active_tasks_lock:
            active_tasks -= 1
        return jsonify({"status": "error", "error": "请输入视频链接"})
    
    task_id = str(int(time.time() * 1000))
    q = queue.Queue()
    
    with sse_lock:
        sse_queues[task_id] = q
    
    logger.info("任务开始 %s url=%s ip=%s", task_id, url[:50], client_ip)

    def process():
        global active_tasks
        try:
            vid = ""
            wav_path = None
            transcript_text = ""

            # ── 步骤1: 解析视频信息 ──
            send_sse(task_id, "step", {
                "step": 1, "name": "解析视频信息",
                "status": "running", "message": "正在获取视频信息...",
            })

            info = get_video_info(url)
            vid = info.get("vid", "")

            if not vid:
                send_sse(task_id, "error", {"error": "无法获取视频信息，请检查链接或网络"})
                q.put(None)
                return

            if info.get("errors"):
                send_sse(task_id, "step", {
                    "step": 1, "name": "解析视频信息",
                    "status": "warning",
                    "message": "视频信息获取成功（有警告）",
                    "data": info,
                })
            else:
                send_sse(task_id, "step", {
                    "step": 1, "name": "解析视频信息",
                    "status": "done",
                    "message": f"标题: {info['title']}",
                    "data": info,
                })

            # 保存元数据
            if vid:
                save_metadata(vid, info)

            # ── 尝试字幕提取（B站/YouTube）──
            transcript_from_subs = False
            platform = info.get("platform", "")
            if vid:
                # 字幕/转写文本统一走 DB 缓存
                transcript_text = check_cached_transcript(vid) or ""
                if not transcript_text and platform in ("bilibili", "youtube"):
                    send_sse(task_id, "step", {
                        "step": 2, "name": "获取文字",
                        "status": "running",
                        "message": "正在尝试提取字幕...",
                    })
                    sub_text = extract_subtitles(vid, url, platform)
                    if sub_text:
                        transcript_text = sub_text
                        transcript_from_subs = True
                elif transcript_text:
                    transcript_from_subs = True

            # ── 步骤2: 获取文字 ──
            if transcript_from_subs:
                # 字幕提取成功，保存到 DB
                if vid and transcript_text:
                    save_transcript(vid, transcript_text)
                send_sse(task_id, "step", {
                    "step": 2, "name": "获取文字",
                    "status": "done",
                    "message": f"字幕提取成功，共 {len(transcript_text)} 字",
                })
                send_sse(task_id, "transcript", {
                    "text": transcript_text, "status": "success",
                })
            else:
                if vid:
                    cached_wav = check_cached_audio(vid)

                if vid and cached_wav:
                    send_sse(task_id, "step", {
                        "step": 2, "name": "获取文字",
                        "status": "done",
                        "message": "使用已缓存的音频",
                    })
                    wav_path = cached_wav
                else:
                    send_sse(task_id, "step", {
                        "step": 2, "name": "获取文字",
                        "status": "running",
                        "message": "正在下载视频音频...",
                    })

                    wav_path = download_audio(url, vid, on_progress=lambda p: send_sse(
                        task_id, "progress", {
                            "step": 2,
                            "percent": p["percent"],
                            "speed": p["speed"],
                            "eta": p["eta"],
                            "message": f"下载中 {p['percent']:.0f}% · {p['speed']} · 剩余 {p['eta']}",
                        }
                    ))

                    if not wav_path:
                        reason = "下载失败"
                        yt_errors = info.get("errors", [])
                        err_text = " ".join(yt_errors).lower()
                        if "premium" in err_text or "大会员" in err_text:
                            reason = "此视频需要大会员才能下载"
                        elif "private" in err_text or "removed" in err_text:
                            reason = "视频已失效或被删除"
                        elif "region" in err_text or "geoblock" in err_text:
                            reason = "该视频在你所在地区不可用"
                        elif "login" in err_text:
                            reason = "需要登录才能下载此视频"
                        send_sse(task_id, "step", {
                            "step": 2, "name": "获取文字", "status": "error",
                            "message": reason,
                        })
                        send_sse(task_id, "done", {"status": "error", "error": reason})
                        return

                    send_sse(task_id, "step", {
                        "step": 2, "name": "获取文字",
                        "status": "done",
                        "message": "音频下载完成",
                    })

            # ── 步骤3: 语音转文字 ──
            if transcript_from_subs:
                send_sse(task_id, "step", {
                    "step": 3, "name": "语音转文字",
                    "status": "skipped",
                    "message": "已通过字幕获取文字，跳过转写",
                })
            else:
                cached_txt = None
                if vid:
                    cached_txt = check_cached_transcript(vid)

                if cached_txt:
                    send_sse(task_id, "step", {
                        "step": 3, "name": "语音转文字",
                        "status": "done",
                        "message": f"使用已缓存的转写，共 {len(cached_txt)} 字",
                    })
                    transcript_text = cached_txt
                    send_sse(task_id, "transcript", {
                        "text": transcript_text, "status": "success",
                    })
                else:
                    send_sse(task_id, "step", {
                        "step": 3, "name": "语音转文字", "status": "running",
                        "message": "正在加载语音模型...",
                    })

                    def progress_callback(current, total):
                        send_sse(task_id, "step", {
                            "step": 3, "name": "语音转文字", "status": "running",
                            "message": f"转写中 {current}/{total} 块...",
                        })

                    result = transcribe(str(wav_path), on_progress=progress_callback)

                    if result["status"] == "error":
                        send_sse(task_id, "step", {
                            "step": 3, "name": "语音转文字",
                            "status": "error",
                            "message": f"转写失败: {result.get('error', '未知错误')}",
                        })
                        send_sse(task_id, "transcript", {"text": "", "status": "error"})
                    else:
                        transcript_text = result["text"]
                        if vid and transcript_text:
                            save_transcript(vid, transcript_text)

                        send_sse(task_id, "step", {
                            "step": 3, "name": "语音转文字",
                            "status": "done",
                            "message": f"转写完成，共 {len(result['text'])} 字",
                            "data": {
                                "duration": result["duration"],
                                "length": len(result["text"]),
                            },
                        })
                        send_sse(task_id, "transcript", {
                            "text": transcript_text, "status": "success",
                        })

            # ── 步骤4: AI 总结 ──
            if deepseek_key and transcript_text:
                # 检查是否有缓存的总结
                cached_sum = None
                if vid:
                    cached_sum = check_cached_summary(vid)

                if cached_sum:
                    send_sse(task_id, "step", {
                        "step": 4, "name": "AI 智能总结",
                        "status": "done",
                        "message": "加载已缓存的总结",
                    })
                    send_sse(task_id, "summary", {
                        "text": cached_sum, "status": "success",
                        "model": "缓存", "tokens": 0,
                    })
                else:
                    send_sse(task_id, "step", {
                        "step": 4, "name": "AI 智能总结", "status": "running",
                        "message": "正在生成总结...",
                    })

                    # 流式总结
                    from summarizer import summarize_stream
                    title = info.get("title", "")
                    full_summary = ""
                    stream_error = None
                    try:
                        send_sse(task_id, "summary_start", {})
                        for chunk in summarize_stream(transcript_text, deepseek_key, title):
                            if chunk.startswith("\x00"):
                                # 进度提示
                                send_sse(task_id, "summary_progress", {"text": chunk[1:]})
                                continue
                            if chunk.startswith("\n\n[") and chunk.endswith("]"):
                                stream_error = chunk.strip("\n[]")
                                break
                            full_summary += chunk
                            send_sse(task_id, "summary_chunk", {"text": chunk})
                    except Exception as e:
                        stream_error = str(e)

                    if stream_error:
                        send_sse(task_id, "step", {
                            "step": 4, "name": "AI 智能总结",
                            "status": "error",
                            "message": stream_error,
                        })
                        send_sse(task_id, "summary", {
                            "text": full_summary, "status": "error",
                            "error": stream_error,
                        })
                    elif full_summary:
                        # 缓存总结
                        if vid:
                            save_summary(vid, full_summary)

                        send_sse(task_id, "step", {
                            "step": 4, "name": "AI 智能总结",
                            "status": "done", "message": "总结完成",
                        })
                        send_sse(task_id, "summary", {
                            "text": full_summary, "status": "success",
                            "model": "deepseek-v4-flash", "tokens": 0,
                        })
                    else:
                        send_sse(task_id, "step", {
                            "step": 4, "name": "AI 智能总结",
                            "status": "error",
                            "message": "生成总结为空",
                        })
            elif not deepseek_key:
                send_sse(task_id, "step", {
                    "step": 4, "name": "AI 智能总结",
                    "status": "skipped",
                    "message": "未提供 DeepSeek Key，跳过总结",
                })
            else:
                send_sse(task_id, "step", {
                    "step": 4, "name": "AI 智能总结",
                    "status": "skipped",
                    "message": "转写失败，无法总结",
                })

            # 完成后清理音频缓存（保留文本和总结）
            if vid and wav_path:
                try:
                    wav_path.unlink(missing_ok=True)
                    logger.info("已清理音频缓存 %s vid=%s", wav_path, vid)
                except Exception: pass

            send_sse(task_id, "done", {"status": "success"})
            logger.info("任务完成 %s vid=%s", task_id, vid)
            
        except Exception as e:
            logger.exception("任务异常 %s", task_id)
            send_sse(task_id, "error", {"error": "处理过程中发生错误，请查看服务端日志"})
            send_sse(task_id, "done", {"status": "error", "error": str(e)})
        finally:
            with active_tasks_lock:
                active_tasks -= 1
            time.sleep(0.3)
            with sse_lock:
                sse_queues.pop(task_id, None)
                q.put(None)
    
    threading.Thread(target=process, daemon=True).start()
    
    def generate():
        while True:
            try:
                msg = q.get(timeout=10)
                if msg is None:
                    break
                yield f"data: {msg}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'event': 'ping', 'data': {}})}\n\n"
                with sse_lock:
                    if task_id not in sse_queues:
                        break
    
    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.route("/api/key", methods=["GET", "POST"])
def api_key():
    """获取或保存 DeepSeek Key"""
    if request.method == "GET":
        key = _load_key()
        # 只返回是否已设置，不返回完整 key（安全考虑）
        return jsonify({
            "has_key": bool(key),
            "key_preview": key[:8] + "..." + key[-4:] if len(key) > 12 else "",
        })
    elif request.method == "POST":
        data = request.get_json()
        key = data.get("key", "").strip()
        if key:
            _save_key(key)
            return jsonify({"status": "success", "message": "Key 已保存"})
        return jsonify({"status": "error", "error": "Key 不能为空"})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """AI 对话接口（支持 RAG 语义检索）"""
    client_ip = request.remote_addr or "unknown"
    if not _check_rate_limit(client_ip):
        return jsonify({"status": "error", "error": "请求过于频繁，请稍后再试"}), 429

    data = request.get_json()
    messages = data.get("messages", [])
    api_key = data.get("deepseek_key", "").strip()
    vid = data.get("vid", "")
    transcript = data.get("transcript", "")
    summary = data.get("summary", "")

    if not api_key:
        api_key = _load_key()

    if not api_key:
        return jsonify({"reply": "", "status": "error", "error": "请先设置 DeepSeek Key"})

    if not messages:
        return jsonify({"reply": "", "status": "error", "error": "消息不能为空"})

    # 取最后一条用户消息
    user_msg = messages[-1].get("content", "") if messages else ""

    # RAG 检索相关段落
    context = ""
    if vid and transcript and user_msg:
        from summarizer import rag_search
        context = rag_search(vid, transcript, user_msg, top_k=5)

    # 构建系统提示
    sys_prompt = "你是一个视频内容讨论助手。请基于以下信息回答用户问题，尽量引用原文内容。如果信息不足以回答，诚实说明。\n\n"
    if summary:
        sys_prompt += f"## AI 总结\n{summary}\n\n"
    if context:
        sys_prompt += f"## 相关原文段落\n{context}\n"
    elif transcript:
        # 无 RAG 命中时用总结 + 转写前 3000 字兜底
        sys_prompt += f"## 转写开头\n{transcript[:3000]}\n"

    # 插入系统提示作为最早消息
    rag_messages = [{"role": "system", "content": sys_prompt}] + messages

    result = chat(rag_messages, api_key)
    return jsonify(result)


@app.route("/api/chat/stream", methods=["POST"])
def api_chat_stream():
    """AI 对话流式接口"""
    client_ip = request.remote_addr or "unknown"
    if not _check_rate_limit(client_ip):
        return jsonify({"status": "error", "error": "请求过于频繁，请稍后再试"}), 429

    data = request.get_json()
    messages = data.get("messages", [])
    api_key = data.get("deepseek_key", "").strip()
    vid = data.get("vid", "")
    transcript = data.get("transcript", "")
    summary = data.get("summary", "")

    if not api_key:
        api_key = _load_key()
    if not api_key:
        return jsonify({"reply": "", "status": "error", "error": "请先设置 DeepSeek Key"})
    if not messages:
        return jsonify({"reply": "", "status": "error", "error": "消息不能为空"})

    user_msg = messages[-1].get("content", "") if messages else ""

    # RAG 检索
    context = ""
    if vid and transcript and user_msg:
        from summarizer import rag_search
        context = rag_search(vid, transcript, user_msg, top_k=5)

    sys_prompt = "你是一个视频内容讨论助手。请基于以下信息回答用户问题，尽量引用原文内容。如果信息不足以回答，诚实说明。\n\n"
    if summary:
        sys_prompt += f"## AI 总结\n{summary}\n\n"
    if context:
        sys_prompt += f"## 相关原文段落\n{context}\n"
    elif transcript:
        sys_prompt += f"## 转写开头\n{transcript[:3000]}\n"

    rag_messages = [{"role": "system", "content": sys_prompt}] + messages

    def generate():
        from summarizer import chat_stream
        try:
            for chunk in chat_stream(rag_messages, api_key):
                yield f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            err_text = "\n\n[错误: {}]".format(e)
            yield "data: {}\n\n".format(json.dumps({"text": err_text}, ensure_ascii=False))
            yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


# ══════════════════════════════════════════
# 对话记录管理
# ══════════════════════════════════════════

def _get_chats_path(vid: str) -> Path:
    """兼容旧路径（已废弃，改用 DB）"""
    return OUTPUT_DIR / "chats" / f"{vid}.json"


def _load_chats(vid: str) -> dict:
    """加载视频的所有对话记录（从 DB）"""
    import sqlite3
    conn = sqlite3.connect(str(OUTPUT_DIR / "data.db"))
    rows = conn.execute(
        "SELECT id, title, messages, created_at FROM chats WHERE vid=? ORDER BY created_at DESC",
        (vid,)
    ).fetchall()
    conn.close()
    result = {"active": "", "list": []}
    for r in rows:
        msgs = json.loads(r[2]) if r[2] else []
        item = {"id": r[0], "title": r[1], "messages": msgs, "time": r[3]}
        result["list"].append(item)
    if result["list"]:
        result["active"] = result["list"][0]["id"]
    return result


def _save_chats(vid: str, data: dict):
    """保存对话记录（到 DB）"""
    from db import save_kb_chat
    active = data.get("active", "")
    for c in data.get("list", []):
        save_kb_chat(c["id"], c.get("title", ""), c.get("messages", []))
    # 保证 vid 列正确
    import sqlite3
    conn = sqlite3.connect(str(OUTPUT_DIR / "data.db"))
    for c in data.get("list", []):
        conn.execute("UPDATE chats SET vid=? WHERE id=?", (vid, c["id"]))
    conn.commit()
    conn.close()


@app.route("/api/chat/<vid>", methods=["GET"])
def api_chat_list(vid):
    """获取视频的所有对话列表"""
    data = _load_chats(vid)
    return jsonify({"status": "success", "data": data})


@app.route("/api/chat/<vid>/save", methods=["POST"])
def api_chat_save(vid):
    """保存/更新对话"""
    req = request.get_json() or {}
    chat_id = req.get("id", "")
    title = req.get("title", "")
    messages = req.get("messages", [])

    if not chat_id or not messages:
        return jsonify({"status": "error", "error": "缺少参数"})

    data = _load_chats(vid)

    # 查找已有对话
    found = False
    for c in data["list"]:
        if c["id"] == chat_id:
            c["title"] = title or c.get("title", "")
            c["messages"] = messages
            c["time"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            found = True
            break

    if not found:
        data["list"].insert(0, {
            "id": chat_id,
            "title": title or f"对话 {len(data['list']) + 1}",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "messages": messages,
        })

    data["active"] = chat_id
    _save_chats(vid, data)

    return jsonify({"status": "success", "id": chat_id})


@app.route("/api/chat/<vid>/delete/<chat_id>", methods=["POST"])
def api_chat_delete(vid, chat_id):
    """删除指定对话"""
    from db import delete_kb_chat
    delete_kb_chat(chat_id)
    return jsonify({"status": "success"})


# ══════════════════════════════════════════
# 文章总结
# ══════════════════════════════════════════

@app.route("/api/article/summarize", methods=["POST"])
def api_article_summarize():
    """文章总结（SSE 流式）"""
    data = request.get_json()
    text = (data.get("text") or "").strip()
    url = (data.get("url") or "").strip()
    title = (data.get("title") or "").strip()
    deepseek_key = (data.get("deepseek_key") or "").strip()

    if not deepseek_key:
        deepseek_key = _load_key()
    if not deepseek_key:
        return jsonify({"status": "error", "error": "请先设置 DeepSeek Key"})
    if not text:
        return jsonify({"status": "error", "error": "文本内容不能为空"})

    title_from_user = bool(title)
    if not title:
        title = text.split("\n")[0][:40] if text else "未命名"

    article_id = "A_" + str(int(time.time() * 1000))
    from db import save_article, save_article_summary
    save_article(article_id, title, url, text, "")

    def _generate_title(article_text: str) -> str:
        """从文章内容提取标题"""
        for line in article_text.split('\n'):
            line = line.strip()
            # 跳过空行、Markdown 标题标记、过短的行
            clean = re.sub(r'^#{1,6}\s*', '', line).strip()
            if len(clean) >= 4:
                # 限制30字以内
                if len(clean) > 30:
                    # 尝试在标点处截断
                    for sep in '，,。！？：:；;、—':
                        idx = clean.find(sep, 10)
                        if 10 <= idx <= 30:
                            return clean[:idx].strip()
                    return clean[:30]
                return clean
        return title  # 找不到合适行，保持占位

    def generate():
        full = ""
        final_title = title
        try:
            from summarizer import summarize_stream
            for chunk in summarize_stream(text, deepseek_key, title):
                if chunk.startswith("\x00"):
                    yield "data: {}\n\n".format(json.dumps({"progress": chunk[1:].strip()}, ensure_ascii=False))
                    continue
                if chunk.startswith("\n\n[") and chunk.endswith("]"):
                    yield "data: {}\n\n".format(json.dumps({"error": chunk.strip("\n[]")}, ensure_ascii=False))
                    yield "data: [DONE]\n\n"; return
                full += chunk
                yield "data: {}\n\n".format(json.dumps({"text": chunk}, ensure_ascii=False))
            if full:
                save_article_summary(article_id, full)
                logger.info("文章总结完成 article_id=%s title_from_user=%s full_len=%d", article_id, title_from_user, len(full))
                if not title_from_user:
                    final_title = _generate_title(text)
                    import sqlite3
                    conn = sqlite3.connect(str(OUTPUT_DIR / "data.db"))
                    conn.execute("UPDATE articles SET title=? WHERE id=?", (final_title, article_id))
                    conn.commit(); conn.close()
                    logger.info("文章标题已更新: %s", final_title)
            yield "data: {}\n\n".format(json.dumps({"done": True, "id": article_id, "title": final_title}, ensure_ascii=False))
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.exception("文章总结异常")
            yield "data: {}\n\n".format(json.dumps({"error": str(e)}, ensure_ascii=False))
            yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


@app.route("/api/article/list")
def api_article_list():
    """列出文章（支持分页和搜索）"""
    from db import list_articles
    page = request.args.get("page", 1, type=int)
    search = request.args.get("search", "").strip()
    items, has_more, total = list_articles(page=page, search=search if search else "")
    return jsonify({"status": "success", "articles": items, "total": total, "page": page, "has_more": has_more})


@app.route("/api/article/<article_id>")
def api_article_detail(article_id):
    """获取单篇文章"""
    from db import load_article
    a = load_article(article_id)
    if not a:
        return jsonify({"status": "error", "error": "文章不存在"}), 404
    return jsonify({"status": "success", "data": a})


@app.route("/api/article/<article_id>/delete", methods=["POST"])
def api_article_delete(article_id):
    from db import delete_article
    delete_article(article_id)
    return jsonify({"status": "success"})


# ══════════════════════════════════════════
# 知识库页面 & API
# ══════════════════════════════════════════

@app.route("/knowledge")
def knowledge_page():
    """知识库问答页面"""
    resp = make_response(send_from_directory(str(STATIC_DIR), "knowledge.html"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return resp


@app.route("/api/kb/list")
def api_kb_list():
    """列出所有已解析视频的摘要信息（支持分页）"""
    from downloader import list_history, load_metadata
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    items = list_history()
    total = len(items)

    # 分页
    start = (page - 1) * per_page
    page_items = items[start:start + per_page]

    result = []
    for item in page_items:
        vid = item.get("vid", "")
        meta = load_metadata(vid) or {}
        result.append({
            "vid": vid,
            "title": item.get("title", ""),
            "uploader": item.get("uploader", ""),
            "duration": meta.get("duration_str", ""),
            "thumbnail": f"/api/thumbnail/{vid}",
            "processed_at": item.get("processed_at", ""),
        })
    return jsonify({"status": "success", "videos": result, "total": total, "page": page, "per_page": per_page})


@app.route("/api/kb/video/<vid>")
def api_kb_video(vid):
    """获取单个视频的总结信息"""
    from downloader import load_metadata, check_cached_summary
    meta = load_metadata(vid)
    if not meta:
        return jsonify({"status": "error", "error": "视频不存在"}), 404
    summary = check_cached_summary(vid) or ""
    return jsonify({
        "status": "success",
        "vid": vid,
        "title": meta.get("title", ""),
        "uploader": meta.get("uploader", ""),
        "duration": meta.get("duration_str", ""),
        "thumbnail": f"/api/thumbnail/{vid}",
        "processed_at": meta.get("processed_at", ""),
        "summary": summary,
    })


@app.route("/api/thumbnail/<vid>")
def api_thumbnail(vid):
    """代理缓存视频封面（B站防盗链需代理，YouTube 直接重定向）"""
    from downloader import load_metadata
    from db import get_thumbnail_path
    cache_file = get_thumbnail_path(vid)

    if cache_file.exists():
        return send_file(str(cache_file), mimetype="image/jpeg")

    meta = load_metadata(vid)
    thumb_url = meta.get("thumbnail", "") if meta else ""
    if not thumb_url or not thumb_url.startswith("http"):
        return "", 404

    # YouTube 直接重定向封面（无防盗链）
    if vid.startswith("YT_"):
        from flask import redirect
        return redirect(thumb_url.replace("http:", "https:"))

    try:
        import urllib.request
        req = urllib.request.Request(
            thumb_url.replace("http:", "https:"),
            headers={"Referer": "https://www.bilibili.com/", "User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
            from db import THUMB_DIR, save_thumbnail
            THUMB_DIR.mkdir(parents=True, exist_ok=True)
            # ffmpeg 压缩封面（720px、JPEG q=2 即高质量压缩）
            import subprocess, tempfile
            ffmpeg_exe = str(BASE_DIR / "tools" / "ffmpeg.exe")
            try:
                tmp_in = Path(tempfile.gettempdir()) / f"thumb_tmp_{vid}.jpg"
                tmp_out = Path(tempfile.gettempdir()) / f"thumb_cmp_{vid}.jpg"
                tmp_in.write_bytes(data)
                subprocess.run([ffmpeg_exe, "-y", "-i", str(tmp_in), "-vf", "scale=720:-1", "-q:v", "2", str(tmp_out)],
                             capture_output=True, timeout=15)
                if tmp_out.exists(): data = tmp_out.read_bytes()
                tmp_in.unlink(missing_ok=True); tmp_out.unlink(missing_ok=True)
            except Exception: pass
            save_thumbnail(vid, data)
            return send_file(str(cache_file), mimetype="image/jpeg")
    except Exception:
        return "", 404


@app.route("/api/kb/ask", methods=["POST"])
def api_kb_ask():
    """全局知识库问答 — 跨视频 + 文章 RAG"""
    client_ip = request.remote_addr or "unknown"
    if not _check_rate_limit(client_ip):
        return jsonify({"status": "error", "error": "请求过于频繁，请稍后再试"}), 429

    data = request.get_json()
    question = (data.get("question") or "").strip()
    api_key = (data.get("deepseek_key") or "").strip()
    target_vid = (data.get("vid") or "").strip()  # 单视频/文章模式
    history = data.get("history") or []  # 对话历史 [{role, content}, ...]

    if not question:
        return jsonify({"reply": "", "status": "error", "error": "问题不能为空"})
    if not api_key:
        api_key = _load_key()
    if not api_key:
        return jsonify({"reply": "", "status": "error", "error": "请先设置 DeepSeek Key"})

    from downloader import load_metadata, check_cached_transcript
    from kb_index import rank_summary_videos
    from summarizer import rag_search, chat as deepseek_chat
    from db import load_article

    # 查询重写：有历史时，让 AI 把短追问改写成完整查询
    search_query = question
    if history and len(question) <= 15:
        try:
            rewrite_prompt = "根据对话上下文，把用户的追问改写成一个完整、清晰的搜索查询语句（30字以内）。\n\n"
            for m in history[-4:]:
                if isinstance(m, dict) and m.get("role") in ("user", "assistant"):
                    c = m.get("content", "")
                    if c: rewrite_prompt += f"{'用户' if m['role']=='user' else 'AI'}: {c[:300]}\n"
            rewrite_prompt += f"用户追问: {question}\n改写后的查询:"
            r = deepseek_chat([
                {"role": "system", "content": "你是查询改写器，只输出改写后的查询语句，不要解释。"},
                {"role": "user", "content": rewrite_prompt}
            ], api_key)
            rewritten = r.get("reply", "").strip()
            if rewritten and len(rewritten) > 2:
                search_query = rewritten
                logger.info("查询重写: '%s' → '%s'", question, search_query)
        except Exception:
            pass

    # 第一步：用持久化 summary 索引初筛相关视频/文章
    video_scores, has_summaries = rank_summary_videos(search_query, target_vid)

    if not has_summaries:
        return jsonify({"reply": "", "status": "error", "error": "还没有已解析的视频或文章"})
    if target_vid and not video_scores:
        return jsonify({"reply": "指定内容未找到。", "status": "error"})

    # 第二步：取 top 5（排除得分过低的无关内容），用 RAG 精查
    context_parts = []
    sources = []
    max_score = video_scores[0][1] if video_scores else 0
    for rid, score in video_scores[:5]:
        # 得分低于最高分 10% 的视为不相关，排除
        if max_score > 0 and score < max_score * 0.1:
            continue
        if rid.startswith("A_"):
            a = load_article(rid)
            if not a or not a.get("text"): continue
            hits = rag_search(rid, a["text"], search_query, top_k=3)
            if hits:
                title = a.get("title", rid)
                context_parts.append(f"【文章：{title}】\n" + hits)
                sources.append({"vid": rid, "title": title, "type": "article"})
        else:
            transcript = check_cached_transcript(rid)
            if not transcript: continue
            hits = rag_search(rid, transcript, search_query, top_k=3)
            if hits:
                meta = load_metadata(rid) or {}
                title = meta.get("title", rid)
                context_parts.append(f"【视频：{title}】\n" + hits)
                sources.append({"vid": rid, "title": title, "type": "video"})

    if not context_parts:
        return jsonify({"reply": "没有找到相关内容。", "status": "success", "sources": []})

    context = "\n\n=====\n\n".join(context_parts)

    # 构建消息列表
    messages = [{"role": "system", "content": "你是一个知识库助手。回复要求：1) 用加粗标题分点，每点简明扼要 2) 关键结论用**加粗**突出 3) 每个观点标注来源 4) 段落间留空行 5) 不要客套话，直接给答案。"}]

    if history:
        for msg in history:
            if isinstance(msg, dict) and msg.get("role") in ("user", "assistant"):
                content = msg.get("content", "")
                if content and content.strip():
                    messages.append({"role": msg["role"], "content": content[:2000]})

    # 第三步：发给 DeepSeek
    prompt = f"""以下是多个视频/文章的相关原文。请尽量综合所有来源回答，每个来源都标注。如有冲突观点也要说明。直接回答，标注来源。

## 相关原文
{context}

## 用户问题
{question}

请回答："""

    messages.append({"role": "user", "content": prompt})

    from summarizer import chat as deepseek_chat
    result = deepseek_chat(messages, api_key)

    result["sources"] = sources
    return jsonify(result)


# ══════════════════════════════════════════
# 知识库对话记录
# ══════════════════════════════════════════

KB_CHATS_DIR = BASE_DIR / "output" / "_knowledge"
KB_CHATS_FILE = KB_CHATS_DIR / "chats.json"


def _load_kb_chats():
    from db import load_kb_chats
    chats = load_kb_chats()
    # 转为旧格式兼容
    result = {"active": "", "list": []}
    for cid, c in chats.items():
        result["list"].append({"id": c["id"], "title": c["title"], "messages": c["messages"], "time": c.get("time", "")})
    if result["list"]:
        result["active"] = result["list"][0]["id"]
    return result


def _save_kb_chats(data):
    from db import save_kb_chat
    for c in data.get("list", []):
        save_kb_chat(c["id"], c.get("title", ""), c.get("messages", []))


@app.route("/api/kb/chat", methods=["GET"])
def api_kb_chat_list():
    return jsonify({"status": "success", "data": _load_kb_chats()})


@app.route("/api/kb/chat/save", methods=["POST"])
def api_kb_chat_save():
    req = request.get_json() or {}
    chat_id = req.get("id", "")
    title = req.get("title", "")
    messages = req.get("messages", [])
    if not chat_id or not messages:
        return jsonify({"status": "error", "error": "缺少参数"})

    data = _load_kb_chats()
    found = False
    for c in data["list"]:
        if c["id"] == chat_id:
            c["title"] = title or c.get("title", "")
            c["messages"] = messages
            c["time"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            found = True
            break
    if not found:
        data["list"].insert(0, {
            "id": chat_id, "title": title or f"对话 {len(data['list'])+1}",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"), "messages": messages,
        })

    data["active"] = chat_id
    _save_kb_chats(data)
    return jsonify({"status": "success", "id": chat_id})


@app.route("/api/kb/chat/delete/<chat_id>", methods=["POST"])
def api_kb_chat_delete(chat_id):
    from db import delete_kb_chat
    delete_kb_chat(chat_id)
    return jsonify({"status": "success"})


@app.route("/api/model/status", methods=["GET"])
def api_model_status():
    """检查语音模型和 ffmpeg 是否存在"""
    model_dir = BASE_DIR / "models" / "sherpa-onnx-sense-voice-small"
    has_model = (model_dir / "model_q8.onnx").exists() and (model_dir / "tokens.txt").exists()
    has_ffmpeg = (BASE_DIR / "tools" / "ffmpeg.exe").exists()
    return jsonify({"status": "success", "has_model": has_model, "has_ffmpeg": has_ffmpeg})


@app.route("/api/model/download", methods=["POST"])
def api_model_download():
    """下载语音模型（SSE 流式进度）"""
    target = BASE_DIR / "models" / "sherpa-onnx-sense-voice-small"
    target.mkdir(parents=True, exist_ok=True)

    base = "https://www.modelscope.cn/models/xiaowangge/sherpa-onnx-sense-voice-small/resolve/master"
    files = [("model_q8.onnx", f"{base}/model_q8.onnx"),
             ("tokens.txt", f"{base}/tokens.txt")]

    def generate():
        for name, url in files:
            dst = target / name
            if dst.exists():
                yield f"data: {json.dumps({'file': name, 'status': 'skip'}, ensure_ascii=False)}\n\n"
                continue
            yield f"data: {json.dumps({'file': name, 'status': 'start', 'size': 'unknown'}, ensure_ascii=False)}\n\n"
            try:
                # 用流式下载并报告进度
                import requests as req
                resp = req.get(url, stream=True, timeout=300)
                total = int(resp.headers.get("content-length", 0))
                downloaded = 0
                yield f"data: {json.dumps({'file': name, 'status': 'progress', 'percent': 0}, ensure_ascii=False)}\n\n"
                with open(dst, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = min(100, int(downloaded * 100 / total))
                        else:
                            pct = min(99, int(downloaded / (230 * 1024 * 1024) * 100))
                        yield f"data: {json.dumps({'file': name, 'status': 'progress', 'percent': pct}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'file': name, 'status': 'done'}, ensure_ascii=False)}\n\n"
            except Exception as e:
                dst.unlink(missing_ok=True)
                yield f"data: {json.dumps({'file': name, 'status': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"
                return
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


@app.route("/api/tools/download", methods=["POST"])
def api_tools_download():
    """下载 ffmpeg.exe（SSE 流式进度）"""
    import zipfile, io

    tools_dir = BASE_DIR / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    target = tools_dir / "ffmpeg.exe"

    # BtbN GitHub releases，国内有 CDN
    url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"

    def generate():
        if target.exists():
            yield f"data: {json.dumps({'file': 'ffmpeg.exe', 'status': 'skip'}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return
        yield f"data: {json.dumps({'file': 'ffmpeg.exe', 'status': 'start', 'size': '~200MB'}, ensure_ascii=False)}\n\n"
        try:
            import requests as req
            logger.info("开始下载 FFmpeg...")
            resp = req.get(url, stream=True, timeout=600, allow_redirects=True)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            logger.info("FFmpeg 压缩包大小: %d MB", total // 1024 // 1024 if total else 0)
            downloaded = 0
            data = io.BytesIO()
            yield f"data: {json.dumps({'file': 'ffmpeg.exe', 'status': 'progress', 'percent': 0, 'text': '开始传输...'}, ensure_ascii=False)}\n\n"
            for chunk in resp.iter_content(chunk_size=65536):
                data.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = min(99, int(downloaded * 100 / total))
                else:
                    pct = min(99, int(downloaded / (200 * 1024 * 1024) * 100))
                yield f"data: {json.dumps({'file': 'ffmpeg.exe', 'status': 'progress', 'percent': pct}, ensure_ascii=False)}\n\n"
            logger.info("FFmpeg 下载完成，开始解压...")
            yield f"data: {json.dumps({'file': 'ffmpeg.exe', 'status': 'progress', 'percent': 99, 'text': '解压中...'}, ensure_ascii=False)}\n\n"
            data.seek(0)
            with zipfile.ZipFile(data) as zf:
                names = [n for n in zf.namelist() if n.endswith('bin/ffmpeg.exe') and not n.startswith('__MACOSX')]
                if not names:
                    raise Exception(f"压缩包中未找到 ffmpeg.exe")
                with zf.open(names[0]) as src, open(target, 'wb') as dst:
                    dst.write(src.read())
            logger.info("FFmpeg 解压完成")
            yield f"data: {json.dumps({'file': 'ffmpeg.exe', 'status': 'done'}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.exception("FFmpeg 下载失败")
            target.unlink(missing_ok=True)
            yield f"data: {json.dumps({'file': 'ffmpeg.exe', 'status': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"
            return
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


@app.route("/api/status", methods=["GET"])
def api_status():
    """检查服务状态"""
    model_ready = load_recognizer_for_status()
    return jsonify({
        "status": "running",
        "model_loaded": model_ready,
    })


@app.route("/api/health", methods=["GET"])
def api_health():
    """健康检查端点，用于容器化部署或监控探活"""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "active_tasks": active_tasks,
        "max_concurrent_tasks": MAX_CONCURRENT_TASKS,
    })


@app.route("/api/history", methods=["GET"])
def api_history():
    """获取已处理的视频/文章列表（支持分页、搜索和类型筛选）"""
    page = request.args.get("page", 1, type=int)
    search = request.args.get("search", "").strip()
    item_type = request.args.get("type", "video")  # video / article

    per_page = 20
    if item_type == "article":
        from db import list_articles
        items, has_more, total = list_articles(page=page, search=search if search else "")
        data = [{"vid": i["id"], "title": i["title"], "duration_str": "",
                 "uploader": "", "has_transcript": True if i.get("text") else False,
                 "has_summary": i["has_summary"], "thumbnail": "",
                 "processed_at": i["processed_at"], "type": "article"} for i in items]
    else:
        items = list_history(search=search if search else None)
        total = len(items)
        start = (page - 1) * per_page
        page_items = items[start:start + per_page]
        has_more = start + per_page < total
        data = [{"vid": i["vid"], "title": i["title"], "duration_str": i.get("duration_str",""),
                 "uploader": i.get("uploader",""), "has_transcript": i.get("has_transcript",False),
                 "has_summary": i.get("has_summary",False), "thumbnail": i.get("thumbnail",""),
                 "processed_at": i.get("processed_at",""), "type": "video"} for i in page_items]

    return jsonify({
        "status": "success",
        "data": data,
        "total": total,
        "page": page,
        "has_more": has_more,
    })


@app.route("/api/history/<vid>/load", methods=["GET"])
def api_history_load(vid):
    """加载单个视频/文章的完整数据"""
    if vid.startswith("A_"):
        from db import load_article
        a = load_article(vid)
        if not a:
            return jsonify({"status": "error", "error": "文章不存在"})
        info = {"vid": vid, "title": a["title"], "url": a.get("url",""),
                "uploader": "", "duration_str": "", "platform": "", "thumbnail": "",
                "processed_at": a.get("processed_at","")}
        return jsonify({
            "status": "success",
            "data": {
                "info": info,
                "transcript": a.get("text","") or "",
                "summary": a.get("summary","") or "",
            },
        })

    info = load_metadata(vid)
    if not info:
        return jsonify({"status": "error", "error": "视频不存在"})

    transcript = check_cached_transcript(vid) or ""
    summary = check_cached_summary(vid) or ""

    return jsonify({
        "status": "success",
        "data": {
            "info": info,
            "transcript": transcript,
            "summary": summary,
        },
    })


@app.route("/api/history/<vid>/delete", methods=["POST"])
def api_history_delete(vid):
    """删除视频或文章"""
    if vid.startswith("A_"):
        from db import delete_article
        delete_article(vid)
        return jsonify({"status": "success", "message": "已删除"})
    ok = delete_video(vid)
    if ok:
        return jsonify({"status": "success", "message": "已删除"})
    return jsonify({"status": "error", "error": "删除失败"})


@app.route("/api/cookies", methods=["GET"])
def api_cookies_get():
    """读取 B站 Cookie 状态"""
    cookie_path = BASE_DIR / "cookies.txt"
    if cookie_path.exists():
        content = cookie_path.read_text(encoding="utf-8").strip()
        # 统计有效 cookie 行数（排除注释行）
        count = sum(1 for line in content.split("\n") if line.strip() and not line.startswith("#"))
        return jsonify({"has_cookie": True, "count": count})
    return jsonify({"has_cookie": False, "count": 0})


@app.route("/api/cookies", methods=["POST"])
def api_cookies():
    """保存 cookies.txt，用于加速 B 站下载"""
    data = request.get_json()
    cookie_str = data.get("cookie", "").strip()

    if not cookie_str:
        return jsonify({"status": "error", "error": "Cookie 内容不能为空"})

    lines = ["# Netscape HTTP Cookie File", "# http://curl.haxx.se/rfc/cookie_spec.html", "# Generated for bilibili.com"]
    for pair in cookie_str.split(";"):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        name, _, value = pair.partition("=")
        name, value = name.strip(), value.strip()
        if not name or not value:
            continue
        lines.append("\t".join([".bilibili.com", "TRUE", "/", "FALSE", "0", name, value]))

    cookie_path = BASE_DIR / "cookies.txt"
    cookie_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info("保存 B站 Cookie %d 个", len(lines) - 3)

    return jsonify({
        "status": "success",
        "message": f"已保存 {len(lines) - 3} 个 Cookie",
        "path": str(cookie_path),
    })



@app.route("/api/proxy", methods=["POST"])
def api_proxy():
    """保存 YouTube 代理地址"""
    data = request.get_json()
    proxy = data.get("proxy", "").strip()
    if not proxy:
        return jsonify({"status": "error", "error": "代理地址不能为空"})
    if not re.match(r"^(https?|socks5?|http)://", proxy):
        return jsonify({"status": "error", "error": "代理地址格式不正确，需以 http://、https:// 或 socks:// 开头"})
    (BASE_DIR / "proxy.txt").write_text(proxy, encoding="utf-8")
    logger.info("保存代理地址 %s", proxy)
    return jsonify({"status": "success", "message": "已保存，重启生效"})

@app.route("/api/proxy", methods=["GET"])
def api_proxy_get():
    """读取代理配置"""
    p = BASE_DIR / "proxy.txt"
    return jsonify({"proxy": p.read_text(encoding="utf-8").strip() if p.exists() else ""})


@app.route("/api/history/<vid>/resummarize", methods=["POST"])
def api_history_resummarize(vid):
    """重新生成 AI 总结（使用缓存的转写文字）"""
    data = request.get_json() or {}
    deepseek_key = data.get("deepseek_key", "").strip()

    if not deepseek_key:
        deepseek_key = _load_key()

    if not deepseek_key:
        return jsonify({"status": "error", "error": "请先在设置中填写 DeepSeek Key"})

    from downloader import check_cached_transcript, load_metadata, check_cached_summary, save_summary
    transcript = check_cached_transcript(vid)
    if not transcript:
        return jsonify({"status": "error", "error": "该视频没有缓存的转写文字，无法重新总结"})

    summary = summarize(transcript, deepseek_key, load_metadata(vid).get("title", ""))

    if summary["status"] == "success":
        save_summary(vid, summary["summary"])

        # 清除旧的 TTS 缓存（总结文字已变，旧语音无意义）
        # TTS 文件现存在系统临时目录，自动清理，无需手动删除

        return jsonify({
            "status": "success",
            "text": summary["summary"],
            "model": summary.get("model", ""),
            "tokens": summary.get("tokens", 0),
        })
    else:
        return jsonify({"status": "error", "error": summary.get("error", "未知错误")})


@app.route("/api/history/<vid>/resummarize/stream", methods=["POST"])
def api_history_resummarize_stream(vid):
    """重新生成 AI 总结（流式）"""
    data = request.get_json() or {}
    deepseek_key = data.get("deepseek_key", "").strip()

    if not deepseek_key:
        deepseek_key = _load_key()
    if not deepseek_key:
        return jsonify({"status": "error", "error": "请先在设置中填写 DeepSeek Key"})

    transcript = check_cached_transcript(vid)
    if not transcript:
        return jsonify({"status": "error", "error": "该视频没有缓存的转写文字，无法重新总结"})

    from summarizer import summarize_stream
    from downloader import load_metadata
    title = load_metadata(vid).get("title", "") if load_metadata(vid) else ""

    def generate():
        full_text = ""
        try:
            for chunk in summarize_stream(transcript, deepseek_key, title):
                if chunk.startswith("\x00"):
                    yield "data: {}\n\n".format(json.dumps({"progress": chunk[1:].strip()}, ensure_ascii=False))
                    continue
                if chunk.startswith("\n\n[") and chunk.endswith("]"):
                    err = chunk.strip("\n[]")
                    yield "data: {}\n\n".format(json.dumps({"error": err}, ensure_ascii=False))
                    yield "data: [DONE]\n\n"
                    return
                full_text += chunk
                yield "data: {}\n\n".format(json.dumps({"text": chunk}, ensure_ascii=False))
            # 缓存总结
            if full_text:
                save_summary(vid, full_text)
                yield "data: {}\n\n".format(json.dumps({"done": True, "model": "deepseek-v4-flash"}, ensure_ascii=False))
            yield "data: [DONE]\n\n"
        except Exception as e:
            err_text = "\n\n[错误: {}]".format(e)
            yield "data: {}\n\n".format(json.dumps({"text": err_text}, ensure_ascii=False))
            yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


@app.route("/api/tts", methods=["POST"])
def api_tts():
    """将文本合成为语音 MP3（使用 Edge TTS）"""
    data = request.get_json()
    text = data.get("text", "").strip()
    vid = data.get("vid", "").strip()

    if not text:
        # 如果没有传 text，尝试用 vid 读缓存的总结
        if vid:
            from downloader import check_cached_summary
            text = check_cached_summary(vid) or ""
        if not text:
            return jsonify({"status": "error", "error": "没有可朗读的文本"})

    # 按音色生成临时文件（不缓存）
    voice = data.get("voice", "zh-CN-XiaoxiaoNeural")
    import uuid, tempfile
    mp3_path = Path(tempfile.gettempdir()) / f"bili_tts_{uuid.uuid4().hex[:8]}.mp3"

    # 合成语音
    import asyncio
    import edge_tts

    async def _synthesize():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(mp3_path))

    try:
        asyncio.run(_synthesize())
        return send_file(str(mp3_path), mimetype="audio/mpeg")
    except Exception as e:
        return jsonify({"status": "error", "error": f"语音合成失败: {str(e)}"})


@app.route("/api/audio/<vid>/<filename>")
def api_audio(vid, filename):
    """提供缓存的音频文件"""
    from downloader import _AUDIO_DIR
    filename = _safe_audio_filename(filename)
    path = _AUDIO_DIR / f"{vid}_{filename}"
    if path.exists():
        return send_file(str(path), mimetype="audio/mpeg")
    return jsonify({"status": "error", "error": "文件不存在"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3195, debug=False, threaded=True)
