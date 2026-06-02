"""
AI 总结模块 - 使用 DeepSeek API 对转写文字进行总结
"""
import json
import re
import logging
import hashlib
import requests
from collections import OrderedDict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

try:
    from joblib import dump as _dump, load as _load
except Exception:  # pragma: no cover - fallback for minimal installs
    import pickle

    def _dump(obj, path):
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def _load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

SUMMARY_PROMPT = """{title_line}请根据以下语音转写，生成一份结构化总结，控制在 {word_limit} 字以内。

输出时从第一句概括性描述直接开始，严禁任何开场白。

结构要求：
- 开头：一两句话概括核心内容
- 主体：分点列出主要观点（数量不限，不遗漏重要信息）
- 结尾：摘录 1-3 句有价值的原话

禁止事项：
- 禁止输出"好的"、"以下是"、"根据您提供的"、"下面我来"等开场白
- 禁止输出你的身份说明或任务确认语句
- 只基于原文，不添加原文没有的观点

转写：
---
{text}
---"""


def _build_title_line(title: str) -> str:
    if title and title.strip() and title.strip() != "未知标题":
        return f"视频标题：{title.strip()}\n\n"
    return ""


def _calc_word_limit(text_length: int) -> int:
    """按 35% 比例计算总结字数（最少 1000，最多 8000）"""
    return max(1000, min(int(text_length * 0.35), 8000))


# ─── Map-Reduce 总结（长文本） ─────────────────────

CHUNK_SIZE = 6500      # 每段字数
CHUNK_OVERLAP = 600    # 段间重叠
MAP_REDUCE_THRESHOLD = 12000   # 一层 Map-Reduce
HIERARCHICAL_THRESHOLD = 20000 # 层级 Reduce

CHUNK_SUMMARY_PROMPT = """{title_line}下面是一段长视频转写的片段，请提取其中所有有价值的信息，不要遗漏。

输出时从"本段概要"直接开始，严禁任何开场白。

输出格式：
- 本段概要：2-3 句说明本段讲了什么
- 关键信息：逐条列出观点、论据、案例、步骤、数据、结论
- 原话摘录：1-2 句原汁原味的原话

规则：
1. 信息召回率优先于压缩率——遗漏重要信息的代价高于轻微冗长
2. 保留案例、步骤、数据、结论，不遗漏
3. 不新增原文没有的内容
4. 去掉重复与口头禅
5. 禁止输出"好的"、"以下是本段总结"、"这段内容"等开场白

片段：
---
{text}
---"""

FINAL_SUMMARY_PROMPT = """{title_line}请将以下多段总结整合为一份完整的视频总结，控制在 {word_limit} 字以内。

输出时从核心内容概括直接开始，严禁任何开场白。

结构：
- 核心内容：3-5 句话概括全片
- 主要内容：按主题汇总所有重点，合并重复，不丢细节
- 观点摘录：保留有价值的原话

禁止事项：
- 禁止输出"好的"、"以下是"、"根据以上分段总结"、"整合如下"等开场白
- 禁止输出任务说明或身份确认语句
- 覆盖完整性优先于极度简短
- 不新增原文没有的内容

分段总结：
---
{text}
---"""


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """按语义断句切分，段间重叠：下一段继承上一段末尾"""
    if len(text) <= chunk_size:
        return [text]

    import re
    sentences = re.split(r'(?<=[。！？\n])\s*', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) <= chunk_size:
            current += s
        else:
            chunks.append(current)
            tail = current[-CHUNK_OVERLAP:] if CHUNK_OVERLAP > 0 and len(current) > CHUNK_OVERLAP else ""
            current = tail + s
    if current:
        chunks.append(current)

    return chunks


def _summarize_chunk(text: str, api_key: str, title: str) -> str:
    """对单个分段进行总结（非流式）"""
    import requests, json as _json
    prompt = CHUNK_SUMMARY_PROMPT.format(
        text=text,
        title_line=_build_title_line(title),
    )
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": "你是一个专业的视频内容分析助手。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 1500,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(DEEPSEEK_API_URL, headers=headers, data=_json.dumps(payload, ensure_ascii=False).encode("utf-8"), timeout=120)
    resp.raise_for_status()
    return _strip_preamble(resp.json()["choices"][0]["message"]["content"])


def _map_reduce_summary_generator(text: str, api_key: str, title: str):
    """Map-Reduce 生成器，逐段输出进度和最终结果"""
    chunks = _chunk_text(text)
    total = len(chunks)
    yield "\x00⏳ 文本 " + str(len(text)) + " 字，分 " + str(total) + " 段总结…\n"

    # Step 1: Map — 每段独立总结，实时反馈
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        yield "\x00▶ 处理第 " + str(i+1) + "/" + str(total) + " 段 (" + str(len(chunk)) + " 字)...\n"
        try:
            s = _summarize_chunk(chunk, api_key, title)
            chunk_summaries.append(f"## 片段 {i+1}\n{s}")
            yield "\x00✅ 第 " + str(i+1) + " 段完成\n\n"
        except Exception as e:
            chunk_summaries.append(f"## 片段 {i+1}\n[总结失败: {e}]")
            yield "\x00❌ 第 " + str(i+1) + " 段失败: " + str(e) + "\n\n"

    # Step 2: Reduce — 超长文本先分组再最终汇总
    if len(text) > HIERARCHICAL_THRESHOLD:
        yield "\x00📝 段数较多，先分组汇总…\n"
        batched = []
        batch_size = 5
        for b in range(0, total, batch_size):
            batch_text = "\n\n".join(chunk_summaries[b:b + batch_size])
            try:
                import json as _json, requests
                wl = max(500, min(int(len(batch_text) * 0.3), 2000))
                p = FINAL_SUMMARY_PROMPT.format(text=batch_text, word_limit=wl, title_line=_build_title_line(title))
                payload = {
                    "model": "deepseek-v4-flash",
                    "messages": [{"role": "user", "content": p}],
                    "temperature": 0.4, "max_tokens": wl * 4,
                }
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                r = requests.post(DEEPSEEK_API_URL, headers=headers,
                                  data=_json.dumps(payload, ensure_ascii=False).encode("utf-8"), timeout=180)
                r.raise_for_status()
                batched.append(_strip_preamble(r.json()["choices"][0]["message"]["content"]))
                yield "\x00✅ 分组 " + str(b//batch_size+1) + " 完成\n"
            except Exception as e:
                batched.append(f"[分组失败: {e}]")
                yield "\x00❌ 分组失败: " + str(e) + "\n"
        combined = "\n\n".join(batched)
    else:
        combined = "\n\n".join(chunk_summaries)

    yield "\x00📝 生成最终总结…\n"
    word_limit = _calc_word_limit(len(text))
    import json as _json, requests
    prompt = FINAL_SUMMARY_PROMPT.format(
        text=combined, word_limit=word_limit,
        title_line=_build_title_line(title),
    )
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4, "max_tokens": word_limit * 4,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(DEEPSEEK_API_URL, headers=headers,
                        data=_json.dumps(payload, ensure_ascii=False).encode("utf-8"), timeout=180)
    resp.raise_for_status()
    final = _strip_preamble(resp.json()["choices"][0]["message"]["content"])
    yield final


def _map_reduce_summary(text: str, api_key: str, title: str) -> str:
    """Map-Reduce：分段总结 → 汇总（非流式，返回最终文本）"""
    parts = []
    for part in _map_reduce_summary_generator(text, api_key, title):
        parts.append(part)
    # 取最后一段（最终总结）
    return parts[-1].split("---\n", 1)[-1] if "---\n" in parts[-1] else parts[-1]


# ─── 总结 ─────────────────────────────────────

# 常见 AI 开场白模式，在结果中自动剔除
_PREAMBLE_PATTERNS = [
    r'^好的[，,]\s*这是根据[您你]提供的[^，,\n]*[，,]\s*',
    r'^好的[，,]\s*以下是[^，,\n]*[：:]\s*',
    r'^好的[，,]\s*下面我来[^，,\n]*[：:]\s*',
    r'^以下是[^，,\n]*的[总结|结构化总结][：:]\s*',
    r'^下面[是给为]您?[^，,\n]*[总结|整理][的]*[：:]\s*',
    r'^根据[您你]提供的[^，,\n]*[，,]?\s*',
    r'^这是根据[^，,\n]*生成[的]*[：:]\s*',
    r'^我已[经]?[为您你]*[^，,\n]*[，,]\s*',
]


def _strip_preamble(text: str) -> str:
    """移除 AI 输出的客套开场白"""
    for pattern in _PREAMBLE_PATTERNS:
        text = re.sub(pattern, '', text, count=1, flags=re.IGNORECASE)
    return text.strip()


def summarize(text: str, api_key: str, title: str = "") -> dict:
    """
    调用 DeepSeek API 进行内容总结
    """
    if not api_key or not api_key.strip():
        return {"summary": "", "status": "error", "error": "请提供 DeepSeek API Key"}
    
    if not text or not text.strip():
        return {"summary": "", "status": "error", "error": "转写文字为空，无法总结"}

    # 长文本使用 Map-Reduce 分段总结
    if len(text) > MAP_REDUCE_THRESHOLD:
        try:
            summary_text = _strip_preamble(_map_reduce_summary(text, api_key, title))
            return {"summary": summary_text, "status": "success", "model": "deepseek-map-reduce"}
        except Exception as e:
            return {"summary": "", "status": "error", "error": f"Map-Reduce 总结失败: {e}"}
    
    word_limit = _calc_word_limit(len(text))
    
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {api_key.strip()}",
    }
    
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "user", "content": SUMMARY_PROMPT.format(text=text, word_limit=word_limit, title_line=_build_title_line(title))}
        ],
        "temperature": 0.4,
        "max_tokens": word_limit * 4,
    }
    
    try:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        resp = requests.post(DEEPSEEK_API_URL, headers=headers, data=body, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        summary = _strip_preamble(data["choices"][0]["message"]["content"])
        return {
            "summary": summary,
            "status": "success",
            "model": data.get("model", "deepseek-chat"),
            "tokens": data.get("usage", {}).get("total_tokens", 0),
        }
        
    except requests.exceptions.Timeout:
        return {"summary": "", "status": "error", "error": "请求超时，请重试"}
    except requests.exceptions.HTTPError:
        err_msg = "API 请求失败"
        try: err_msg = resp.json().get("error", {}).get("message", err_msg)
        except: pass
        if resp.status_code == 401: err_msg = "API Key 无效"
        elif resp.status_code == 402: err_msg = "余额不足"
        elif resp.status_code == 429: err_msg = "请求频繁"
        return {"summary": "", "status": "error", "error": err_msg}
    except Exception as e:
        return {"summary": "", "status": "error", "error": str(e)}


def chat(messages: list, api_key: str) -> dict:
    """
    对话接口，传入完整消息历史
    messages: [{"role": "user"/"assistant", "content": "..."}]
    """
    if not api_key or not api_key.strip():
        return {"reply": "", "status": "error", "error": "请提供 DeepSeek API Key"}
    
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {api_key.strip()}",
    }
    
    payload = {
        "model": "deepseek-v4-flash",
        "messages": messages,
        "temperature": 0.8,
        "max_tokens": 2000,
    }
    
    try:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        resp = requests.post(DEEPSEEK_API_URL, headers=headers, data=body, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        return {
            "reply": data["choices"][0]["message"]["content"],
            "status": "success",
            "tokens": data.get("usage", {}).get("total_tokens", 0),
        }
    except requests.exceptions.Timeout:
        return {"reply": "", "status": "error", "error": "请求超时"}
    except requests.exceptions.HTTPError:
        err_msg = "API 请求失败"
        try: err_msg = resp.json().get("error", {}).get("message", err_msg)
        except: pass
        if resp.status_code == 401: err_msg = "API Key 无效"
        elif resp.status_code == 402: err_msg = "余额不足"
        return {"reply": "", "status": "error", "error": err_msg}
    except Exception as e:
        return {"reply": "", "status": "error", "error": str(e)}


def summarize_stream(text: str, api_key: str, title: str = ""):
    """流式总结，逐块返回文本；长文本自动切换 Map-Reduce 并显示进度"""
    if len(text) > MAP_REDUCE_THRESHOLD:
        for piece in _map_reduce_summary_generator(text, api_key, title):
            yield piece
        return

    word_limit = _calc_word_limit(len(text))
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "user", "content": SUMMARY_PROMPT.format(text=text, word_limit=word_limit, title_line=_build_title_line(title))}
        ],
        "temperature": 0.4,
        "max_tokens": word_limit * 4,
        "stream": True,
    }
    yield from _strip_preamble_stream(_stream_deepseek(payload, api_key))


def _strip_preamble_stream(chunks):
    """流式输出时先缓冲开头，去掉开场白后再逐字输出"""
    buf = ""
    yielded = False
    for chunk in chunks:
        if yielded:
            yield chunk
            continue
        buf += chunk
        # 累计到一定长度时检查是否有开场白
        if len(buf) > 120 or (buf and chunk in ("\n", "。", "！", "？", "：", ":")):
            cleaned = _strip_preamble(buf)
            if cleaned != buf and cleaned:
                # 有开场白被移除，输出清理后的内容
                yield cleaned
            else:
                yield buf
            yielded = True
    if not yielded and buf:
        yield _strip_preamble(buf)


def chat_stream(messages: list, api_key: str):
    """流式对话，逐块返回文本"""
    payload = {
        "model": "deepseek-v4-flash",
        "messages": messages,
        "temperature": 0.8,
        "max_tokens": 2000,
        "stream": True,
    }
    yield from _stream_deepseek(payload, api_key)


def _stream_deepseek(payload: dict, api_key: str):
    """调用 DeepSeek 流式 API，逐块 yield 文本"""
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {api_key.strip()}",
    }
    try:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        resp = requests.post(DEEPSEEK_API_URL, headers=headers, data=body, timeout=180, stream=True)
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
    except requests.exceptions.Timeout:
        yield "\n\n[请求超时]"
    except requests.exceptions.HTTPError:
        err_msg = "API 请求失败"
        try: err_msg = resp.json().get("error", {}).get("message", err_msg)
        except: pass
        if resp.status_code == 401: err_msg = "API Key 无效"
        elif resp.status_code == 402: err_msg = "余额不足"
        elif resp.status_code == 429: err_msg = "请求频繁，请稍后再试"
        yield f"\n\n[{err_msg}]"
    except Exception as e:
        yield f"\n\n[错误: {e}]"


# ═══════════════════════════════════════════════════════════════
# RAG 语义检索引擎
# ═══════════════════════════════════════════════════════════════

def _split_paragraphs(text: str, min_chars: int = 10) -> list[str]:
    """将转写文字切分为段落（按空行优先，否则按句号切块）"""
    # 先按空行分
    parts = re.split(r'\n\s*\n', text)
    paragraphs = []
    for p in parts:
        p = p.strip()
        if len(p) >= min_chars:
            paragraphs.append(p)
        else:
            # 太短的合并到相邻段落
            if paragraphs:
                paragraphs[-1] += '\n' + p
            else:
                paragraphs.append(p)

    # 如果段落还太少，用句号强行切
    if len(paragraphs) < 3:
        raw = '\n'.join(paragraphs)
        paragraphs = [s.strip() + '。' for s in raw.split('。') if len(s.strip()) >= 10]

    return [p for p in paragraphs if len(p) >= min_chars]


class RAGEngine:
    """轻量 RAG 引擎：TF-IDF + 余弦相似度"""

    def __init__(self):
        self.paragraphs: list[str] = []
        self.matrix = None
        self.vectorizer = None

    def build(self, transcript: str):
        """构建索引"""
        self.paragraphs = _split_paragraphs(transcript)
        if not self.paragraphs:
            return

        # 用字符级 ngram (2-4) 分词，对中文效果好
        self.vectorizer = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(2, 4),
            max_features=5000,
        )
        self.matrix = self.vectorizer.fit_transform(self.paragraphs)

    def search(self, query: str, top_k: int = 5) -> list[str]:
        """搜索最相关段落"""
        if self.matrix is None or not self.paragraphs:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for i in top_indices:
            if scores[i] > 0:
                results.append(self.paragraphs[i])

        return results


# 全局缓存（按 vid 缓存 RAG 引擎，LRU 防止长期运行内存无上限增长）
_RAG_INDEX_VERSION = 1
_RAG_CACHE_MAX = 16
_rag_cache: OrderedDict[str, tuple[str, RAGEngine]] = OrderedDict()


def _fingerprint_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _rag_index_path(vid: str):
    from downloader import vid_dir
    return vid_dir(vid) / "rag_index.joblib"


def _get_cached_engine(vid: str, fingerprint: str) -> RAGEngine | None:
    cached = _rag_cache.get(vid)
    if not cached:
        return None

    cached_fingerprint, engine = cached
    if cached_fingerprint != fingerprint:
        _rag_cache.pop(vid, None)
        return None

    _rag_cache.move_to_end(vid)
    return engine


def _put_cached_engine(vid: str, fingerprint: str, engine: RAGEngine):
    _rag_cache[vid] = (fingerprint, engine)
    _rag_cache.move_to_end(vid)
    while len(_rag_cache) > _RAG_CACHE_MAX:
        _rag_cache.popitem(last=False)


def _load_persisted_engine(vid: str, fingerprint: str) -> RAGEngine | None:
    path = _rag_index_path(vid)
    if not path.exists():
        return None

    checksum_path = path.with_suffix(".sha256")
    if checksum_path.exists():
        try:
            stored = checksum_path.read_text(encoding="utf-8").strip()
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if stored and actual and stored != actual:
                return None
        except Exception:
            pass

    try:
        payload = _load(path)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("version") != _RAG_INDEX_VERSION:
        return None
    if payload.get("fingerprint") != fingerprint:
        return None

    engine = payload.get("engine")
    if not isinstance(engine, RAGEngine):
        return None

    return engine


def _save_persisted_engine(vid: str, fingerprint: str, engine: RAGEngine):
    path = _rag_index_path(vid)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _dump({
            "version": _RAG_INDEX_VERSION,
            "fingerprint": fingerprint,
            "engine": engine,
        }, path)
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_suffix(".sha256").write_text(checksum, encoding="utf-8")
    except Exception:
        pass


def rag_search(vid: str, transcript: str, query: str, top_k: int = 5) -> str:
    """
    RAG 搜索：从转写中找到与问题最相关的段落
    返回拼接好的上下文字符串
    """
    fingerprint = _fingerprint_text(transcript)
    engine = _get_cached_engine(vid, fingerprint)

    if engine is None:
        engine = _load_persisted_engine(vid, fingerprint)

    if engine is None:
        engine = RAGEngine()
        engine.build(transcript)
        _save_persisted_engine(vid, fingerprint, engine)

    _put_cached_engine(vid, fingerprint, engine)

    hits = engine.search(query, top_k)
    if not hits:
        # 没找到 → 用最后一段 + 总结兜底
        return ""

    return "\n\n---\n\n".join(hits)


def rag_clear(vid: str):
    """清除指定视频的 RAG 缓存"""
    _rag_cache.pop(vid, None)
    try:
        path = _rag_index_path(vid)
        if path.exists():
            path.unlink()
    except Exception:
        pass
