"""
B站视频下载模块 - 使用 yt-dlp 下载音频并提取为 WAV 16kHz

存储结构：
  output/{vid}/
    ├── info.json        # 视频元数据（标题、UP主、时长、封面、URL）
    ├── audio.wav        # 音频缓存
    ├── transcript.txt   # 转写文本缓存
    └── summary.md       # AI 总结缓存
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import re
import json
import logging
import subprocess
import threading
from datetime import datetime
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
PYTHON_EXE = BASE_DIR / "py310" / "python.exe"
_SAFE_VID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class InvalidPathError(ValueError):
    """Raised when a user-provided id would escape the output directory."""


def _run_ytdlp(args: list, timeout: int = 60):
    """运行 yt-dlp（使用 python -m 方式避免 exe shebang 路径问题）"""
    cmd = [str(PYTHON_EXE), "-m", "yt_dlp"] + args
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        cmd, capture_output=True, encoding="utf-8", errors="replace",
        timeout=timeout, env=env,
    )


def _find_ffmpeg() -> str | None:
    """查找可用的 ffmpeg"""
    ffmpeg_paths = [
        str(BASE_DIR / "tools" / "ffmpeg.exe"),
        r"F:\ffmpeg\ffmpeg-6.0-essentials_build\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        "ffmpeg",
    ]
    for fp in ffmpeg_paths:
        try:
            subprocess.run([fp, "-version"], capture_output=True, timeout=5)
            return fp
        except Exception:
            continue
    return None


# ─── 视频信息 ─────────────────────────────────────────

def _clean_url(url: str) -> str:
    """清洗视频链接，去掉无关的查询参数（如 t、spm_id_from 等）"""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    # B站：去掉所有查询参数
    if "bilibili.com" in parsed.netloc:
        return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))
    # YouTube：只保留 v 参数
    if "youtube.com" in parsed.netloc or "youtu.be" in parsed.netloc:
        qs = parse_qs(parsed.query)
        v = qs.get("v", [None])[0]
        clean_qs = urlencode({"v": v}) if v else ""
        return urlunparse((parsed.scheme, parsed.netloc, path, "", clean_qs, ""))
    # 其他平台：去掉所有查询参数
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def get_video_info(url: str) -> dict:
    """获取视频基本信息（支持B站/YouTube等所有yt-dlp支持的平台）"""
    try:
        args = [
            "--print", "%(title)s",
            "--print", "%(uploader)s",
            "--print", "%(duration)s",
            "--print", "%(thumbnail)s",
            "--print", "%(id)s",
            "--print", "%(extractor)s",
            "--socket-timeout", "30",
        ] + _get_cookie_args(url) + _get_proxy_args()
        args += [url]
        result = _run_ytdlp(args, timeout=90)
        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        errors = [l for l in result.stderr.strip().split("\n") if l.strip() and "WARNING" in l]

        if not lines and result.stderr.strip():
            errors.insert(0, f"yt-dlp stderr: {result.stderr[:200]}")

        raw_vid = lines[4] if len(lines) > 4 else ""
        extractor = lines[5] if len(lines) > 5 else ""

        # YouTube 加 YT_ 前缀避免与 B站 BV 号冲突
        if extractor.lower() == "youtube":
            raw_vid = f"YT_{raw_vid}"
            platform = "youtube"
        elif "bilibili" in extractor.lower():
            platform = "bilibili"
        else:
            platform = extractor.lower()

        info = {
            "title": lines[0] if len(lines) > 0 else "未知标题",
            "uploader": lines[1] if len(lines) > 1 else "未知UP主",
            "duration": lines[2] if len(lines) > 2 else "0",
            "thumbnail": lines[3] if len(lines) > 3 else "",
            "vid": raw_vid,
            "url": _clean_url(url),
            "extractor": extractor,
            "platform": platform,
        }

        try:
            duration_sec = int(float(info["duration"]))
            m, s = divmod(duration_sec, 60)
            info["duration_str"] = f"{m}:{s:02d}"
        except (ValueError, TypeError):
            info["duration_str"] = info["duration"]

        info["errors"] = errors
        return info
    except subprocess.TimeoutExpired:
        return {"title": "未知", "uploader": "未知", "duration": "0", "duration_str": "0:00", "errors": ["请求超时"]}
    except Exception as e:
        return {"title": "未知", "uploader": "未知", "duration": "0", "duration_str": "0:00", "errors": [str(e)]}


# ─── 存储路径 ─────────────────────────────────────────

def vid_dir(vid: str) -> Path:
    """兼容旧路径（仅用于封面图路径）"""
    return OUTPUT_DIR / "thumbnails"


def save_metadata(vid: str, info: dict):
    """保存视频元数据到数据库"""
    from db import save_video
    save_video(vid, info)


def load_metadata(vid: str) -> dict | None:
    """加载视频元数据"""
    from db import load_metadata as db_load
    return db_load(vid)


# ─── 缓存检查 ─────────────────────────────────────────

def check_cached_audio(vid: str) -> Path | None:
    """检查是否已有缓存的 WAV 文件"""
    wav = vid_dir(vid) / "audio.wav"
    return wav if wav.exists() else None


def check_cached_transcript(vid: str) -> str | None:
    """检查是否已有缓存的转写文本"""
    from db import check_cached_transcript as db_check
    return db_check(vid)


def check_cached_summary(vid: str) -> str | None:
    """检查是否已有缓存的 AI 总结"""
    from db import check_cached_summary as db_check
    return db_check(vid)


# ─── 字幕提取 ─────────────────────────────────────────

def check_cached_subtitles(vid: str) -> str | None:
    """检查是否有缓存的字幕文本"""
    p = vid_dir(vid) / "subtitles.txt"
    if p.exists():
        content = p.read_text(encoding="utf-8").strip()
        return content if content else None
    return None


def extract_subtitles(vid: str, url: str, platform: str = "bilibili") -> str | None:
    """
    提取 B站/YouTube 视频字幕（自动字幕优先，其次手动字幕）
    返回字幕纯文本，无字幕返回 None
    """
    d = vid_dir(vid)
    d.mkdir(parents=True, exist_ok=True)

    if platform == "bilibili":
        langs = "zh-Hans,zh-CN,zh,zh-TW,zh-Hant,ai-zh"
    else:
        langs = "zh-Hans,zh-CN,zh,zh-TW,en"

    try:
        # 先用 --write-auto-subs 尝试自动字幕
        args = [
            "--write-auto-subs",
            "--sub-lang", langs,
            "--sub-format", "srt",
            "--skip-download",
            "--output", str(d / "subtitles"),
            "--socket-timeout", "30",
        ] + _get_cookie_args(url) + _get_proxy_args() + [url]
        _run_ytdlp(args, timeout=60)

        srt_files = list(d.glob("subtitles.*.srt"))
        if not srt_files:
            args[0] = "--write-subs"
            _run_ytdlp(args, timeout=60)
            srt_files = list(d.glob("subtitles.*.srt"))

        if srt_files:
            text = _parse_srt(srt_files[0])
            if text.strip():
                (d / "subtitles.txt").write_text(text, encoding="utf-8")
                for f in srt_files:
                    try:
                        f.unlink()
                    except Exception:
                        pass
                logger.info("提取字幕成功 %s，共 %d 字", vid, len(text))
                return text

    except Exception:
        logger.debug("字幕提取失败 %s", vid)

    return None


def _parse_srt(srt_path: Path) -> str:
    """解析 SRT 字幕文件，提取纯文本"""
    content = srt_path.read_text(encoding="utf-8", errors="replace")
    content = re.sub(r'^\d+\s*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{3}', '', content)
    content = re.sub(r'<[^>]+>', '', content)
    lines = [l.strip() for l in content.split('\n') if l.strip()]
    return '\n'.join(lines)


def save_transcript(vid: str, text: str):
    """保存转写文本"""
    from db import save_transcript as db_save
    db_save(vid, text)


def save_summary(vid: str, text: str):
    """保存 AI 总结"""
    from db import save_summary as db_save
    db_save(vid, text)


# ─── Cookie 支持 ──────────────────────────────────────

COOKIE_FILE = BASE_DIR / "cookies.txt"
PROXY_FILE = BASE_DIR / "proxy.txt"


def _get_proxy_args() -> list:
    """读取代理配置"""
    if PROXY_FILE.exists():
        proxy = PROXY_FILE.read_text(encoding="utf-8").strip()
        if proxy:
            return ["--proxy", proxy]
    return []

def _get_cookie_args(url_or_vid: str = "") -> list:
    """获取 yt-dlp Cookie 参数"""
    if COOKIE_FILE.exists():
        return ["--cookies", str(COOKIE_FILE)]
    return []


# ─── 下载音频 ─────────────────────────────────────────

def download_audio(url: str, vid: str, on_progress=None) -> Path | None:
    """
    下载视频音频到 output/{vid}/audio.wav
    直接下载 m4a → ffmpeg 转 WAV 16kHz 单声道

    on_progress(dict) — 下载进度回调，收到 {"percent": 12.5, "speed": "2.3MiB/s", "eta": "00:14"}
    """
    d = vid_dir(vid)
    d.mkdir(parents=True, exist_ok=True)

    # 清理残留 .part 文件
    for stale in d.glob("*.part"):
        try:
            stale.unlink()
        except Exception:
            pass

    output_wav = d / "audio.wav"
    temp_dl = d / "_dl"

    try:
        # 步骤1: 流式下载 m4a 音频（可读取进度）
        cmd = [str(PYTHON_EXE), "-m", "yt_dlp",
               "-f", "bestaudio[ext=m4a]/bestaudio/best",
               "-o", str(temp_dl) + ".%(ext)s",
               "--no-playlist", "--no-mtime",
               "--concurrent-fragments", "5",
               "--newline"] + _get_cookie_args(url) + _get_proxy_args()
        cmd += [url]

        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            encoding="utf-8", errors="replace",
            env=env, bufsize=1,
        )

        # 解析下载进度: [download]  12.5% of ~37.45MiB at  2.3MiB/s ETA 00:14
        progress_re = re.compile(
            r'\[download\]\s+([\d.]+)%\s+of\s+[~\s]*([\d.]+[KMG]?iB).*?at\s+([\d.]+\s*[KMG]?i?B/s).*?ETA\s+(\S+)'
        )

        output_lines = []
        for line in proc.stdout:
            line = line.rstrip()
            output_lines.append(line)

            m = progress_re.search(line)
            if m and on_progress:
                on_progress({
                    "percent": float(m.group(1)),
                    "size": m.group(2),
                    "speed": m.group(3),
                    "eta": m.group(4),
                })

        proc.wait(timeout=600)

        # 查找下载的文件
        downloaded = None
        for ext in ["m4a", "mp3", "mp4", "webm", "opus", "aac"]:
            candidate = d / f"_dl.{ext}"
            if candidate.exists() and candidate.stat().st_size > 1024:
                downloaded = candidate
                break

        if not downloaded:
            err_lines = [l for l in output_lines if "ERROR" in l or "WARNING" in l]
            if err_lines:
                logger.warning("yt-dlp issues: %s", err_lines[-3:])
            return None

        # 步骤2: 转 WAV 16kHz 单声道
        try:
            ffmpeg_exe = _find_ffmpeg()

            if not ffmpeg_exe:
                logger.error("ffmpeg 未找到，无法转换音频格式")
                return None

            subprocess.run([
                ffmpeg_exe, "-y",
                "-i", str(downloaded),
                "-ac", "1",
                "-ar", "16000",
                "-sample_fmt", "s16",
                str(output_wav)
            ], capture_output=True, text=True, timeout=120)
        finally:
            if downloaded.exists():
                try:
                    downloaded.unlink()
                except Exception:
                    pass

        return output_wav if output_wav.exists() else None

    except subprocess.TimeoutExpired:
        logger.error("下载超时")
        return None
    except Exception as e:
        logger.exception("下载异常")
        return None


# ─── 历史记录 ─────────────────────────────────────────

def list_history(search: str = None) -> list[dict]:
    """列出所有已处理的视频（兼容旧接口，不分页）"""
    from db import list_history as db_list
    items, _, _ = db_list(page=1, page_size=10000, search=search or "")
    return items


def delete_video(vid: str) -> bool:
    """删除整个视频文件夹"""
    import shutil
    try:
        d = vid_dir(vid)
    except InvalidPathError:
        return False
    if d.exists():
        shutil.rmtree(d)
        return True
    return False
