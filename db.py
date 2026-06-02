"""
SQLite 数据库存储模块 — 替代原来的文件目录存储。

表结构：
  videos:   vid, url, title, uploader, duration, duration_str, platform, thumbnail, processed_at, transcript, summary
  articles: id, title, url, text, summary, processed_at
  chats:    id, vid, title, messages(JSON), created_at
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sqlite3
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
DB_PATH = OUTPUT_DIR / "data.db"
THUMB_DIR = OUTPUT_DIR / "thumbnails"


def _conn() -> sqlite3.Connection:
    """获取数据库连接（自动创建表和目录）"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _init_tables(conn)
    return conn


def _init_tables(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS videos (
            vid          TEXT PRIMARY KEY,
            url          TEXT,
            title        TEXT,
            uploader     TEXT,
            duration     TEXT,
            duration_str TEXT,
            platform     TEXT,
            thumbnail    TEXT,
            processed_at TEXT,
            transcript   TEXT,
            summary      TEXT
        );
        CREATE TABLE IF NOT EXISTS articles (
            id           TEXT PRIMARY KEY,
            title        TEXT,
            url          TEXT,
            text         TEXT,
            summary      TEXT,
            processed_at TEXT
        );
        CREATE TABLE IF NOT EXISTS chats (
            id         TEXT PRIMARY KEY,
            vid        TEXT,
            title      TEXT,
            messages   TEXT,
            created_at TEXT
        );
    """)


# ─── 文章 ─────────────────────────────────────

def save_article(article_id: str, title: str, url: str, text: str, summary: str = ""):
    db = _conn()
    db.execute("""
        INSERT OR REPLACE INTO articles (id, title, url, text, summary, processed_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (article_id, title, url, text, summary, datetime.now().strftime("%Y-%m-%d %H:%M")))
    db.commit()
    db.close()


def load_article(article_id: str) -> dict | None:
    db = _conn()
    row = db.execute("SELECT * FROM articles WHERE id=?", (article_id,)).fetchone()
    db.close()
    if not row: return None
    return {"id": row[0], "title": row[1], "url": row[2], "text": row[3],
            "summary": row[4], "processed_at": row[5]}


def list_articles(page: int = 1, page_size: int = 20, search: str = ""):
    db = _conn()
    where, params = "", []
    if search:
        where = "WHERE title LIKE ?"
        params = [f"%{search}%"]
    offset = (page - 1) * page_size
    rows = db.execute(
        f"SELECT id, title, url, summary, processed_at FROM articles {where} ORDER BY processed_at DESC LIMIT ? OFFSET ?",
        params + [page_size + 1, offset]
    ).fetchall()
    count = db.execute(f"SELECT COUNT(*) FROM articles {where}", params).fetchone()[0]
    db.close()
    has_more = len(rows) > page_size
    items = [{"id": r[0], "title": r[1], "url": r[2], "has_summary": bool(r[3]),
              "summary": r[3] or "", "processed_at": r[4]} for r in rows[:page_size]]
    return items, has_more, count


def delete_article(article_id: str):
    db = _conn()
    db.execute("DELETE FROM articles WHERE id=?", (article_id,))
    db.commit()
    db.close()


def save_article_summary(article_id: str, summary: str):
    db = _conn()
    db.execute("UPDATE articles SET summary=? WHERE id=?", (summary, article_id))
    db.commit()
    db.close()


def load_all_article_summaries() -> dict[str, str]:
    """返回所有有总结的文章 {id: summary}"""
    db = _conn()
    rows = db.execute("SELECT id, summary FROM articles WHERE summary IS NOT NULL AND summary != ''").fetchall()
    db.close()
    return {r[0]: r[1] for r in rows}


# ─── 视频元数据 ─────────────────────────────────

def save_video(vid: str, info: dict):
    db = _conn()
    db.execute("""
        INSERT OR REPLACE INTO videos (vid, url, title, uploader, duration, duration_str, platform, thumbnail, processed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (vid, info.get("url", ""), info.get("title", ""), info.get("uploader", ""),
          info.get("duration", ""), info.get("duration_str", ""), info.get("platform", ""),
          info.get("thumbnail", ""), datetime.now().strftime("%Y-%m-%d %H:%M")))
    db.commit()
    db.close()


def load_metadata(vid: str) -> dict | None:
    db = _conn()
    row = db.execute("SELECT * FROM videos WHERE vid=?", (vid,)).fetchone()
    db.close()
    if not row: return None
    cols = ["vid", "url", "title", "uploader", "duration", "duration_str", "platform", "thumbnail",
            "processed_at", "transcript", "summary"]
    return dict(zip(cols, row))


# ─── 转写 ─────────────────────────────────────

def save_transcript(vid: str, text: str):
    db = _conn()
    db.execute("UPDATE videos SET transcript=? WHERE vid=?", (text, vid))
    db.commit()
    db.close()


def check_cached_transcript(vid: str) -> str | None:
    db = _conn()
    row = db.execute("SELECT transcript FROM videos WHERE vid=?", (vid,)).fetchone()
    db.close()
    return row[0] if row and row[0] else None


# ─── 总结 ─────────────────────────────────────

def save_summary(vid: str, text: str):
    db = _conn()
    db.execute("UPDATE videos SET summary=? WHERE vid=?", (text, vid))
    db.commit()
    db.close()


def check_cached_summary(vid: str) -> str | None:
    db = _conn()
    row = db.execute("SELECT summary FROM videos WHERE vid=?", (vid,)).fetchone()
    db.close()
    return row[0] if row and row[0] else None


# ─── 封面 ─────────────────────────────────────

def save_thumbnail(vid: str, data: bytes):
    path = THUMB_DIR / f"{vid}.jpg"
    path.write_bytes(data)


def get_thumbnail_path(vid: str) -> Path:
    return THUMB_DIR / f"{vid}.jpg"


# ─── 历史记录 ─────────────────────────────────

def list_history(page: int = 1, page_size: int = 20, search: str = "") -> list[dict]:
    db = _conn()
    where = ""
    params = []
    if search:
        where = "WHERE title LIKE ? OR uploader LIKE ?"
        params = [f"%{search}%", f"%{search}%"]
    offset = (page - 1) * page_size
    rows = db.execute(
        f"SELECT vid, title, uploader, duration_str, platform, thumbnail, processed_at, summary FROM videos {where} ORDER BY processed_at DESC LIMIT ? OFFSET ?",
        params + [page_size + 1, offset]
    ).fetchall()
    count = db.execute(f"SELECT COUNT(*) FROM videos {where}", params).fetchone()[0]
    db.close()

    has_more = len(rows) > page_size
    items = []
    for r in rows[:page_size]:
        items.append({
            "vid": r[0], "title": r[1], "uploader": r[2], "duration_str": r[3],
            "platform": r[4], "thumbnail": f"/api/thumbnail/{r[0]}",
            "processed_at": r[6], "has_summary": bool(r[7]),
        })
    return items, has_more, count


def delete_video(vid: str):
    db = _conn()
    db.execute("DELETE FROM videos WHERE vid=?", (vid,))
    db.execute("DELETE FROM chats WHERE vid=?", (vid,))
    db.commit()
    db.close()
    # 清理封面
    thumb = THUMB_DIR / f"{vid}.jpg"
    thumb.unlink(missing_ok=True)


# ─── 知识库对话 ──────────────────────────────

def load_kb_chats() -> dict:
    """返回 KB 对话（vid 为空的才是知识库全局对话）"""
    db = _conn()
    rows = db.execute("SELECT id, title, messages, created_at FROM chats WHERE vid IS NULL OR vid = '' ORDER BY created_at DESC").fetchall()
    db.close()
    result = {}
    for r in rows:
        msgs = json.loads(r[2]) if r[2] else []
        result[r[0]] = {"id": r[0], "title": r[1], "messages": msgs, "time": r[3]}
    return result


def save_kb_chat(chat_id: str, title: str, messages: list):
    db = _conn()
    db.execute("""
        INSERT INTO chats (id, title, messages, created_at) VALUES (?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET title=excluded.title, messages=excluded.messages
    """, (chat_id, title, json.dumps(messages, ensure_ascii=False), datetime.now().strftime("%Y-%m-%d %H:%M")))
    db.commit()
    db.close()


def delete_kb_chat(chat_id: str):
    db = _conn()
    db.execute("DELETE FROM chats WHERE id=?", (chat_id,))
    db.commit()
    db.close()


# ─── 数据迁移 ─────────────────────────────────

def migrate_from_files():
    """将 output/{vid}/ 目录中的旧数据迁移到 SQLite"""
    from pathlib import Path
    import shutil

    if not OUTPUT_DIR.exists(): return
    db = _conn()
    migrated = 0

    for d in OUTPUT_DIR.iterdir():
        if not d.is_dir() or d.name.startswith("_"): continue
        vid = d.name

        # 检查是否已迁移
        row = db.execute("SELECT vid FROM videos WHERE vid=?", (vid,)).fetchone()
        if row: continue

        # 读 info.json
        info_file = d / "info.json"
        if not info_file.exists(): continue
        info = json.loads(info_file.read_text(encoding="utf-8"))

        # 读 transcript
        transcript_file = d / "transcript.txt"
        transcript = transcript_file.read_text(encoding="utf-8") if transcript_file.exists() else ""

        # 读 summary
        summary_file = d / "summary.md"
        summary = summary_file.read_text(encoding="utf-8") if summary_file.exists() else ""

        # 迁移封面
        old_thumb = d / "thumbnail.jpg"
        new_thumb = THUMB_DIR / f"{vid}.jpg"
        if old_thumb.exists() and not new_thumb.exists():
            shutil.copy2(old_thumb, new_thumb)

        # 写入 DB（thumbnail 存原始 URL）
        db.execute("""
            INSERT OR REPLACE INTO videos (vid, url, title, uploader, duration, duration_str, platform, thumbnail, processed_at, transcript, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (vid, info.get("url", ""), info.get("title", ""), info.get("uploader", ""),
              info.get("duration", ""), info.get("duration_str", ""), info.get("platform", ""),
              info.get("thumbnail", ""), info.get("processed_at", datetime.now().strftime("%Y-%m-%d %H:%M")),
              transcript, summary))
        db.commit()
        migrated += 1
        print(f"  迁移: {vid} ({info.get('title','')[:30]})")

    # 迁移知识库对话
    kb_dir = OUTPUT_DIR / "_knowledge"
    kb_file = kb_dir / "chats.json"
    if kb_file.exists():
        data = json.loads(kb_file.read_text(encoding="utf-8"))
        for c in data.get("list", []):
            cid = c.get("id", "")
            if not cid: continue
            row = db.execute("SELECT id FROM chats WHERE id=?", (cid,)).fetchone()
            if not row:
                db.execute("INSERT OR REPLACE INTO chats (id, title, messages, created_at) VALUES (?, ?, ?, ?)",
                           (cid, c.get("title", ""), json.dumps(c.get("messages", []), ensure_ascii=False),
                            c.get("time", c.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M")))))
                db.commit()
                print(f"  迁移KB对话: {c.get('title','')[:20]}")

    # 迁移视频内对话
    for d in OUTPUT_DIR.iterdir():
        if not d.is_dir() or d.name.startswith("_"): continue
        chat_file = d / "chats.json"
        if not chat_file.exists(): continue
        data = json.loads(chat_file.read_text(encoding="utf-8"))
        for c in data.get("list", []):
            cid = c.get("id", "")
            if not cid: continue
            # 视频内对话用 vid 前缀区分
            db_id = f"{d.name}_{cid}"
            row = db.execute("SELECT id FROM chats WHERE id=?", (db_id,)).fetchone()
            if not row:
                db.execute("INSERT OR REPLACE INTO chats (id, vid, title, messages, created_at) VALUES (?, ?, ?, ?, ?)",
                           (db_id, d.name, c.get("title", ""), json.dumps(c.get("messages", []), ensure_ascii=False),
                            c.get("time", datetime.now().strftime("%Y-%m-%d %H:%M"))))
                db.commit()
                print(f"  迁移视频对话: {d.name} / {c.get('title','')[:20]}")

    db.close()
    print(f"\n迁移完成: {migrated} 个视频")
