"""SQLite persistence: users, clicks, served impressions, bookmarks, publisher articles."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, mind_user_id TEXT, interests TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS clicks (
    id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, news_id TEXT NOT NULL, source TEXT, ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS clicks_user ON clicks(user_id, ts);
CREATE INDEX IF NOT EXISTS clicks_ts ON clicks(ts);
-- §1.2: every article shown in the feed is logged; `clicked` flips to 1 when the user opens it.
CREATE TABLE IF NOT EXISTS impressions (
    id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, news_id TEXT NOT NULL, position INTEGER NOT NULL,
    served_at REAL NOT NULL, clicked INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS impressions_user ON impressions(user_id, news_id);
CREATE TABLE IF NOT EXISTS bookmarks (
    user_id TEXT NOT NULL, news_id TEXT NOT NULL, ts REAL NOT NULL, PRIMARY KEY (user_id, news_id)
);
CREATE TABLE IF NOT EXISTS articles (
    news_id TEXT PRIMARY KEY, category TEXT NOT NULL, subcategory TEXT NOT NULL, title TEXT NOT NULL,
    abstract TEXT NOT NULL, url TEXT NOT NULL, publisher TEXT, ts REAL NOT NULL
);
"""


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    def _q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    # users
    def create_user(self, name: str, interests: list[str], mind_user_id: str | None) -> dict:
        uid = uuid.uuid4().hex
        self._q(
            "INSERT INTO users (id, name, mind_user_id, interests, created_at) VALUES (?,?,?,?,?)",
            (uid, name, mind_user_id, json.dumps(interests), time.time()),
        )
        return self.get_user(uid)

    def get_user(self, uid: str) -> dict | None:
        rows = self._q("SELECT * FROM users WHERE id=?", (uid,))
        if not rows:
            return None
        u = dict(rows[0])
        u["interests"] = json.loads(u["interests"])
        return u

    def set_interests(self, uid: str, interests: list[str]) -> None:
        self._q("UPDATE users SET interests=? WHERE id=?", (json.dumps(interests), uid))

    def all_users(self) -> list[dict]:
        return [dict(r) for r in self._q("SELECT id, mind_user_id FROM users")]

    # clicks + impressions
    def add_click(self, uid: str, news_id: str, source: str | None) -> None:
        now = time.time()
        with self.lock:
            self.db.execute("INSERT INTO clicks (user_id, news_id, source, ts) VALUES (?,?,?,?)", (uid, news_id, source, now))
            self.db.execute(
                "UPDATE impressions SET clicked=1 WHERE id=(SELECT id FROM impressions WHERE user_id=? AND news_id=? "
                "ORDER BY served_at DESC LIMIT 1)",
                (uid, news_id),
            )

    def clicks(self, uid: str) -> list[tuple[str, float]]:
        return [(r["news_id"], r["ts"]) for r in self._q("SELECT news_id, ts FROM clicks WHERE user_id=? ORDER BY ts", (uid,))]

    def all_clicks(self) -> list[tuple[str, str]]:
        return [(r["user_id"], r["news_id"]) for r in self._q("SELECT DISTINCT user_id, news_id FROM clicks")]

    def recent_click_counts(self, since: float) -> dict[str, int]:
        rows = self._q("SELECT news_id, COUNT(*) AS n FROM clicks WHERE ts>=? GROUP BY news_id", (since,))
        return {r["news_id"]: r["n"] for r in rows}

    def log_impressions(self, uid: str, news_ids: list[str]) -> None:
        now = time.time()
        with self.lock:
            self.db.executemany(
                "INSERT INTO impressions (user_id, news_id, position, served_at) VALUES (?,?,?,?)",
                [(uid, n, i + 1, now) for i, n in enumerate(news_ids)],
            )

    def stats(self) -> dict:
        r = self._q(
            "SELECT (SELECT COUNT(*) FROM users) AS users, (SELECT COUNT(*) FROM clicks) AS clicks, "
            "(SELECT COUNT(*) FROM impressions) AS impressions, "
            "(SELECT COALESCE(SUM(clicked),0) FROM impressions) AS clicked_impressions"
        )[0]
        s = dict(r)
        s["feed_ctr"] = round(s["clicked_impressions"] / s["impressions"], 4) if s["impressions"] else None
        return s

    # bookmarks
    def bookmarks(self, uid: str) -> list[str]:
        return [r["news_id"] for r in self._q("SELECT news_id FROM bookmarks WHERE user_id=? ORDER BY ts DESC", (uid,))]

    def set_bookmark(self, uid: str, news_id: str, on: bool) -> None:
        if on:
            self._q("INSERT OR IGNORE INTO bookmarks (user_id, news_id, ts) VALUES (?,?,?)", (uid, news_id, time.time()))
        else:
            self._q("DELETE FROM bookmarks WHERE user_id=? AND news_id=?", (uid, news_id))

    # publisher articles
    def add_article(self, a: dict) -> None:
        self._q(
            "INSERT INTO articles (news_id, category, subcategory, title, abstract, url, publisher, ts) VALUES (?,?,?,?,?,?,?,?)",
            (a["news_id"], a["category"], a["subcategory"], a["title"], a["abstract"], a["url"], a.get("publisher"), time.time()),
        )

    def articles(self) -> list[dict]:
        return [dict(r) for r in self._q("SELECT * FROM articles ORDER BY ts")]
