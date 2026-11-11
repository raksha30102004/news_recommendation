"""FastAPI app: JSON API + static single-page frontend.

Run locally:  uvicorn app.main:app --reload
"""
from __future__ import annotations

import asyncio
import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from recsys.config import ARTIFACTS_DIR, ROOT, TOP_K

from .service import REPORT_METRICS, Recommender
from .store import Store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

STATE_DIR = Path(os.environ.get("STATE_DIR", ROOT / "state"))
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")
RETRAIN_INTERVAL_HOURS = float(os.environ.get("RETRAIN_INTERVAL_HOURS", "24"))
STATIC = Path(__file__).parent / "static"

store = Store(STATE_DIR / "app.db")
rec = Recommender(Path(os.environ.get("ARTIFACTS_DIR", ARTIFACTS_DIR)), STATE_DIR, store)


async def retrain_loop() -> None:
    """Periodic retraining (report §4.10.1 recommends daily)."""
    while True:
        await asyncio.sleep(RETRAIN_INTERVAL_HOURS * 3600)
        try:
            await asyncio.to_thread(rec.retrain)
        except Exception:  # keep serving the previous model if a retrain fails
            log.exception("scheduled retrain failed")


@asynccontextmanager
async def lifespan(_: FastAPI):
    task = asyncio.create_task(retrain_loop()) if RETRAIN_INTERVAL_HOURS > 0 else None
    yield
    if task:
        task.cancel()


app = FastAPI(title="Newsfeed — ALS news recommender", lifespan=lifespan)


# ---------------------------------------------------------------- auth helpers
def current_user(x_user_id: str = Header(default="")) -> dict:
    user = store.get_user(x_user_id) if x_user_id else None
    if not user:
        raise HTTPException(401, "unknown user — create a profile first")
    return user


def optional_user(x_user_id: str = Header(default="")) -> dict | None:
    return store.get_user(x_user_id) if x_user_id else None


def require_admin(x_admin_token: str = Header(default="")) -> None:
    if not ADMIN_TOKEN:
        raise HTTPException(403, "admin endpoints are disabled: set the ADMIN_TOKEN environment variable")
    if not secrets.compare_digest(x_admin_token, ADMIN_TOKEN):
        raise HTTPException(403, "bad admin token")


# ---------------------------------------------------------------- users
class NewUser(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    interests: list[str] = []
    mind_user_id: str | None = None


class Interests(BaseModel):
    interests: list[str]


@app.post("/api/users")
def create_user(body: NewUser):
    if body.mind_user_id and body.mind_user_id not in rec.mind_index:
        raise HTTPException(404, "unknown MIND user")
    return store.create_user(body.name.strip(), body.interests, body.mind_user_id)


@app.get("/api/me")
def me(user: dict = Depends(current_user)):
    return {**user, "history_size": len(rec.history(user, rec.state)), "bookmarks": len(store.bookmarks(user["id"]))}


@app.put("/api/me/interests")
def set_interests(body: Interests, user: dict = Depends(current_user)):
    store.set_interests(user["id"], body.interests)
    return {"ok": True}


@app.get("/api/demo-users")
def demo_users():
    return rec.meta.get("demo_users", [])


# ---------------------------------------------------------------- feeds
@app.get("/api/feed/recommended")
def recommended(k: int = Query(TOP_K, ge=1, le=30), user: dict = Depends(current_user)):
    return rec.recommend(user, k)


@app.get("/api/feed/seen")
def seen(user: dict = Depends(current_user)):
    return rec.seen(user)


@app.get("/api/feed/trending")
def trending():
    return rec.trending()


@app.get("/api/search")
def search(q: str = Query(min_length=1, max_length=100), category: str | None = None):
    return rec.search(q, category)


@app.get("/api/categories")
def categories():
    return rec.categories()


@app.get("/api/explore/{category}")
def explore(category: str, sub: str | None = None):
    return rec.explore(category, sub)


@app.get("/api/articles/{news_id}")
def article(news_id: str, user: dict | None = Depends(optional_user)):
    a = rec.get(news_id, user)
    if not a:
        raise HTTPException(404, "article not found")
    return a


class Click(BaseModel):
    news_id: str
    source: str | None = None


@app.post("/api/clicks")
def click(body: Click, user: dict = Depends(current_user)):
    if body.news_id not in rec.state.index:
        raise HTTPException(404, "article not found")
    store.add_click(user["id"], body.news_id, body.source)
    return {"ok": True}


@app.get("/api/bookmarks")
def bookmarks(user: dict = Depends(current_user)):
    st = rec.state
    return [rec.article(st.index[n], st) for n in store.bookmarks(user["id"]) if n in st.index]


@app.put("/api/bookmarks/{news_id}")
def add_bookmark(news_id: str, user: dict = Depends(current_user)):
    if news_id not in rec.state.index:
        raise HTTPException(404, "article not found")
    store.set_bookmark(user["id"], news_id, True)
    return {"ok": True}


@app.delete("/api/bookmarks/{news_id}")
def remove_bookmark(news_id: str, user: dict = Depends(current_user)):
    store.set_bookmark(user["id"], news_id, False)
    return {"ok": True}


# ---------------------------------------------------------------- model info + admin
@app.get("/api/model")
def model_info():
    m = rec.meta
    return {
        "algorithm": "Alternating Least Squares (implicit) + category boosting",
        "params": m.get("als_params"),
        "boost": rec.boost,
        "reproduced": m.get("metrics"),
        "eval_impressions": m.get("eval_impressions"),
        "report": REPORT_METRICS,
        "dataset": {"users": m.get("n_users"), "articles": m.get("n_items"), "interactions": m.get("n_interactions")},
        "model": rec.model_info,
        "retraining": rec.retraining,
        "retrain_interval_hours": RETRAIN_INTERVAL_HOURS,
        "app": store.stats(),
        "admin_enabled": bool(ADMIN_TOKEN),
    }


@app.post("/api/admin/retrain", dependencies=[Depends(require_admin)])
async def retrain():
    return await asyncio.to_thread(rec.retrain)


class Article(BaseModel):
    category: str = Field(min_length=1, max_length=40)
    subcategory: str = Field(default="", max_length=60)
    title: str = Field(min_length=5, max_length=300)
    abstract: str = Field(min_length=10, max_length=2000)
    url: str = Field(min_length=8, max_length=500, pattern=r"^https?://")
    publisher: str | None = Field(default=None, max_length=80)


@app.post("/api/admin/articles", dependencies=[Depends(require_admin)])
def publish(body: Article):
    return rec.publish(body.category.lower(), body.subcategory.lower(), body.title, body.abstract, body.url, body.publisher)


@app.get("/healthz")
def healthz():
    return {"ok": True, "items": len(rec.state.news)}


# ---------------------------------------------------------------- frontend
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/{path:path}", include_in_schema=False)
def spa(path: str):
    if path.startswith("api/"):
        raise HTTPException(404, "not found")
    return FileResponse(STATIC / "index.html")
