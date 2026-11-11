"""Runtime recommender: catalog, ALS scoring, feeds, and periodic retraining."""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from recsys.config import ALS_PARAMS, DEFAULT_BOOST, TOP_K
from recsys.model import ALSScorer, train_als

from .store import Store

log = logging.getLogger("recommender")

CATEGORY_LABELS = {
    "news": "News", "sports": "Sports", "finance": "Finance", "foodanddrink": "Food & Drink",
    "lifestyle": "Lifestyle", "travel": "Travel", "video": "Video", "weather": "Weather", "health": "Health",
    "autos": "Autos", "tv": "TV", "music": "Music", "movies": "Movies", "entertainment": "Entertainment",
    "kids": "Kids", "middleeast": "Middle East", "northamerica": "North America", "games": "Games",
}
ACRONYMS = {"nfl", "nba", "mlb", "nhl", "ncaa", "us", "tv", "uk", "ufc", "mls", "pga", "wnba", "f1", "ncaaf", "ncaab", "diy", "mma", "cma", "lgbtq", "bbq", "wwe", "nascar", "lpga"}
WORDS = {"icehockey": "Ice Hockey", "newsus": "US News", "newsworld": "World", "newspolitics": "Politics"}

# Report results table (§4.4), shown alongside the metrics reproduced by train.py.
REPORT_METRICS = {
    "LightGBM": {"auc": 0.6037, "mrr": 0.3020, "ndcg@5": 0.2872, "ndcg@10": 0.3522},
    "ALS": {"auc": 0.6148, "mrr": 0.3387, "ndcg@5": 0.3220, "ndcg@10": 0.3820},
    "GraphSAGE": {"auc": 0.5655, "mrr": 0.2583, "ndcg@5": 0.2369, "ndcg@10": 0.3110},
}


def label_category(c: str) -> str:
    return CATEGORY_LABELS.get(c, c.replace("_", " ").title())


def label_subcategory(sub: str, cat: str) -> str:
    s = sub[len(cat):] if sub.startswith(cat) and len(sub) > len(cat) else sub
    words = re.split(r"[_\-]+", s)
    return " ".join(WORDS.get(w) or (w.upper() if w in ACRONYMS else w.capitalize()) for w in words if w) or label_category(cat)


@dataclass
class State:
    """Immutable snapshot swapped atomically on retrain/publish, so readers never see a half-built model."""
    news: pd.DataFrame
    index: dict[str, int]
    categories: list[str]
    subcategories: list[str]
    scorer: ALSScorer
    popularity: np.ndarray
    displayable: np.ndarray
    published: np.ndarray
    pool: np.ndarray
    title_lc: pd.Series
    abstract_lc: pd.Series


class Recommender:
    def __init__(
        self, artifacts_dir: Path, state_dir: Path, store: Store,
        min_clicks: int = 3, max_per_subcategory: int = 3, max_per_category: int = 6,
    ):
        self.store = store
        self.min_clicks = min_clicks
        self.max_per_sub = max_per_subcategory
        self.max_per_cat = max_per_category
        self.meta = json.loads((artifacts_dir / "meta.json").read_text())
        self.boost = self.meta.get("boost", DEFAULT_BOOST)
        self.base_news = pd.read_csv(artifacts_dir / "news.csv.gz", dtype=str, keep_default_na=False)
        self.base_X = sparse.load_npz(artifacts_dir / "interactions.npz").tocsr()
        self.mind_users: list[str] = json.loads((artifacts_dir / "users.json").read_text())
        self.mind_index = {u: i for i, u in enumerate(self.mind_users)}
        self.trending_base = {k: v for k, v in self.meta.get("trending", [])}

        self.model_dir = state_dir / "model"
        factors = np.load(artifacts_dir / "item_factors.npy")
        self.model_info = {"trained_at": self.meta.get("trained_at"), "source": "MIND-small (build)", "app_clicks": 0}
        if (self.model_dir / "item_factors.npy").exists():
            factors = np.load(self.model_dir / "item_factors.npy")
            self.model_info = json.loads((self.model_dir / "model.json").read_text())

        self._lock = threading.Lock()
        self._retrain_lock = threading.Lock()
        self.retraining = False
        self.state = self._build_state(factors)

    # ------------------------------------------------------------------ catalog / model state
    def _build_state(self, factors: np.ndarray) -> State:
        pub = pd.DataFrame(self.store.articles())
        news = self.base_news
        if len(pub):
            pub = pub[["news_id", "category", "subcategory", "title", "abstract", "url"]]
            news = pd.concat([news, pub], ignore_index=True)
        n = len(news)
        factors = factors[:n]
        if factors.shape[0] < n:  # newly published articles have no factors until the next retrain
            factors = np.vstack([factors, np.zeros((n - factors.shape[0], factors.shape[1]), dtype=factors.dtype)])
        cat_codes, categories = pd.factorize(news["category"], sort=True)
        sub_codes, subcategories = pd.factorize(news["subcategory"], sort=True)
        scorer = ALSScorer(factors, cat_codes, sub_codes, ALS_PARAMS["alpha"], ALS_PARAMS["regularization"], self.boost)

        pop = np.zeros(n)
        pop[: self.base_X.shape[1]] = np.asarray(self.base_X.sum(axis=0)).ravel()
        index = {nid: i for i, nid in enumerate(news["news_id"])}
        for _, nid in self.store.all_clicks():
            if nid in index:
                pop[index[nid]] += 1
        displayable = ((news["title"] != "") & (news["abstract"] != "") & (news["category"] != "")).to_numpy()
        published = np.zeros(n, dtype=bool)
        published[len(self.base_news):] = True
        # Candidate pool: ALS can only score items it has factors for (§3.4.4 cold-start limitation); require a
        # few clicks so single-click noise doesn't dominate. Publisher articles join via category boosting alone.
        pool = displayable & ((scorer.has_factors & (pop >= self.min_clicks)) | published)
        return State(
            news=news, index=index, categories=list(categories), subcategories=list(subcategories), scorer=scorer,
            popularity=pop, displayable=displayable, published=published, pool=pool,
            title_lc=news["title"].str.lower(), abstract_lc=news["abstract"].str.lower(),
        )

    def article(self, i: int, st: State | None = None, **extra) -> dict:
        st = st or self.state
        r = st.news.iloc[i]
        return {
            "id": r["news_id"], "title": r["title"], "abstract": r["abstract"], "url": r["url"],
            "category": r["category"], "category_label": label_category(r["category"]),
            "subcategory": r["subcategory"], "subcategory_label": label_subcategory(r["subcategory"], r["category"]),
            "is_new": bool(st.published[i]), **extra,
        }

    # ------------------------------------------------------------------ users
    def history(self, user: dict, st: State) -> list[int]:
        """Clicked item indices, oldest first: the user's MIND history (if linked) then in-app clicks."""
        items: list[int] = []
        mu = user.get("mind_user_id")
        if mu in self.mind_index:
            r = self.mind_index[mu]
            items.extend(int(x) for x in self.base_X.indices[self.base_X.indptr[r] : self.base_X.indptr[r + 1]])
        items.extend(st.index[n] for n, _ in self.store.clicks(user["id"]) if n in st.index)
        return items

    def interest_codes(self, user: dict, st: State) -> np.ndarray:
        return np.array([st.categories.index(c) for c in user.get("interests", []) if c in st.categories], dtype=np.int64)

    # ------------------------------------------------------------------ feeds
    def recommend(self, user: dict, k: int = TOP_K) -> list[dict]:
        st = self.state
        hist = np.array(self.history(user, st), dtype=np.int64)
        interests = self.interest_codes(user, st)
        if len(hist) == 0 and len(interests) == 0:
            return [dict(a, reason="Trending now — open a few articles to personalise your feed") for a in self.trending(k)]

        cand = np.where(st.pool)[0]
        if len(hist):
            cand = cand[~np.isin(cand, hist)]
        u = st.scorer.user_vector(hist)
        scores = st.scorer.score(hist, cand, interests, user_vec=u)
        scores += 1e-3 * np.log1p(st.popularity[cand]) / np.log1p(st.popularity.max())  # tie-break for cold users

        # Top-k with per-subcategory and per-category caps so the list stays diverse (report §4.7: relevance
        # vs. diversity). Before any clicks, the category cap spreads the feed evenly over declared interests.
        max_cat = self.max_per_cat if u is not None else max(1, -(-k // max(len(interests), 1)))
        order = np.argsort(-scores)
        picked, per_sub, per_cat = [], {}, {}
        for relax in (False, True):  # second pass fills any remaining slots if the caps were too strict
            for j in order:
                i = int(cand[j])
                s, c = int(st.scorer.sub[i]), int(st.scorer.cat[i])
                if any(p[0] == i for p in picked):
                    continue
                if not relax and (per_sub.get(s, 0) >= self.max_per_sub or per_cat.get(c, 0) >= max_cat):
                    continue
                per_sub[s] = per_sub.get(s, 0) + 1
                per_cat[c] = per_cat.get(c, 0) + 1
                picked.append((i, float(scores[j])))
                if len(picked) == k:
                    break
            if len(picked) == k:
                break

        # Fresh slot: ALS can't score a just-published article until it has clicks (§3.4.4), so the best
        # unseen publisher article in a category this user reads/follows gets one guaranteed position.
        fresh = np.where(st.published[cand])[0]
        if len(fresh) and not any(st.published[i] for i, _ in picked):
            pc, _ = st.scorer.preferences(hist, interests)
            j = fresh[np.argmax(scores[fresh])]
            if pc[st.scorer.cat[cand[j]]] > 0:
                picked.insert(min(3, len(picked)), (int(cand[j]), float(scores[j])))
                picked = picked[:k]

        out = []
        for i, score in picked:
            b = st.scorer.because(hist, i) if u is not None else None
            if st.published[i] and not st.scorer.has_factors[i]:
                reason = f"Just published in {label_category(st.news.iloc[i]['category'])}, which you read"
            elif b is not None:
                reason = f"Because you read “{st.news.iloc[b]['title']}”"
            elif u is not None and st.scorer.has_factors[i]:
                reason = "Readers with similar reading habits also clicked this"
            else:
                reason = f"Popular in {label_category(st.news.iloc[i]['category'])}, a topic you follow"
            out.append(self.article(i, st, score=round(score, 4), reason=reason))
        self.store.log_impressions(user["id"], [a["id"] for a in out])
        return out

    def seen(self, user: dict, limit: int = 60) -> list[dict]:
        st = self.state
        hist = self.history(user, st)[::-1]
        out, seen = [], set()
        for i in hist:
            if i not in seen and st.displayable[i]:
                seen.add(i)
                out.append(self.article(i, st))
            if len(out) == limit:
                break
        return out

    def trending(self, limit: int = 30) -> list[dict]:
        st = self.state
        counts = dict(self.trending_base)
        for nid, c in self.store.recent_click_counts(time.time() - 86400).items():
            counts[nid] = counts.get(nid, 0) + c
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        out = []
        for nid, c in ranked:
            i = st.index.get(nid)
            if i is not None and st.displayable[i]:
                out.append(self.article(i, st, clicks_24h=c))
            if len(out) == limit:
                break
        return out

    def search(self, q: str, category: str | None = None, limit: int = 30) -> list[dict]:
        st = self.state
        tokens = [t for t in re.findall(r"\w+", q.lower()) if len(t) > 1]
        if not tokens:
            return []
        mask = st.displayable.copy()
        if category:
            mask &= (st.news["category"] == category).to_numpy()
        title_hits = np.zeros(len(mask))
        for t in tokens:
            in_title = st.title_lc.str.contains(t, regex=False).to_numpy()
            mask &= in_title | st.abstract_lc.str.contains(t, regex=False).to_numpy()
            title_hits += in_title
        idx = np.where(mask)[0]
        order = np.lexsort((-st.popularity[idx], -title_hits[idx]))
        return [self.article(int(i), st) for i in idx[order][:limit]]

    def explore(self, category: str, subcategory: str | None = None, limit: int = 40) -> dict:
        st = self.state
        mask = st.displayable & (st.news["category"] == category).to_numpy()
        subs = st.news.loc[mask, "subcategory"].value_counts()
        if subcategory:
            mask &= (st.news["subcategory"] == subcategory).to_numpy()
        idx = np.where(mask)[0]
        order = np.lexsort((-st.popularity[idx], ~st.published[idx]))  # publisher articles first, then popular
        return {
            "subcategories": [{"id": s, "label": label_subcategory(s, category), "count": int(c)} for s, c in subs.head(12).items()],
            "articles": [self.article(int(i), st) for i in idx[order][:limit]],
        }

    def categories(self) -> list[dict]:
        st = self.state
        vc = st.news.loc[st.displayable, "category"].value_counts()
        return [{"id": c, "label": label_category(c), "count": int(n)} for c, n in vc.items() if n >= 20]

    def get(self, news_id: str, user: dict | None = None) -> dict | None:
        st = self.state
        i = st.index.get(news_id)
        if i is None:
            return None
        sc = st.scorer
        if sc.has_factors[i]:  # "More like this": nearest same-category neighbours in the ALS item space
            cand = np.where(st.pool & (sc.cat == sc.cat[i]) & sc.has_factors)[0]
            sims = sc.Yn[cand] @ sc.Yn[i]
            related = [int(cand[j]) for j in np.argsort(-sims)[:6] if cand[j] != i][:5]
        else:
            same = np.where(st.pool & (sc.sub == sc.sub[i]))[0]
            related = [int(j) for j in same[np.argsort(-st.popularity[same])][:6] if j != i][:5]
        bookmarked = user is not None and news_id in self.store.bookmarks(user["id"])
        return self.article(i, st, bookmarked=bookmarked, related=[self.article(j, st) for j in related])

    # ------------------------------------------------------------------ publishing + retraining
    def publish(self, category: str, subcategory: str, title: str, abstract: str, url: str, publisher: str | None) -> dict:
        nid = "P" + uuid.uuid4().hex[:10].upper()
        self.store.add_article(
            {"news_id": nid, "category": category, "subcategory": subcategory or category, "title": title,
             "abstract": abstract, "url": url, "publisher": publisher}
        )
        with self._lock:
            self.state = self._build_state(self.state.scorer.Y.astype(np.float32))
        return self.article(self.state.index[nid])

    def retrain(self) -> dict:
        """§4.10.1: retrain ALS on MIND interactions + every click logged by the app, then hot-swap the model."""
        if not self._retrain_lock.acquire(blocking=False):
            return {"status": "already_running"}
        self.retraining = True
        try:
            t0 = time.time()
            st = self.state
            n_items = len(st.news)
            base = self.base_X
            X = sparse.csr_matrix((base.data, base.indices, base.indptr), shape=(base.shape[0], n_items))
            new_rows: dict[str, int] = {}
            rows, cols = [], []
            linked = {u["id"]: u["mind_user_id"] for u in self.store.all_users()}
            clicks = self.store.all_clicks()
            for uid, nid in clicks:
                if nid not in st.index:
                    continue
                mu = linked.get(uid)
                if mu in self.mind_index:
                    r = self.mind_index[mu]
                else:
                    r = new_rows.setdefault(uid, base.shape[0] + len(new_rows))
                rows.append(r)
                cols.append(st.index[nid])
            extra = sparse.csr_matrix(
                (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(base.shape[0] + len(new_rows), n_items)
            )
            X = sparse.vstack([X, sparse.csr_matrix((len(new_rows), n_items), dtype=np.float32)]).tocsr() + extra
            X.data[:] = 1.0  # binary interactions, as in §2.5.2
            factors = train_als(X)
            info = {
                "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source": "MIND-small + in-app clicks (retrain)",
                "app_clicks": len(clicks), "n_users": X.shape[0], "n_items": n_items,
                "seconds": round(time.time() - t0, 1),
            }
            self.model_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.model_dir / "item_factors.npy", factors)
            (self.model_dir / "model.json").write_text(json.dumps(info))
            with self._lock:
                self.state = self._build_state(factors)
                self.model_info = info
            log.info("retrained ALS in %.1fs on %d users / %d clicks", info["seconds"], X.shape[0], X.nnz)
            return {"status": "ok", **info}
        finally:
            self.retraining = False
            self._retrain_lock.release()
