"""Offline pipeline: evaluate ALS on MIND-small dev, then train the production model and write ./artifacts.

    python train.py              # evaluate + train production model
    python train.py --skip-eval  # production model only (faster, used in Docker builds)

Evaluation protocol (report §4.1–4.4): fit on MINDsmall_train interactions (+ dev click *histories*, which
precede the dev impressions, so no label leakage), score every dev impression log, average AUC / MRR /
nDCG@5 / nDCG@10. Category-boost weights are tuned on a 10% slice of dev impressions and the reported
metrics come from the other 90%.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import sparse

from recsys.config import ALS_PARAMS, ARTIFACTS_DIR, DATA_DIR, DEFAULT_BOOST, MIND_SPLITS
from recsys.data import build_matrix, history_pairs, impression_click_pairs, load_behaviors, load_news, parse_impressions
from recsys.metrics import evaluate
from recsys.model import ALSScorer, train_als

BOOST_GRID = [0.0, 0.1, 0.2, 0.4, 0.8, 1.6]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def encode(series: pd.Series) -> tuple[np.ndarray, list[str]]:
    codes, uniques = pd.factorize(series, sort=True)
    return codes.astype(np.int64), list(uniques)


def run_eval(news, item_index, cat_codes, sub_codes, train_b, dev_b) -> dict:
    pairs = pd.concat([history_pairs(train_b), impression_click_pairs(train_b)[["user_id", "news_id"]], history_pairs(dev_b)])
    users = sorted(pairs["user_id"].unique())
    X = build_matrix(pairs, users, item_index)
    log(f"eval matrix: {X.shape[0]:,} users x {X.shape[1]:,} items, {X.nnz:,} clicks")
    t = time.time()
    scorer = ALSScorer(train_als(X), cat_codes, sub_codes)
    log(f"ALS fit in {time.time() - t:.1f}s")

    # Precompute the three score components per impression so boost weights can be grid-searched cheaply.
    user_cache: dict[str, tuple] = {}
    comps, labels = [], []
    for user_id, hist, cands, y in parse_impressions(dev_b):
        if user_id not in user_cache:
            h = np.array([item_index[n] for n in hist if n in item_index], dtype=np.int64)
            u = scorer.user_vector(h)
            pc, ps = scorer.preferences(h)
            user_cache[user_id] = (u / np.linalg.norm(u) if u is not None else None, pc, ps)
        un, pc, ps = user_cache[user_id]
        c = np.array([item_index.get(n, -1) for n in cands], dtype=np.int64)
        known = c >= 0
        cos, bc, bs = np.zeros(len(c)), np.zeros(len(c)), np.zeros(len(c))
        if un is not None:
            cos[known] = scorer.Yn[c[known]] @ un
        bc[known] = pc[cat_codes[c[known]]]
        bs[known] = ps[sub_codes[c[known]]]
        comps.append((cos, bc, bs))
        labels.append(y)
    log(f"scored {len(labels):,} dev impressions")

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(labels))
    tune, test = perm[: len(perm) // 10], perm[len(perm) // 10 :]

    def metrics_for(idx, wc, ws):
        return evaluate([labels[i] for i in idx], [comps[i][0] + wc * comps[i][1] + ws * comps[i][2] for i in idx])

    best, best_w = -1.0, (0.0, 0.0)
    for wc in BOOST_GRID:
        for ws in BOOST_GRID:
            m = metrics_for(tune, wc, ws)
            if m["auc"] > best:
                best, best_w = m["auc"], (wc, ws)
    log(f"best boost on tuning slice: category={best_w[0]} subcategory={best_w[1]} (AUC {best:.4f})")

    pure = metrics_for(test, 0.0, 0.0)
    boosted = metrics_for(test, *best_w)
    log(f"ALS (no boost)        {pure}")
    log(f"ALS + category boost  {boosted}")
    return {
        "boost": {"category": best_w[0], "subcategory": best_w[1]},
        "metrics": {"als": pure, "als_category_boost": boosted},
        "eval_impressions": int(len(test)),
    }


def pick_demo_users(X: sparse.csr_matrix, users: list[str], news: pd.DataFrame, n: int = 8) -> list[dict]:
    """A handful of real MIND users with readable histories and different dominant interests."""
    counts = np.diff(X.indptr)
    cats = news["category"].to_numpy()
    rng = np.random.default_rng(7)
    out, used = [], set()
    for target in ["sports", "news", "finance", "foodanddrink", "lifestyle", "health", "travel", "tv", "autos", "music", "movies"]:
        for u in rng.permutation(np.where((counts >= 15) & (counts <= 60))[0]):
            items = X.indices[X.indptr[u] : X.indptr[u + 1]]
            vc = pd.Series(cats[items]).value_counts(normalize=True)
            if vc.index[0] == target and vc.iloc[0] >= 0.4 and users[u] not in used:
                out.append({"mind_user_id": users[u], "clicks": int(counts[u]), "top_categories": list(vc.index[:3])})
                used.add(users[u])
                break
        if len(out) == n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    log("loading MIND-small")
    news = load_news([DATA_DIR / s / "news.tsv" for s in MIND_SPLITS])
    train_b = load_behaviors(DATA_DIR / "MINDsmall_train" / "behaviors.tsv")
    dev_b = load_behaviors(DATA_DIR / "MINDsmall_dev" / "behaviors.tsv")
    item_index = {nid: i for i, nid in enumerate(news["news_id"])}
    cat_codes, categories = encode(news["category"])
    sub_codes, subcategories = encode(news["subcategory"])
    log(f"{len(news):,} articles ({int(news['displayable'].sum()):,} displayable), {len(categories)} categories")

    meta: dict = {"als_params": ALS_PARAMS, "boost": dict(DEFAULT_BOOST)}
    if not args.skip_eval:
        meta.update(run_eval(news, item_index, cat_codes, sub_codes, train_b, dev_b))
    else:
        prev = ARTIFACTS_DIR / "meta.json"
        if prev.exists():  # keep previously measured metrics/boost if we're only refreshing the model
            old = json.loads(prev.read_text())
            meta.update({k: old[k] for k in ("boost", "metrics", "eval_impressions") if k in old})

    log("training production model on all MIND-small interactions (train + dev)")
    clicks = pd.concat([impression_click_pairs(train_b), impression_click_pairs(dev_b)])
    pairs = pd.concat([history_pairs(train_b), history_pairs(dev_b), clicks[["user_id", "news_id"]]])
    users = sorted(pairs["user_id"].unique())
    X = build_matrix(pairs, users, item_index)
    t = time.time()
    item_factors = train_als(X)
    log(f"ALS fit on {X.shape[0]:,} x {X.shape[1]:,} ({X.nnz:,} clicks) in {time.time() - t:.1f}s")

    # Trending = most-clicked displayable articles in the final 24h of the logs.
    recent = clicks[clicks["time"] >= clicks["time"].max() - pd.Timedelta(hours=24)]
    displayable = set(news.loc[news["displayable"], "news_id"])
    trend = recent[recent["news_id"].isin(displayable)]["news_id"].value_counts().head(300)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    np.save(ARTIFACTS_DIR / "item_factors.npy", item_factors)
    sparse.save_npz(ARTIFACTS_DIR / "interactions.npz", X)
    (ARTIFACTS_DIR / "users.json").write_text(json.dumps(users))
    news.drop(columns=["displayable"]).to_csv(ARTIFACTS_DIR / "news.csv.gz", index=False)
    meta.update(
        {
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_users": X.shape[0],
            "n_items": X.shape[1],
            "n_interactions": int(X.nnz),
            "trending": [[k, int(v)] for k, v in trend.items()],
            "demo_users": pick_demo_users(X, users, news),
        }
    )
    (ARTIFACTS_DIR / "meta.json").write_text(json.dumps(meta, indent=1))
    log(f"artifacts written to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
