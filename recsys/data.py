"""Loading + preprocessing of MIND-small (report §2.4–2.5)."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

NEWS_COLS = ["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
BEHAVIOR_COLS = ["impression_id", "user_id", "time", "history", "impressions"]


def load_news(paths: list[Path]) -> pd.DataFrame:
    """Union of news.tsv files.

    Every article stays in the ALS item index (a click on an article without an abstract is still a valid
    collaborative signal). §2.5.1's null removal is applied at serving time instead: only articles with a
    complete title + abstract + category are ever shown (see `displayable`).
    """
    frames = [
        pd.read_csv(p, sep="\t", header=None, names=NEWS_COLS, quoting=csv.QUOTE_NONE, dtype=str)
        for p in paths
    ]
    news = pd.concat(frames, ignore_index=True).drop_duplicates("news_id")
    for col in ("category", "subcategory", "title", "abstract", "url"):
        news[col] = news[col].fillna("").str.strip()
    news["displayable"] = (news["title"] != "") & (news["abstract"] != "") & (news["category"] != "")
    return news[["news_id", "category", "subcategory", "title", "abstract", "url", "displayable"]].reset_index(drop=True)


def load_behaviors(path: Path) -> pd.DataFrame:
    b = pd.read_csv(path, sep="\t", header=None, names=BEHAVIOR_COLS, dtype=str)
    b["history"] = b["history"].fillna("")
    b["time"] = pd.to_datetime(b["time"], format="%m/%d/%Y %I:%M:%S %p")
    return b


def history_pairs(behaviors: pd.DataFrame) -> pd.DataFrame:
    """(user_id, news_id) clicks from the History column. History is constant per user, so dedupe first."""
    h = behaviors.drop_duplicates("user_id")[["user_id", "history"]]
    h = h.assign(news_id=h["history"].str.split()).explode("news_id").dropna(subset=["news_id"])
    return h[["user_id", "news_id"]]


def impression_click_pairs(behaviors: pd.DataFrame) -> pd.DataFrame:
    """(user_id, news_id, time) for impressions marked as clicked (NewsID-1)."""
    imp = behaviors.assign(item=behaviors["impressions"].str.split()).explode("item")
    imp = imp[imp["item"].str.endswith("-1")]
    return pd.DataFrame({"user_id": imp["user_id"].values, "news_id": imp["item"].str[:-2].values, "time": imp["time"].values})


def parse_impressions(behaviors: pd.DataFrame):
    """Yield (user_id, history list, candidate ids, labels) per impression log (for evaluation)."""
    for user_id, history, impressions in zip(behaviors["user_id"], behaviors["history"], behaviors["impressions"]):
        items = impressions.split()
        yield user_id, history.split(), [x[:-2] for x in items], np.array([int(x[-1]) for x in items], dtype=np.int8)


def build_matrix(pairs: pd.DataFrame, user_ids: list[str], item_index: dict[str, int]) -> sparse.csr_matrix:
    """Binary user x item CSR matrix (§2.5.2: 1 if the user clicked, stored in Compressed Sparse Row format)."""
    user_index = {u: i for i, u in enumerate(user_ids)}
    p = pairs[pairs["news_id"].isin(item_index.keys()) & pairs["user_id"].isin(user_index.keys())]
    p = p.drop_duplicates(["user_id", "news_id"])
    rows = p["user_id"].map(user_index).to_numpy()
    cols = p["news_id"].map(item_index).to_numpy()
    data = np.ones(len(p), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(item_index)))
