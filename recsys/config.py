"""Model + pipeline configuration, taken directly from the project report (COMP9727)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

# §3.4.1 Rationale for Parameter Selection (ALS, `implicit` library)
ALS_PARAMS = {
    "factors": 64,
    "regularization": 0.01,
    "iterations": 15,
    "alpha": 40.0,
    "random_state": 42,
}

# §4.7 Justification for choosing the top 10 articles to recommend
TOP_K = 10

# Category boosting (§4.8): score = cosine(user, item) + w_cat * P(cat|user) + w_sub * P(subcat|user)
# Values found by train.py's grid search on MIND-small dev (used as-is with --skip-eval; a full run re-tunes them).
DEFAULT_BOOST = {"category": 0.4, "subcategory": 1.6}

MIND_SPLITS = ("MINDsmall_train", "MINDsmall_dev")
MIND_MIRROR = "https://huggingface.co/datasets/huyva/MIND-small/resolve/main"
MIND_FILES = {
    "MINDsmall_train": {"news.tsv": "train/news.tsv", "behaviors.tsv": "train/behaviors.tsv"},
    "MINDsmall_dev": {"news.tsv": "dev/news.tsv", "behaviors.tsv": "dev/behaviors.tsv"},
}
