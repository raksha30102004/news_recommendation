---
title: News Recommender
emoji: 📰
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
short_description: ALS news recommender trained on Microsoft MIND
---

# News — an ALS news recommender you can deploy

A working version of the system proposed in *Recommender Systems: Project Implementation* (COMP9727, UNSW). It uses the model that won in that report, **Alternating Least Squares with category boosting**. The model is trained on Microsoft's MIND-small and served as a mobile-first web app modelled on the report's UI mockup (Figure 1).

## What's implemented (mapped to the report)

| Report | Implementation |
|---|---|
| §2.5.2 Binary user×article click matrix in CSR format | `recsys/data.py → build_matrix`, built from MIND click histories and impression clicks |
| §3.4.1 `factors=64, regularization=0.01, iterations=15, alpha=40, random_state=42` | `recsys/config.py → ALS_PARAMS`, trained with `implicit` |
| §3.4.3 Cosine similarity between the user and article latent vectors | `recsys/model.py → ALSScorer.score` |
| §4.8 Category boosting | `+ w_cat·P(category\|user) + w_sub·P(subcategory\|user)`. The weights are tuned on a 10 % slice of dev impressions |
| §4.7 Recommend the top 10 | `TOP_K = 10`, with at most 3 per subcategory and 6 per category for diversity |
| §4.4 AUC / MRR / nDCG@5 / nDCG@10 | `recsys/metrics.py`, matching the official MIND evaluation. Results are shown on the Profile page |
| §4.9.1 Precomputed embeddings for near-real-time serving | Item factors are precomputed. A user's vector is re-solved from their click history on each request (the exact ALS user step, about 1 ms), so a click changes the feed immediately |
| §1.2 Clicks are added to impressions and the model is retrained | Every served article is logged in `impressions`. Opening one marks it clicked and adds it to `clicks` |
| §4.10.1 Daily retraining | A background job retrains ALS on MIND plus all in-app clicks every `RETRAIN_INTERVAL_HOURS` (default 24), then hot-swaps the model |
| §1 Independent publishers can post articles | `POST /api/admin/articles`. New articles can't be scored by ALS until they get clicks (§3.4.4), so the best match gets one "fresh" slot in readers' feeds |
| Figure 1 mockup | Recommended / Seen Before / Trending tabs, search, and a Home / Explore / Bookmark / Profile tab bar |

### Reproduced results (MIND-small dev, 65,837 held-out impressions)

| Model | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| **This deployment: ALS + category boost** | **0.6200** | 0.3017 | **0.3280** | **0.3873** |
| ALS without boost | 0.5356 | 0.2465 | 0.2576 | 0.3180 |
| Report: ALS | 0.6148 | 0.3387 | 0.3220 | 0.3820 |
| Report: LightGBM | 0.6037 | 0.3020 | 0.2872 | 0.3522 |
| Report: GraphSAGE | 0.5655 | 0.2583 | 0.2369 | 0.3110 |

Protocol: ALS is fitted on the train split's interactions plus the dev click *histories*. Those histories come before the dev impressions, so no labels leak. Every dev impression is scored, and the boost weights are picked on a separate 10 % of impressions.

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py      # MIND-small train + dev (~210 MB)
python train.py                      # evaluate, train the production model, write ./artifacts (~2 min, ~1.3 GB RAM)
ADMIN_TOKEN=change-me uvicorn app.main:app --reload
```

Open http://localhost:8000. You can either create a profile and pick some topics, or browse as one of the real MIND readers listed on the start page.

## Deploy

The Docker image downloads MIND-small, trains the model at build time, and ships only the API, the frontend and about 32 MB of artifacts.

```bash
docker build -t news-recommender .
docker run -p 8000:8000 -v news-state:/data -e ADMIN_TOKEN=change-me news-recommender
```

- **Render:** push this folder to a Git repo, then New → Blueprint. `render.yaml` sets up the Docker service, a persistent disk at `/data`, a generated `ADMIN_TOKEN` and a health check.
- **Fly.io, Railway, Cloud Run, etc.:** use the Dockerfile as-is. The app listens on `$PORT`. Mount a volume at `/data` so users, clicks and retrained models survive redeploys. Without a volume the app still works, but that state resets on every deploy.
- **Resources:** serving uses about 300 MB of RAM and a retrain peaks around 400 MB, so 512 MB works and 1 GB is comfortable. The build needs about 1.5 GB. Use `--build-arg TRAIN_ARGS=--skip-eval` to skip evaluation and build faster.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `STATE_DIR` | `./state` (`/data` in Docker) | SQLite database and retrained models |
| `ADMIN_TOKEN` | *(unset)* | Enables the admin endpoints below. If unset they return 403 |
| `RETRAIN_INTERVAL_HOURS` | `24` | How often to retrain in the background. `0` turns it off |
| `MIND_MIRROR` | Hugging Face mirror | Base URL for the MIND-small TSV files |

### Admin API

```bash
# Retrain now, on MIND plus every in-app click
curl -X POST https://<host>/api/admin/retrain -H "X-Admin-Token: $ADMIN_TOKEN"

# Publish an article (shown in Explore and search immediately; can appear in feeds via the fresh slot)
curl -X POST https://<host>/api/admin/articles -H "X-Admin-Token: $ADMIN_TOKEN" -H "Content-Type: application/json" \
  -d '{"category":"travel","subcategory":"traveltips","title":"…","abstract":"…","url":"https://…","publisher":"…"}'
```

## Project layout

```
recsys/         model code shared by training and serving (config, data loading, ALS + boosting, metrics)
train.py        offline pipeline: evaluation, then the production model, written to ./artifacts
scripts/        dataset download
app/main.py     FastAPI routes, auth headers, background retrain loop
app/service.py  runtime recommender: catalog, feeds, search, explanations, publishing, retraining
app/store.py    SQLite: users, clicks, impressions, bookmarks, published articles
app/static/     single-page frontend (no build step)
```

## Notes and limitations

- **Identity is anonymous.** Each browser gets a random profile ID in `localStorage`, sent as `X-User-Id`. Add real auth before storing anything sensitive.
- **Cold start (§3.4.4).** Brand-new users get popular articles from the topics they pick, spread evenly across those topics. Their first click switches them to ALS. Brand-new articles rely on category boosting and the fresh slot until a retrain gives them latent factors.
- **Clicks on articles with no abstract are kept.** §2.5.1 drops these articles. Here their clicks still train ALS, because they're valid collaborative signal, but the articles are never displayed.
- **Original links are offline.** MIND's MSN article URLs from 2019 no longer resolve, so dataset articles link to a news search for their headline. Published articles link to their real URL.
- **Data source.** Microsoft's original download bucket no longer allows public access. `scripts/download_data.py` uses a Hugging Face mirror whose files match the official MIND-small exactly: 51,282 train articles and 156,965 train impression logs.
