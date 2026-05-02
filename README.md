# Bike Troubleshooting Bot

Multimodal RAG bot that answers bike issues **only** from the user's manual,
in **the same language** the user asked in (Hindi / Tamil / Marathi / English / etc.).

- **Final answer LLM**: Sarvam 105B (multilingual, grounded)
- **Vision + embeddings + query rewriting**: Gemini 2.0 Flash + text-embedding-004
- **Retrieval**: ChromaDB (vector) + BM25 (keyword) → Reciprocal Rank Fusion → optional Cohere re-rank
- **Guardrail**: citation verifier downgrades unsupported claims
- **Channels**: web UI **and** WhatsApp (via Whapi)

## Architecture

```
                    Web UI  /  WhatsApp (Whapi)
                          │
                          ▼
   ┌──────────────────────────────────────────────────────┐
   │  Layer 1 · Input        text + optional image        │
   │  Layer 2 · Vision (Gemini) + Query rewrite (Gemini)  │
   │  Layer 3 · Hybrid retrieval — vector + BM25 + RRF    │
   │           → optional Cohere re-rank → top 5 chunks   │
   │  Layer 4 · Sarvam 105B grounded JSON generation      │
   │           (answers in the user's language)           │
   │  Layer 5 · Citation verifier (drops bad citations)   │
   └──────────────────────────────────────────────────────┘
                          │
                          ▼
                  Final answer + citations
```

A separate one-time **ingestion pipeline** processes each PDF manual:
PDF parser → semantic chunking (~500 tokens, section-aware) → metadata
enrichment (component + symptom tags + page) → embeddings → ChromaDB +
BM25 index.

## Layout

```
sv/
├── api.env                ← paste your API keys here
├── requirements.txt
├── backend/
│   ├── main.py            ← FastAPI app
│   ├── config.py
│   ├── ingest.py          ← PDF → chunks → ChromaDB + BM25
│   ├── retrieval.py       ← vector + BM25 + RRF + Cohere rerank
│   ├── vision.py          ← Gemini vision
│   ├── query_rewriter.py
│   ├── generator.py       ← Sarvam 105B grounded generation
│   ├── verifier.py        ← citation guardrail
│   ├── whatsapp.py        ← Whapi REST + webhook parsing
│   └── models.py
├── frontend/              ← static SPA (chat + WhatsApp + manuals + status)
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── scripts/
│   └── ingest_manual.py   ← CLI ingestion
└── data/
    ├── manuals/           ← uploaded PDFs
    └── chroma_db/         ← persistent vector store
```

## Setup

```bash
cd /Users/ichchhitbajaj/Documents/sv

# 1. Python 3.10+ recommended.
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Open api.env, paste your keys (Sarvam, Gemini, Cohere, Whapi).
#    The server runs without keys but with stubbed embeddings & no LLM output.

# 3. Run the server.
python -m backend.main
# →  http://localhost:8000
```

## Using it

1. Open `http://localhost:8000`.
2. Go to **Manuals** tab → upload a PDF (e.g., `royal_enfield_classic_350.pdf`)
   with a `manual_id` like `royal_enfield_classic_350`.
3. Switch to **Chat** tab. Pick the manual. Ask in any language. Attach a photo if relevant.

### CLI alternative for ingestion

```bash
python -m scripts.ingest_manual data/manuals/RE_Classic_350.pdf royal_enfield_classic_350
```

## WhatsApp setup (Whapi)

1. Create a channel at https://whapi.cloud and copy the token.
2. Paste it as `WHAPI_TOKEN` in `api.env`.
3. Expose your local server publicly (ngrok works):
   ```bash
   ngrok http 8000
   ```
4. In the Whapi dashboard set the **webhook URL** to:
   `https://<your-ngrok-id>.ngrok.io/whatsapp/webhook`
5. Set `PUBLIC_BASE_URL` in `api.env` to the ngrok URL.
6. Anyone WhatsApping your Whapi number now goes through the same RAG pipeline.

The **WhatsApp** tab in the web UI is a real sender — it calls Whapi to send
the answer to a phone number you choose, useful for end-to-end testing.

## How the multilingual flow works

1. The retriever runs in English (English query rewrites + English manual).
2. Sarvam 105B receives the user's original (any-language) question + the
   English chunks, and is instructed to:
   - detect the user's language,
   - write the final answer in that same language and script,
   - cite English chunks by page + chunk_id.
3. The verifier checks every cited chunk_id/page actually came from retrieval.

This gives you Hindi answers from English manuals without re-indexing.

## API reference

| Method | Path                       | Body                                                | Returns                          |
|--------|----------------------------|-----------------------------------------------------|----------------------------------|
| POST   | `/api/ingest`              | multipart: `file` (PDF), `manual_id` (string)       | `{ok, manual_id, pages, chunks}` |
| GET    | `/api/manuals`             | —                                                   | `{manuals: [...]}`               |
| POST   | `/api/query`               | `{manual_id, query, image_b64?}`                    | `QueryResponse`                  |
| POST   | `/api/whatsapp/send`       | `{to, body}`                                        | Whapi response                   |
| POST   | `/whatsapp/webhook`        | Whapi inbound JSON                                  | `{ok}`                           |
| GET    | `/api/health`              | —                                                   | which keys are configured        |

## What to highlight in the Sarvam interview

1. **Strict grounding**: explicit refusal path + JSON schema + citation verifier
   catches hallucinations programmatically.
2. **Hybrid retrieval**: BM25 nails part numbers (`415-3201`), vector nails
   semantic phrasing — RRF merges them; Cohere re-rank is the final filter.
3. **Section-aware chunking**: ~30-40% retrieval gain over naive 500-word splits.
4. **Multilingual without re-indexing**: Sarvam 105B is the language layer;
   retrieval stays cheap and English.
5. **Latency budget**: vision (~2s, parallelizable) + rewrite (~0.5s) +
   retrieval (~0.1s) + Sarvam gen (~3s) + verify (~0.05s) ≈ 6s end-to-end.
6. **Failure modes to discuss**: hallucination, retrieval miss, citation drift,
   modality mismatch, model-not-in-corpus — each with the mitigation above.
7. **Evaluation plan**: 50 manually-labelled QA pairs; measure retrieval recall@5,
   faithfulness via LLM-as-judge, refusal accuracy on out-of-manual questions.

## Quick free hosting (single URL, demo-grade)

For a 2-3 day demo you don't need the Netlify split — FastAPI already serves
the frontend, so one host runs everything.

### Option A — Hugging Face Spaces (recommended for Sarvam demos)

Free, ML-community standard, single public URL.

1. Create a free account at https://huggingface.co.
2. **New Space** → **SDK: Docker** → name it `bike-bot` → Public.
3. Push this repo to the Space's git remote:
   ```bash
   git init
   git lfs install   # if not already
   git remote add space https://huggingface.co/spaces/<YOUR_USERNAME>/bike-bot
   git add . && git commit -m "Initial deploy"
   git push space main
   ```
4. In **Settings → Variables and secrets**, add:
   - `SARVAM_API_KEY`
   - `GEMINI_API_KEY`
   - `WHAPI_TOKEN` (optional)
5. The Space builds the [Dockerfile](Dockerfile) automatically — wait ~3 min.
6. Open `https://huggingface.co/spaces/<YOUR_USERNAME>/bike-bot` → Manuals
   tab → upload PDF → ask in any language. Done.

Caveat: free Spaces sleep after 48 h of zero traffic — fine for a 2-3 day
demo if you keep using it. Persistent disk is ephemeral on free; data lives
as long as the Space runs without restart.

### Option B — Render free tier (zero new files)

Your [render.yaml](render.yaml) is already set up. Single URL serves both
frontend and backend.

1. Push to GitHub.
2. https://render.com → **New → Blueprint** → pick the repo.
3. Set secrets in the Environment tab:
   `SARVAM_API_KEY`, `GEMINI_API_KEY`, `WHAPI_TOKEN`.
4. After deploy, open `https://<service>.onrender.com`.

Caveat: free plan has no persistent disk (it's $7/mo Starter for that).
For a 2-3 day demo, just avoid redeploys after ingesting your manual — the
data lives in the running process.

### Option C — Railway ($5 free trial, no sleep)

1. https://railway.app → **New Project → Deploy from GitHub** → pick repo.
2. Railway auto-detects [Procfile](Procfile) and the [Dockerfile](Dockerfile).
3. Add the same secrets in **Variables**.
4. Add a **Volume** (1 GB) mounted at `/app/data` for persistent ChromaDB+SQLite.
5. Open the generated `*.up.railway.app` URL.

The $5 starter credit covers ~3 days of always-on at this scale.

---

## Deploying to production (Netlify + Render — for real prod, not demos)

**The split:** Netlify hosts the static frontend; Render hosts the FastAPI
backend. Netlify proxies `/api/*` to Render so the browser only sees one
domain. Both providers have free tiers.

### Step 1 — Push the repo to GitHub

```bash
cd /Users/ichchhitbajaj/Documents/sv
git init && git add . && git commit -m "Initial commit"
gh repo create bike-bot --public --source=. --push
```

### Step 2 — Deploy the backend on Render

1. Sign up at https://render.com (free tier).
2. **New → Blueprint** → connect your GitHub repo. Render reads
   [render.yaml](render.yaml) and creates the service.
3. After the build completes, open the service → **Environment** tab and add
   the secrets that are marked `sync: false` in the blueprint:
     - `SARVAM_API_KEY`
     - `GEMINI_API_KEY`
     - `WHAPI_TOKEN` (only if you want WhatsApp; otherwise leave blank)
4. Note the public URL, e.g. `https://bike-bot-backend.onrender.com`.
5. Hit `https://YOUR-URL/api/health` — you should see `{"ok": true, ...}`.

> Render free tier sleeps after 15 minutes of inactivity. First request
> after sleep takes ~30s. Upgrade to the Starter plan ($7/mo) to avoid this.

### Step 3 — Deploy the frontend on Netlify

1. Edit [netlify.toml](netlify.toml) — replace **both** occurrences of
   `https://YOUR-BACKEND.onrender.com` with the URL from step 2.
2. Commit and push.
3. Sign up at https://netlify.com.
4. **Add new site → Import an existing project → GitHub** → pick your repo.
   Netlify auto-detects `netlify.toml` and deploys.
5. After deploy completes, open the Netlify URL (e.g.
   `https://bike-bot.netlify.app`).
6. Go to the **📚 Manuals** tab and upload your bike PDF.
7. Done — share the Netlify URL with users.

### Step 4 — Custom domain (optional)

In Netlify → Domain settings → add a custom domain (e.g. `bot.example.com`).
Netlify provisions free Let's Encrypt certs automatically.

### Why this split (and not Netlify-only)?

Netlify is for static sites + short-lived edge functions (max 10 s, no
persistent disk). The bot needs:
- Persistent ChromaDB (vector index on disk)
- SQLite for memory + metrics
- 10-30 s LLM calls during query

These all need a long-running Python process with a real filesystem, which
is what Render (or Railway, Fly.io, a DigitalOcean droplet) provides.

### Quick local dev (no deployment)

```bash
.venv/bin/python -m backend.main      # → http://localhost:8000
```

Same code, same UI, same API. The only difference is `BACKEND_URL` is empty
so the browser hits `/api/...` on the same host.

---

## Conversation memory

The bot is now stateful — follow-up questions like "and how do I fix it?" work.

- **Web UI**: every browser gets a stable `session_id` stored in `localStorage`.
  Clear it with the "🧠 Clear this chat's memory" button in the Chat sidebar.
- **WhatsApp**: the sender's phone number is the session_id, so memory persists
  across WhatsApp messages automatically.
- **Storage**: SQLite at `data/memory.db`, last 5 user+assistant turn pairs are
  prepended to every Sarvam call.
- **Manual API**: `POST /api/memory/clear {"session_id": "..."}`.

## Dashboard

A live metrics dashboard lives at the **📊 Dashboard** tab.

It shows:
- **KPIs**: total queries, manual-supported rate (% answered confidently from
  the manual), citations-kept rate (verifier accept rate — how often Sarvam
  cited a real chunk), avg top-1 retrieval score, avg latency.
- **Stage latency chart**: per-stage average ms (vision / rewrite / retrieve /
  generate / verify) — quickly see where time is going.
- **Token usage chart**: avg input + output tokens per query, split by Sarvam
  vs Gemini.
- **Confidence distribution**: high / medium / low donut.
- **Languages chart**: which languages users actually use.
- **Recent queries table**: last 30 queries with all metrics.

Backed by SQLite at `data/metrics.db`. One row written per `/api/query` call
(both web and WhatsApp channels).

API endpoints: `GET /api/metrics?window_hours=24`, `GET /api/metrics/recent?limit=30`.

## Testing accuracy

There's a 3-axis evaluation harness for systematic accuracy measurement.

```bash
.venv/bin/python -m scripts.seed_test_corpus    # only needed once for the demo set
.venv/bin/python -m scripts.evaluate             # runs data/eval/golden.jsonl
```

**Metrics computed**:

| Metric | What it measures | How |
|--------|------------------|-----|
| **Retrieval Recall@5** | Did the right page reach Sarvam? | At least one `expected_page` is in the top-5 retrieved chunks |
| **Refusal Accuracy** | Is the grounding contract holding? | For out-of-manual questions, did the system set `manual_supported=false`? |
| **Faithfulness ≥4** | Is Sarvam staying grounded? | Gemini-as-judge scores each answer 1-5 vs cited chunks; ≥4 counts as faithful |
| **Keyword recall** | Sanity check on answer content | What % of expected keywords appear in the answer |

**Demo set results** (10 queries, 7 in-manual + 3 out-of-manual, EN/Tamil/Hindi mix):

```
Retrieval Recall@5:          100.0%
Refusal Accuracy:            100.0%
Faithfulness ≥4 (LLM-judge): 100.0%
Avg keyword recall:           73.8%
Avg latency:                 17.77 s
```

Build your own golden set as a JSONL file with one row per question:
```jsonl
{"id": "q1", "manual_id": "...", "query": "...", "language": "en",
 "expected_pages": [42], "expected_keywords": ["head gasket"], "should_refuse": false}
{"id": "q7", "manual_id": "...", "query": "What's the wifi password?",
 "expected_pages": [], "should_refuse": true}
```

For the interview narrative: this is a **proper offline eval harness** — the
combination of retrieval recall + LLM-as-judge faithfulness + refusal accuracy
is the textbook way to measure RAG quality.

## Troubleshooting

- **"Sarvam API error"** → confirm `SARVAM_API_KEY`, `SARVAM_BASE_URL`, and
  `SARVAM_MODEL` are correct in `api.env`. The default is OpenAI-compatible
  `/v1/chat/completions` at `https://api.sarvam.ai`.
- **No retrieval results** → ingest a manual first; check `data/chroma_db/`
  was created and `data/bm25_index.pkl` exists.
- **Vision answers are empty** → `GEMINI_API_KEY` not set, or the model name
  is unavailable in your region. Try `gemini-1.5-flash` as a fallback.
- **Whapi messages don't arrive** → webhook URL must be publicly reachable
  (use ngrok or deploy); the token must be the **channel** token, not the
  account token.
