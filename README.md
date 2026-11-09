# Pipeline Triage Agent

An AI agent that investigates failed data pipeline runs the way an on-call data
engineer would: it finds the failed run, reads the failing task's log, checks
whether table volumes look wrong, looks up the matching runbook, and proposes a
root cause and a fix for a human to approve.

The project is built in four stages. Two are done:

- **Stage 1, MCP server:** gives any LLM client safe, read-only access to pipeline
  run status, task logs and table row counts.
- **Stage 2, runbook search (RAG):** runbooks embedded into PostgreSQL with
  pgvector, so a client can search them with an error message and get back the
  matching diagnosis and fix.

## Why MCP

The [Model Context Protocol](https://modelcontextprotocol.io) (MCP) is a standard
way to expose tools to LLM applications. Think of it as USB-C for AI tools: the
server is written once, and any MCP client can plug into it, whether that is
Claude Code, Claude Desktop, the MCP Inspector, or the LangGraph agent coming in
stage 3. The agent never touches files or databases directly; it can only call
the tools below, and every tool is read-only.

## Architecture

```mermaid
flowchart LR
    subgraph clients[MCP clients]
        CC[Claude Code / Claude Desktop]
        INS[MCP Inspector]
        AG["LangGraph agent<br/>(stage 3)"]
    end

    subgraph server[pipeline-triage MCP server]
        T[Pipeline tools<br/>get_failed_runs<br/>get_run_status<br/>get_task_log<br/>query_row_counts<br/>list_tables]
        R[search_runbooks]
        S[RunStore]
        I[RunbookIndex]
        E["Embedder<br/>bge-small-en-v1.5 (ONNX)"]
    end

    D[("sample_data/<br/>runs.json, row_counts.json, logs/")]
    PG[("PostgreSQL + pgvector<br/>runbooks, runbook_chunks")]
    MD["sample_data/runbooks/*.md"]

    CC -- stdio --> T
    CC -- stdio --> R
    INS -- stdio --> T
    AG -. stdio .-> T
    AG -. stdio .-> R
    T --> S --> D
    R --> I --> PG
    I --> E
    MD -- pipeline-triage-ingest --> I
```

The pipeline tools only talk to `RunStore`, which reads JSON fixtures and log
files today and can be swapped for a database-backed store later. Runbook search
only talks to `RunbookIndex`, which owns the PostgreSQL tables.

## Tools

| Tool | What it returns |
| --- | --- |
| `get_failed_runs(dag_id?, since?, limit=20)` | Failed runs, newest first, with the names of the tasks that failed. |
| `get_run_status(run_id)` | One run's state and every task's state, try number and timings. |
| `get_task_log(run_id, task_id, tail_lines=200)` | The last lines of a task's log, and whether it was truncated. |
| `query_row_counts(table, days=7)` | Daily row counts plus the latest day's change against the earlier days' average. |
| `list_tables()` | Tables that `query_row_counts` can inspect. |
| `search_runbooks(query, k=3)` | The `k` runbooks that best match an error message, each with its similarity score, the section that matched, and the full runbook. Only available when `DATABASE_URL` is set. |

Every tool returns structured JSON with an output schema, so clients can parse
results without scraping text. Unknown ids and invalid arguments come back as
MCP tool errors with a message that says what exists, so a model can correct
itself instead of guessing.

## Sample incidents

`src/pipeline_triage/sample_data/` holds ten Airflow-style runs across four DAGs.
Five of them failed, each with a different root cause. These are the seeded
incidents that the stage 4 evals will score the agent against.

| Run | Failed task | Root cause shown in the logs |
| --- | --- | --- |
| `trip_events_daily__2026-10-28` | `validate_schema` | **Schema drift**: an upstream release renamed `speed_kmh` to `speed_kph` and changed `event_ts` to a string. |
| `driver_scores_hourly__2026-10-29T03` | `wait_for_trip_events` | **Upstream dependency**: the sensor timed out because the run above never loaded. |
| `vendor_payments_sync__2026-10-29` | `extract_vendor_api` | **Credential expiry**: the OAuth client secret expired, so the token refresh got a 401. |
| `device_health_rollup__2026-10-30` | `aggregate_metrics` | **Out of memory**: the pandas groupby blew past the pod's 4 GiB limit and was OOMKilled. |
| `trip_events_daily__2026-10-30` | `reconcile_counts` | **Partial load**: only 37 of 96 source files arrived, so rows fell 62.2% below the baseline. |

The second incident is caused by the first, which gives the agent a chance to
show it can link related failures instead of triaging each one in isolation.

## Runbook search (RAG)

Retrieval-Augmented Generation (RAG) means fetching the relevant documents first
and letting the model answer from them, like an open-book exam. Stage 2 builds the
retrieval half: given an error from a log, find the runbook that explains it.

**How it works**

1. **Corpus.** `sample_data/runbooks/` has nine runbooks: one for each of the
   five seeded root causes, plus four decoys that look similar on the surface
   (API rate limiting next to expired credentials, duplicate rows next to a
   partial load, a warehouse lock timeout, a full staging disk). The decoys keep
   the search honest.
2. **Chunking.** Each runbook is split on its `##` sections (Symptoms, Likely
   causes, Diagnosis, Fix, Prevention). Each chunk is embedded together with its
   runbook title and section name.
3. **Embeddings.** [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
   runs locally through [fastembed](https://github.com/qdrant/fastembed) (ONNX).
   It needs no API key, no GPU and no PyTorch, and produces 384-dimensional vectors.
   Queries and passages are embedded differently, as the model expects.
4. **Storage.** The `runbooks` and `runbook_chunks` tables live in PostgreSQL with
   the pgvector extension. Chunks are indexed with HNSW on cosine distance.
   `pipeline-triage-ingest` rebuilds both tables in one transaction, so re-running
   it is safe.
5. **Search.** The query is embedded, the nearest sections are fetched through
   the index, and the best section per runbook decides its rank. Each hit returns
   the whole runbook, so the caller gets the fix even when only the Symptoms
   section matched.

**Retrieval evaluation**

`sample_data/labels.json` records the expected root cause and runbook for each
seeded incident. `pipeline-triage-eval-retrieval` searches with the evidence from
each failing task's log (the log minus Airflow's own bookkeeping lines) and
reports whether the right runbook came back.

| Embedder | hit@1 | hit@3 | MRR |
| --- | --- | --- | --- |
| Keyword baseline (`HashingEmbedder`, used in unit tests) | 80% | 100% | 0.87 |
| bge-small-en-v1.5 (real model, checked in CI) | 100% | 100% | 1.00 |

Five incidents is a small set, so read these as a regression check rather than a
benchmark. The CI test requires the real model to rank the right runbook first
for every incident.

## Project layout

```
src/pipeline_triage/
  models.py        Pydantic models for runs, tasks, logs and row counts
  store.py         RunStore: read-only access to the data directory
  mcp_server.py    MCP server and its tools (entry point: main)
  embeddings.py    Embedder interface: real model and a test stand-in
  runbooks.py      Runbook loading, section chunking, RunbookIndex (pgvector)
  retrieval_eval.py  Scores runbook search against the labelled incidents
  rag_cli.py       pipeline-triage-ingest and pipeline-triage-eval-retrieval
  sample_data/     Seeded runs, row counts, task logs, runbooks and labels
tests/
  test_store.py       Unit tests for RunStore
  test_mcp_server.py  End-to-end tests through a real MCP client
  test_runbooks.py    Chunking, indexing and search (database tests need Postgres)
  test_retrieval_eval_model.py  Retrieval quality with the real model
docker-compose.yml   Local PostgreSQL with pgvector
```

## Local setup

Requires Python 3.11 or newer. On Ubuntu/Debian, install the venv module first
(`sudo apt install python3-venv`) and use `python3` where the commands below say `python`.

```bash
git clone https://github.com/RiverburnCruiser/pipeline-triage-agent.git
cd pipeline-triage-agent
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Start PostgreSQL with pgvector (needs Docker) and load the environment variables:

```bash
cp .env.example .env
docker compose up -d
set -a && source .env && set +a     # exports DATABASE_URL for this terminal
```

Build the runbook index. The first run downloads the embedding model (about 70 MB):

```bash
pipeline-triage-ingest              # Indexed 9 runbooks as 45 sections.
pipeline-triage-eval-retrieval      # prints the retrieval report
```

### Running the tests

Run the same checks GitHub Actions runs on every pull request:

```bash
pytest
ruff check . && ruff format --check .
mypy
```

Tests that need PostgreSQL run when `TRIAGE_TEST_DATABASE_URL` is set (it is in
`.env.example`) and are skipped otherwise. The real-model retrieval test also needs
`TRIAGE_TEST_REAL_MODEL=1`. Without Docker, `pytest` still runs everything else.

The database tests use their own database, `triage_test`, in the same container.
They drop and rebuild the `runbooks` and `runbook_chunks` tables, and some build
them with the test embedder, so sharing a database with `DATABASE_URL` would wipe
the real index. `search_runbooks` would then fail with `different vector dimensions`
until you ran `pipeline-triage-ingest` again. To prevent that:

- pytest creates `triage_test` on first use, so it works on an existing Docker
  volume and in CI without any setup step.
- pytest fails with an explanation if `TRIAGE_TEST_DATABASE_URL` and `DATABASE_URL`
  point at the same database.

If your `.env` was copied from an older `.env.example`, change
`TRIAGE_TEST_DATABASE_URL` to end in `/triage_test`, then run
`pipeline-triage-ingest` once to restore the real index.

### Running the server

The server speaks MCP over stdio, so it will sit waiting for a client:

```bash
pipeline-triage-mcp
```

With `DATABASE_URL` set it also exposes `search_runbooks`, and loads the
embedding model at startup. To point it at your own pipeline data, set
`TRIAGE_DATA_DIR` to a folder with the same layout as `sample_data/`.

## Try it with an MCP client

### Claude Code

From the repo folder, with the virtual environment active:

```bash
claude mcp add pipeline-triage -e DATABASE_URL="$DATABASE_URL" -- "$(which pipeline-triage-mcp)"
```

Leave out `-e DATABASE_URL=...` to run without runbook search.

Then start `claude` and ask something like:

> Which pipeline runs failed most recently? Pick the vendor payments one, read
> the failing task's log, find the matching runbook and tell me the fix.

Run `/mcp` inside Claude Code to check the server is connected.

### MCP Inspector

The Inspector is a web UI for calling tools by hand, which is handy for
checking schemas and error messages. It needs Node.js.

```bash
npx @modelcontextprotocol/inspector "$(which pipeline-triage-mcp)"
```

### Claude Desktop

Add this to `claude_desktop_config.json`, using the absolute path printed by
`which pipeline-triage-mcp` (on Windows, `where pipeline-triage-mcp`):

```json
{
  "mcpServers": {
    "pipeline-triage": {
      "command": "/absolute/path/to/.venv/bin/pipeline-triage-mcp",
      "env": { "DATABASE_URL": "postgresql://triage:triage@localhost:5432/triage" }
    }
  }
}
```

## Roadmap

| Stage | Scope | Status |
| --- | --- | --- |
| 1. MCP server | Read-only tools over run status, task logs and row counts, with seeded incidents and tests. | Done |
| 2. Runbook RAG | Runbooks embedded into PostgreSQL with pgvector, exposed as a `search_runbooks` tool, with a retrieval evaluation over the seeded incidents. | Done |
| 3. LangGraph agent | An agent that calls the MCP tools and the runbook search, classifies the root cause, drafts a fix, and waits for human approval before any action. | Planned |
| 4. Ship it | FastAPI service, Docker Compose (app + PostgreSQL), tracing with Langfuse, and an eval suite over the seeded incidents running in GitHub Actions. | Planned |

## Design notes

- **Read-only by design.** Every tool is annotated `readOnlyHint`. The only path
  to an action will be the human approval step in stage 3.
- **No client-controlled file paths.** Log locations come from `runs.json`, and
  the store still refuses any path that resolves outside the data directory.
- **Internal fields stay internal.** The log file path is excluded from tool
  output, so clients see run and task ids only.
