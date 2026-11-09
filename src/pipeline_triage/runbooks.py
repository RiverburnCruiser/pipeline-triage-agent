"""Runbook retrieval: split runbooks into sections, embed them, search with pgvector.

Two tables:

    runbooks(id, title, body)                        one row per runbook file
    runbook_chunks(runbook_id, section, content,     one row per "## " section,
                   embedding vector(dim))            embedded as title + section + text

A search embeds the query, ranks chunks by cosine distance, keeps the best chunk
per runbook and returns the whole runbook, so the caller gets the full fix even
when the query only matched the Symptoms section.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from pydantic import BaseModel, Field

from pipeline_triage.embeddings import Embedder
from pipeline_triage.store import SAMPLE_DATA_DIR

SAMPLE_RUNBOOKS_DIR = SAMPLE_DATA_DIR / "runbooks"


class Runbook(BaseModel):
    id: str = Field(description="File name without .md, e.g. 'worker-oom'.")
    title: str
    body: str


class Chunk(BaseModel):
    runbook_id: str
    section: str
    content: str

    def embedding_text(self, title: str) -> str:
        return f"{title}\n{self.section}\n{self.content}"


class RunbookHit(BaseModel):
    runbook_id: str
    title: str
    score: float = Field(description="Cosine similarity of the best matching section, 0 to 1.")
    matched_section: str
    body: str = Field(description="The full runbook in Markdown.")


def load_runbooks(directory: Path = SAMPLE_RUNBOOKS_DIR) -> list[Runbook]:
    runbooks = []
    for path in sorted(directory.glob("*.md")):
        body = path.read_text(encoding="utf-8").strip()
        title_line = body.splitlines()[0] if body else ""
        if not title_line.startswith("# "):
            raise ValueError(f"{path.name} must start with a '# Title' line.")
        runbooks.append(Runbook(id=path.stem, title=title_line[2:].strip(), body=body))
    return runbooks


_SECTION = re.compile(r"^## +(.+)$", re.MULTILINE)


def split_sections(runbook: Runbook) -> list[Chunk]:
    """One chunk per '## ' section. Text before the first section is ignored."""
    matches = list(_SECTION.finditer(runbook.body))
    chunks = []
    for match, nxt in zip(matches, [*matches[1:], None], strict=True):
        end = nxt.start() if nxt else len(runbook.body)
        content = runbook.body[match.end() : end].strip()
        if content:
            chunks.append(Chunk(runbook_id=runbook.id, section=match.group(1), content=content))
    return chunks


class RunbookIndex:
    def __init__(self, conn: psycopg.Connection, embedder: Embedder) -> None:
        self._conn = conn
        self._embedder = embedder

    @classmethod
    def connect(cls, database_url: str, embedder: Embedder) -> RunbookIndex:
        conn = psycopg.connect(database_url, autocommit=True)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        return cls(conn, embedder)

    def close(self) -> None:
        self._conn.close()

    def rebuild(self, runbooks: list[Runbook]) -> int:
        """Replace the whole index with these runbooks. Returns the number of chunks.

        The corpus is small, so a full rebuild in one transaction is simpler and
        safer than incremental upserts, and it also handles a changed embedding size.
        """
        chunks = [c for rb in runbooks for c in split_sections(rb)]
        titles = {rb.id: rb.title for rb in runbooks}
        vectors = self._embedder.embed_documents(
            [c.embedding_text(titles[c.runbook_id]) for c in chunks]
        )
        dim = self._embedder.dim
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS runbook_chunks, runbooks")
            cur.execute(
                "CREATE TABLE runbooks (id text PRIMARY KEY, title text NOT NULL,"
                " body text NOT NULL)"
            )
            cur.execute(
                f"""CREATE TABLE runbook_chunks (
                    id serial PRIMARY KEY,
                    runbook_id text NOT NULL REFERENCES runbooks (id),
                    section text NOT NULL,
                    content text NOT NULL,
                    embedding vector({dim}) NOT NULL)"""
            )
            cur.executemany(
                "INSERT INTO runbooks (id, title, body) VALUES (%s, %s, %s)",
                [(rb.id, rb.title, rb.body) for rb in runbooks],
            )
            cur.executemany(
                "INSERT INTO runbook_chunks (runbook_id, section, content, embedding)"
                " VALUES (%s, %s, %s, %s)",
                [
                    (c.runbook_id, c.section, c.content, np.asarray(v, dtype=np.float32))
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )
            # Approximate nearest-neighbour index; search() orders by the same operator.
            cur.execute("CREATE INDEX ON runbook_chunks USING hnsw (embedding vector_cosine_ops)")
        return len(chunks)

    def search(self, query: str, k: int = 3) -> list[RunbookHit]:
        """Top `k` runbooks for the query, best first.

        Fetches the nearest sections through the HNSW index, then keeps the best
        section per runbook. Over-fetching by 5x leaves room for runbooks that
        match on several sections.
        """
        vector = np.asarray(self._embedder.embed_query(query), dtype=np.float32)
        rows = self._conn.execute(
            """SELECT c.runbook_id, r.title, 1 - (c.embedding <=> %(v)s), c.section, r.body
               FROM runbook_chunks c JOIN runbooks r ON r.id = c.runbook_id
               ORDER BY c.embedding <=> %(v)s
               LIMIT %(limit)s""",
            {"v": vector, "limit": k * 5},
        ).fetchall()
        hits: dict[str, RunbookHit] = {}
        for runbook_id, title, score, section, body in rows:
            if runbook_id not in hits:
                hits[runbook_id] = RunbookHit(
                    runbook_id=runbook_id,
                    title=title,
                    score=round(float(score), 4),
                    matched_section=section,
                    body=body,
                )
        return list(hits.values())[:k]
