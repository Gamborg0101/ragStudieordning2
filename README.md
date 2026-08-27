# Studieordning RAG

A retrieval-augmented Q&A system over Aarhus University's Danish "studieordninger" (academic
regulations / curricula) — ~250 HTML documents covering admission requirements, ECTS structure,
exam rules, and course descriptions for bachelor and master programmes.

The focus of this project isn't the RAG pipeline itself — it's the **evaluation-driven process**
used to find and fix retrieval failures. Below is that process, with the actual numbers.

## Architecture

```
corpus/*.html --(BeautifulSoup)--> plain text --(RecursiveCharacterTextSplitter)--> chunks
    --(nomic-embed-text via Ollama)--> vectors --> InMemoryVectorStore --(cached as .npz)

question --> vector search (MMR) --> retrieved chunks --> deep agent (qwen2.5:7b) --> answer
```

- **Ingestion** (`datacollection/`): parses the HTML corpus, chunks it (1000 chars, 200 overlap),
  embeds it with `nomic-embed-text`, and persists the result to `data/vector_store.npz` so
  re-runs skip the expensive embedding step.
- **Answering** (`index.py`): a [`deepagents`](https://github.com/langchain-ai/deepagents) agent
  (Ollama `qwen2.5:7b`) that calls a retrieval tool and delegates chunk analysis to a subagent.
- **Evaluation** (`evaluation.py`): two harnesses — retrieval (hit-rate on a labeled question set)
  and generation (LLM-as-judge grading against gold answers).

## The retrieval problem

`eval/retrieval.jsonl` holds a small set of hand-labeled questions, each with the id of the
document that should be retrieved. The metric: does the correct document show up in the top-4
retrieved chunks?

**Baseline** (plain cosine-similarity search, k=4): **6/11 hits (55%)**.

Reading the misses by hand showed a pattern: several documents shared large blocks of
near-identical boilerplate — administrative sections like admission requirements
(*"1.3 Adgangskrav og forudsætninger..."*) restated almost verbatim across dozens of unrelated
programmes, differing only in a subject name and a short list of requirements. In embedding
space, a chunk wasn't competing against irrelevant content — it was competing against dozens of
near-duplicates of itself, and losing.

### Iteration 1 — Maximal Marginal Relevance

Swapped plain similarity search for MMR, which reranks candidates to penalize redundancy with
what's already selected, instead of purely ranking by relevance. Tuned `λ` (relevance/diversity
trade-off) and `fetch_k` (candidate pool size) against the eval set:

| Config                          | Hit rate |
|----------------------------------|----------|
| Plain similarity search          | 6/11 (55%) |
| MMR, λ=0.5, fetch_k=20 (default) | 7/11 (64%) |
| MMR, λ=0.7, fetch_k=20           | 8/11 (73%) |

MMR helped, but tuning `fetch_k` from 10 up to 40 stopped moving three specific misses at all —
a sign the problem wasn't rank-order, it was that the correct chunk wasn't even reaching the
candidate pool in the first place.

### Iteration 2 — fixing the root cause

Traced one of those misses to its source chunk directly: the content was correct and complete,
but it was one of ~50+ chunks sharing the same "admission requirements" template across the
corpus, with the actual differentiator (the programme name) buried inside a wall of identical
boilerplate. The fix: extract each document's `<title>` during ingestion and prepend it to every
one of its chunks before embedding, giving the embedding model a strong, repeated anchor back to
the correct document.

| Config                                  | Hit rate |
|-------------------------------------------|----------|
| MMR (λ=0.7) + title-prefixed chunks       | **9/11 (82%)** |

### What's still failing, and why

The two remaining misses share one label (`"family": "eksamenssprog"` in the eval set) — both
ask which language a programme teaches/examines in. Unlike the admission-requirements case, this
rule is stated almost identically **university-wide**, repeated per-course inside every single
document — a much larger-scale duplication than title-prefixing can out-weigh with a short prefix
against that much repeated text.

Current direction: **metadata filtering**. Fuzzy-match (`rapidfuzz`) the incoming question
against a `source → title` lookup table built from the corpus, to identify which document the
question is about *before* running vector search — then restrict the search to that document's
chunks only, so structurally identical chunks from unrelated documents can't compete at all. This
is in progress in `evaluation.py`.

### Honest caveats

- The eval set is 11 questions. Directionally useful, not statistically strong — one flipped
  answer swings the hit rate by ~9 points. Expanding it is the next priority before drawing
  stronger conclusions.
- Metadata filtering requires knowing which document a question refers to, which only works
  because the eval questions name the programme explicitly. Real user questions may not.

## Generation evaluation

`eval/generation.jsonl` holds question/gold-answer pairs. `grade_answer` asks a second model
(`qwen2.5:7b`, temperature 0) to judge the agent's answer against the gold answer on:
reasoning, whether it asked for clarification, whether it contains conflicting statements, and
factual accuracy — an LLM-as-judge pattern, separate from the retrieval metric above.

## Running it

```
pip install -r requirements.txt
python3 evaluation.py     # runs the retrieval eval (see main-level call at bottom of file)
python3 index.py          # runs the agent against a single example query
```

Requires a local [Ollama](https://ollama.com) instance serving `nomic-embed-text` and
`qwen2.5:7b`.
