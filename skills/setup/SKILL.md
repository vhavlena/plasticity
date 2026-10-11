---
name: setup
description: Check and configure plasticity long-term memory (build, semantic search, scope)
disable-model-invocation: true
allowed-tools: Bash(plasticity *)
---

Run every `plasticity` command below with the `--db` prefix given at session
start (`plasticity --db '...'`), so it uses this session's memory.

Check the plasticity installation and help the user configure it:

1. Run `plasticity stats`. The first run builds the engine from source (needs Go
   ≥ 1.27; takes about a minute). If Go is missing, tell the user to install it
   from https://go.dev/dl.
2. Run `plasticity embed status`. If semantic search is not active, offer:
   - `plasticity embed enable hugot` — local model all-MiniLM-L6-v2 (~90 MB download), or
   - `plasticity embed enable ollama --model nomic-embed-text` — if Ollama runs locally.
   Embeddings are also enabled automatically in the background unless the
   plugin option `embeddings` is `off`.
3. Explain the plugin options (changed via `/plugin` → plasticity → configure):
   `scope` (global or per project), `recall_limit`, `reflect`, `embeddings`, `log`.
4. Memory lives in `~/.plasticity/` (override with `PLASTICITY_HOME`).
