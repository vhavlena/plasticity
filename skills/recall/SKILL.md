---
name: recall
description: Search plasticity long-term memory, e.g. /plasticity:recall why we chose SQLite
argument-hint: <query>
disable-model-invocation: true
allowed-tools: Bash(plasticity *)
---

Search long-term memory for: $ARGUMENTS

Run `plasticity recall "$ARGUMENTS" --session <SESSION_ID>` (the session ID
given at session start; omit `--session` if you do not have one) and present the hits as a short
list: type, title, the essential content and rationale, and — for hits that are
not direct matches (`is_seed: false`) — which memory they are associated with
(`path_titles`). If nothing relevant is found, say so and suggest
`plasticity search` with different keywords.
