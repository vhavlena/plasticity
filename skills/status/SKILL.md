---
name: status
description: Show the state of plasticity long-term memory (size, semantic search, recent activity)
disable-model-invocation: true
allowed-tools: Bash(plasticity *)
---

Report the state of long-term memory concisely:

1. `plasticity stats` — neurons by type, links by kind, dormant neurons, pending traces.
2. `plasticity embed status` — whether semantic search is active and how many neurons are indexed.
3. `plasticity log --since 24h --limit 200` — summarize recent activity: sessions,
   prompts with recalls, memories reported as used, neurons added (and duplicates
   caught), reflection nudges, errors.

Mention problems (errors, missing vectors, many empty recalls) and how to fix them.
