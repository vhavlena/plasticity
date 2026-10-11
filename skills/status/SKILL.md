---
name: status
description: Show the state of plasticity long-term memory (size, semantic search, recent activity)
disable-model-invocation: true
allowed-tools: Bash(plasticity *)
---

Report the state of long-term memory concisely:

1. `plasticity stats` — neurons by type, links by kind, dormant neurons, pending traces.
2. `plasticity embed status` — whether semantic search is active and how many neurons are indexed.
3. `plasticity eval report --since 168h` — activity of the last week: recalls and
   empty-recall rate, usage rate (injected memories reported as used), neurons
   created and duplicates caught, nudges, learned and pruned links, errors.
   Use `plasticity log --since 24h --limit 200` for details of individual events.

Mention problems (errors, missing vectors, many empty recalls) and how to fix them.
