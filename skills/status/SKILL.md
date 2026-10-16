---
name: status
description: Show the state of plasticity long-term memory (size, semantic search, recent activity)
disable-model-invocation: true
allowed-tools: Bash(plasticity *)
---

Run every `plasticity` command below with the `--db` prefix given at session
start (`plasticity --db '...'`), so it uses this session's memory.

Report the state of long-term memory concisely:

1. `plasticity stats` — neurons by type, links by kind, dormant neurons, pending traces.
2. `plasticity embed status` — whether semantic search is active and how many neurons are indexed.
3. `plasticity eval report --since 168h` — activity of the last week: recalls and
   empty-recall rate, usage rate (injected memories reported as used), neurons
   created and duplicates caught, nudges, learned links, the user's
   notes (`/plasticity:flag`), parameter changes, errors.
   If parameters changed, add `plasticity eval report --by-change` to compare
   the periods before and after.
   Use `plasticity log --since 24h --limit 200` for details of individual events.

Mention problems (errors, missing vectors, many empty recalls) and how to fix them.
