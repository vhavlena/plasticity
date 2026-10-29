---
name: graph
description: Open an interactive visualization of plasticity long-term memory (the knowledge graph) in the browser
disable-model-invocation: true
allowed-tools: Bash(plasticity *)
---

Run every `plasticity` command below with the `--db` prefix given at session
start (`plasticity --db '...'`), so it uses this session's memory.

Run `plasticity viz` (add `--3d` if the user asks for the 3D view). It writes the whole graph as a self-contained HTML page
(next to the database, or `-o FILE`) and opens it in the browser. Report the
file path from the output; if `opened` is false, tell the user to open that
file themselves. Mention briefly what the page offers: filters by neuron type
and synapse kind, a minimum-weight slider, search, and a details panel with
content, rationale and connections when a neuron is clicked; a 2D/3D switch,
and in 3D an orbit camera, community coloring, and a replay of spreading
activation from a selected neuron.
