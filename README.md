# plasticity

Long-term memory for Claude Code, shaped like a brain. Knowledge is stored in
typed **neurons** connected by weighted **synapses**. Neurons used together
get wired together; connections that are not used fade and disappear.

## Install

Requires [Go](https://go.dev/dl) ≥ 1.27 (the engine is built from source on
first use).

```
/plugin marketplace add vhavlena/plasticity
/plugin install plasticity@plasticity
```

For development: `claude --plugin-dir /path/to/plasticity`.

## How it works in Claude Code

- **First session:** a background hook builds the engine and downloads a
  local embedding model (~90 MB). Until it is ready, search is keyword-only.
- **Every prompt:** relevant memories are recalled and injected into the
  context (up to 6 entries, ~5 ms). Task notifications of finished
  background commands and agents are skipped, except for failed tasks,
  which are recalled on their error output.
- **While working:** Claude recalls more by itself when the need appears,
  for example before a design decision, in unfamiliar code, or when stuck on
  an error.
- **Storing:** Claude stores durable knowledge (decisions with
  rationale, root causes, procedures, preferences) with `plasticity add`,
  and reports the memories that helped with `plasticity used`. Those
  reports teach the graph which knowledge belongs together.
- **End of a task:** once per session, after substantive work, Claude is
  asked to store anything worth keeping.
- **End of a session:** the session is closed. Its history (first prompt,
  the memories it used and created) is kept outside the graph and listed by
  `plasticity sessions`. With the `episodes` option, an `episode` neuron also
  records the prompts and edited files, linked to that knowledge; episodes
  are never injected into prompts, only found by explicit recall and search.

**Commands:** `/plasticity:remember <text>`, `/plasticity:recall <query>`,
`/plasticity:status`, `/plasticity:graph` (interactive view of the graph
in the browser), `/plasticity:setup`, and `/plasticity:flag
miss|noise|helped <what happened>` to record observations while evaluating.

**Options** (`/plugin` → plasticity → configure):

| option | default | meaning |
|---|---|---|
| `scope` | `global` | one memory for all projects, or `project` (one per git repository) |
| `recall_limit` | 6 | memories injected per prompt |
| `reflect` | true | end-of-task request to store or report memories |
| `session_idle_hours` | 48 | a session left without a clean exit is closed after this idle time; resuming it later reopens it (and refreshes its episode) |
| `episodes` | false | store each session as an `episode` neuron |
| `embeddings` | `auto` | `auto` (local model), `ollama`, or `off` |
| `notifications` | `failed` | task notifications of background commands and agents arrive as prompts: `failed` recalls only for failed tasks (on their error output, confident matches only), `all` treats them like prompts, `off` never recalls |
| `log` | true | local event log (`plasticity eval report` summarizes it) |

All data stays in `~/.plasticity/` (or `$PLASTICITY_HOME`) and never leaves
your machine.

## Architecture

**Neurons** have a type (`concept`, `fact`, `decision`, `rationale`,
`procedure`, `episode`, `question`), a title, content, rationale, tags and
importance.

**Synapses** come in two kinds:
- **associative** synapses are learned: symmetric, created and strengthened
  only by co-use, and fade towards a small floor weight but are never removed;
- **typed** synapses (`supports`, `derives_from`, `contradicts`, `part_of`,
  `explicit`) are authored: directed, slowly decaying towards a floor weight.

**Recall** runs in three steps:
1. Seeds come from BM25 keyword search, fused with embedding similarity when
   embeddings are enabled. Weak matches produce weak seeds, and a query with
   no good match returns nothing.
2. Activation spreads from the seeds along synapses and is multiplied by each
   synapse's weight, so strong connections reach deeper than weak ones (with
   defaults: weight 0.9 reaches 7 hops, 0.3 reaches 1 hop).
3. Results are ranked by activation, recency of use (ACT-R) and importance.
   Each result carries the path that activated it.

**Learning:**
- **Hebbian strengthening:** reinforcing co-used neurons applies
  `Δw = η·aᵢ·aⱼ·(1−w)`.
- **Spacing effect:** a link reinforced when nearly forgotten becomes
  markedly more stable.
- **Weakening (LTD):** recalled-but-unused neurons lose weight on their links
  to the used ones.
- **Homeostatic scaling:** stops hubs from taking over by capping a
  neuron's summed link weight above the floors.

**Forgetting is counted in recalls, not days.** A synapse ages each time one
endpoint is recalled without the other: `w = w₀·e^(−age/S)`. Unused parts of
the graph never change, and a vacation erodes nothing. Weakening (decay, LTD,
scaling) never removes a link: it stops at a floor weight. Consolidation marks
isolated, unused neurons dormant; dormant neurons stay
searchable.

**Storage** is a single SQLite file in WAL mode (pure Go, no server), so
concurrent hooks and sessions can share it. Every constant is tunable with
`plasticity params`.

## Development

```sh
go build -o plasticity ./cmd/plasticity
go test -race ./...
plasticity --help                           # full CLI
plasticity sessions                         # past sessions with the memories they used and created
plasticity viz [--3d]                       # interactive 2D/3D HTML view of the graph (export -f html|json|dot)
plasticity eval run SCENARIO.json           # offline evaluation of recall and learning
plasticity eval report --since 168h         # how memory behaved in real sessions (+ your notes)
plasticity eval report --by-change          # compare periods between parameter changes
plasticity eval export -o mine.json         # replayable scenario from real sessions
```

```
cmd/plasticity       CLI
internal/engine      recall, learning, consolidation, sessions
internal/plasticity  synaptic dynamics
internal/retrieval   spreading activation, personalized PageRank
internal/store       SQLite schema, FTS5 search
internal/embed       pluggable embedders (hugot in-process, Ollama)
internal/hooks       Claude Code hook handlers
internal/eval        scenario replay, metrics, usage reports
internal/viz         self-contained interactive HTML visualization (2D canvas, 3D WebGL)
```

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
