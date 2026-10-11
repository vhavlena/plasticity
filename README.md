# plasticity

A knowledge graph that behaves like a brain, built as long-term memory for
Claude Code. Knowledge is stored in typed **neurons** (nodes holding
information and its rationale) connected by weighted **synapses** (edges).
Neurons used together get wired together, and unused connections fade and
eventually disappear.

This repository contains part 1: the Go core and a JSON-speaking CLI. Part 2
will wire the CLI into Claude Code through hooks (automatic recall and
learning) and skills (deliberate storing and linking).

## Build

```sh
go build -o plasticity ./cmd/plasticity
go test -race ./...
```

Storage is a single SQLite file in WAL mode (pure Go, no cgo, no server). The
default location is `~/.plasticity/graph.db`; override it with `--db` or
`$PLASTICITY_DB`. Several processes (parallel hooks, concurrent sessions) can
use the same file at once: readers never block, and writers queue for
milliseconds.

## Model

**Neuron types:** `concept`, `fact`, `decision`, `rationale`, `procedure`,
`episode`, `question`. Each neuron has a title, content, rationale, tags,
source and an importance between 0 and 1.

**Synapse kinds:**
- `associative` synapses are learned. They are symmetric and are created and
  strengthened only through co-activation.
- `supports`, `derives_from`, `contradicts`, `part_of` and `explicit` are
  typed links that you create yourself. They are directed and decay slowly
  towards a floor weight. They are never pruned unless you pass
  `--prune-typed`.

## Algorithms

### Time is counted in recalls, not days

The graph has no notion of wall-clock time, so a vacation erodes nothing.
Forgetting is caused by **interference**: each neuron counts the recalls it
fired in (`fires`), and a synapse records both endpoints' counts at its last
update. Its age is

```
age(A–B) = (fires_A − mark_A) + (fires_B − mark_B)
```

which is the number of times A or B was recalled without the other. When A
and B fire in the same recall, the synapse's marks advance too, so being
recalled together never ages it. Parts of the graph you don't use don't change
at all, and unrelated projects in the same database don't interfere with each
other. Stability `S` is measured in such solo firings: the default
`initial_stability` of 20 means a fresh link loses 63% of its weight after 20
solo firings of its endpoints. A global recall counter (`tick`) serves only as
the clock for ACT-R recency, dormancy and trace retention. Wall-clock time is
kept only for informational timestamps (`created_at`, `updated_at`).

| Mechanism | Rule |
|---|---|
| Forgetting by interference | Effective weight `w = w₀·e^(−age/S)`, computed lazily when read. `age` is the number of times either endpoint fired (was recalled) without the other since the last update |
| Hebbian LTP | `Δw = η·aᵢ·aⱼ·(1−w)`: soft-bounded, so a weight approaches but never reaches 1 |
| Independence | When learning from recall traces, a pair is potentiated only if its co-activation is real evidence: the two nodes came from different seeds and neither lies on the other's activation path. Otherwise one node fired *because of* the other, or both because of one seed, and learning would be circular: a link would strengthen itself, or triangles would keep closing. Explicit `reinforce` and `feedback` are always learned |
| Spacing effect (FSRS-style) | `S ← S·(1 + α·(g + 1 − R))`, where `R = e^(−age/S)` at the moment of reinforcement. Reinforcing a link that was nearly forgotten makes it last much longer; repeating it many times in a row helps only a little |
| Anti-Hebbian LTD | `Δw = −η_d·aᵢ·aⱼ·w`, applied to links between used nodes and nodes that were recalled but not used |
| Homeostatic scaling | If a node's associative weights sum to more than `W_max`, all of them are scaled down by the same factor (stops hubs from taking over) |
| Pruning | During `consolidate`, associative edges whose effective weight is below `θ_prune` are deleted |
| ACT-R base level | `B = ln Σ (t−tⱼ)^−d` over a node's recorded accesses, where `t` is a global recall counter. Used as a ranking prior and for dormancy |
| Dormancy | Isolated nodes with low `B` are marked dormant. They stay searchable and wake up when recalled |

### Recall: weight-gated spreading activation

1. **Seeds.** An FTS5 BM25 search over title, content, rationale and tags
   (column-weighted) finds candidate nodes. Each candidate's strength is
   `(bm25 / bm25_best) · C`, where `C` is the **query confidence**: the
   IDF-weighted share of the query terms that the candidates match. Query words
   that appear nowhere in the graph count against `C`, scaled by
   `absent_term_weight`.

   The relative BM25 factor ranks candidates within one query, and `C` makes
   strengths comparable across queries. A query whose best match covers only
   a small, unimportant part of it therefore produces weak seeds, which spread
   activation less far. Candidates below `min_seed_strength` are dropped, so a
   query with no good match returns nothing. Queries about several topics are
   not penalized, because `C` measures what the candidates match together.

   **Semantic seeds (optional).** With embeddings enabled, the query is also
   embedded and compared against every node's vector by cosine similarity.
   Similarity is calibrated to a strength `S = clamp((cos − 0.25)/(0.6 − 0.25))`
   (`vector_cos_low`, `vector_cos_high`). The lexical strength `L` and `S` are
   fused by noisy-OR: `seed = 1 − (1 − L)(1 − S)`. Either signal alone can
   seed a node, and agreement makes a seed stronger. The learning gate uses
   `max(C, best S)`. A paraphrase with no shared words, such as "how do
   memories get stronger" finding "Hebbian learning", can therefore still
   seed recall.
2. **Spreading.** Activation flows outward from each seed. Crossing an edge of
   weight `w` from a node with degree `k` multiplies the activation by
   `δ·w / max(1, k/K₀)^β`. A node is reached only while its activation stays
   at or above the threshold `θ`.

   Every factor is at most 1, so the search runs best-first (a Dijkstra-style
   search on `−ln a`). It visits only nodes that will be returned, and strong
   connections carry activation further than weak ones. With the defaults
   (`δ=0.8`, `θ=0.1`), a chain of edges with weight 0.9 reaches 7 hops, weight
   0.6 reaches 3 hops, and weight 0.3 reaches 1 hop.
3. **Combining seeds.** When a node is reached from several seeds, the
   activations are combined as `1 − ∏(1 − aₛ)` (noisy-OR).
4. **Optional re-ranking.** `--rerank ppr` runs personalized PageRank (as in
   HippoRAG) on the reached subgraph.
5. **Final score.** `0.7·graph + 0.2·σ(B) + 0.1·importance`. Each result
   includes the path that activated it.
6. **Recording.** Recall records an access for each returned node. With
   `--session`, it may also store a trace that `reinforce --session` uses
   later. A learning gate keeps weak matches from rewiring the graph: the trace
   is stored only if `C ≥ learn_min_confidence`, and it includes only nodes
   with activation of at least `learn_min_activation`. Otherwise the result
   explains why in `trace_skipped`. Each result also lists the query terms with
   their document frequency, IDF and whether they matched, which helps when
   tuning.

Every constant can be inspected with `plasticity params`, changed with
`plasticity params set KEY VALUE` and restored with `plasticity params reset KEY`.

## CLI

```sh
plasticity add -t decision --title "Use SQLite WAL" -c "..." -r "hooks are processes" --tags db
plasticity add --stdin --link ID:supports:0.8 < node.json
plasticity get ID                         # neuron + synapses with current weights
plasticity update ID --importance 0.9
plasticity link SRC DST -k supports -w 0.8
plasticity search "query"                 # BM25 only
plasticity recall "query" -s SESSION      # associative recall, stores a trace
plasticity reinforce -s SESSION           # Hebbian learning from the session's traces
plasticity reinforce ID ID ID             # these neurons were used together
plasticity feedback -s SESSION -u ID,ID   # LTP for used neurons, LTD for unused ones
plasticity consolidate                    # pruning, homeostasis, dormancy
plasticity stats | export -f dot | params
plasticity embed enable hugot             # download all-MiniLM-L6-v2, index all neurons
plasticity embed enable ollama --model nomic-embed-text
plasticity embed status | reindex [--all] | disable | prune
plasticity --no-embed recall "query"      # lexical-only baseline on the same graph
```

Global flags: `--db`, `--pretty`.

## Embeddings

Embeddings are optional, free and local. Without them, recall is purely
lexical. The embedder configuration is stored in the database, because all
vectors in a graph must come from the same model.

| backend | how it runs | default model | cost |
|---|---|---|---|
| `hugot` | in-process, pure Go (no cgo, no server) | `sentence-transformers/all-MiniLM-L6-v2`, 384 dims, ~90 MB, downloaded next to the database | ~80 ms per CLI call to load the model, ~15 ms per further text; ~390 MB peak memory |
| `ollama` | HTTP to a local Ollama server | `nomic-embed-text`, with its `search_query:` / `search_document:` prefixes | depends on the server |

- **Indexing.** Nodes are embedded when they are added or updated. Text is
  truncated to `embed_max_words` (200) words, because inference cost grows
  quadratically with length.
- **Failures don't block.** If the embedder is unavailable, the node is
  stored without a vector, recall falls back to BM25, and a warning goes to
  stderr. `embed reindex` fills in missing vectors later.
- **Search.** Vectors are searched by brute force. This takes a few
  milliseconds per thousand nodes; an approximate index such as HNSW is a
  later option.
- **Several embedders side by side.** Vectors are stored per embedder ID.
  `embed enable` with a different model indexes alongside the existing
  vectors, and switching back to an earlier embedder re-embeds nothing.
  `embed disable` stops using embeddings but keeps the vectors, and
  `embed prune` deletes every embedder's vectors except the active one's.
- **Known library issues.** hugot v0.8.2's Go backend crashes on batches whose
  size isn't one of its internal buckets, so texts are embedded one at a
  time. Its matmul code also trips Go's checkptr checks under `-race`, so
  the real-model tests skip themselves there. To run those tests, set
  `PLASTICITY_TEST_MODEL` to a downloaded model directory.

### Plugging in an embedder

Any type that implements `embed.Embedder` can be used:

```go
type Embedder interface {
	ID() string // vectors are stored per ID; never mixed across IDs
	Embed(ctx context.Context, text string, kind embed.Kind) ([]float32, error) // L2-normalized
	Close() error
}
```

There are two ways to plug one in:

- **Programmatically**, for experiments and tests. Pass it straight to the
  engine with `engine.Open(path, engine.WithEmbedder(myEmb))`. This overrides
  the stored configuration, and the caller keeps ownership of the embedder.
  `engine.WithoutEmbeddings()` gives a lexical-only engine on the same
  database, which makes A/B comparisons easy.
- **By name**, so the embedder can be selected from configuration and the
  CLI. Register a provider:

  ```go
  embed.Register("mymodel", embed.Provider{
      Factory: func(ctx context.Context, c embed.Config) (embed.Embedder, error) {
          return newMyEmbedder(c.Model, c.Options["dim"])
      },
      // Optional: fill in default config values, or set up once on enable
      // (e.g. download a model into modelsDir).
      Defaults: func(c embed.Config) embed.Config { return c },
      Prepare:  func(ctx context.Context, c embed.Config, modelsDir string) (embed.Config, error) { return c, nil },
  })
  ```

  After that, `plasticity embed enable mymodel --model X --opt dim=256`
  works. The built-in `hugot` and `ollama` providers register themselves the
  same way, in `internal/embed/hugot.go` and `internal/embed/ollama.go`.

## Layout

```
cmd/plasticity       CLI (cobra)
internal/model       Node, Edge, Params
internal/store       SQLite schema, transactions, FTS5 search
internal/plasticity  pure synaptic dynamics
internal/retrieval   spreading activation, personalized PageRank
internal/embed       embedders: hugot (in-process ONNX), Ollama
internal/engine      orchestration: recall, learning, consolidation, export
```
