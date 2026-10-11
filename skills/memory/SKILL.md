---
name: memory
description: Long-term memory (plasticity knowledge graph). Use when a decision is made, a root cause or non-obvious fact is discovered, a procedure works, or the user states a preference — to store it as a neuron; when you need past decisions, rationale or project context — to search memory; and when injected <plasticity-memories> helped — to report them as used.
allowed-tools: Bash(plasticity *)
---

# Plasticity long-term memory

The memory is a graph of **neurons** (units of knowledge with a rationale)
connected by weighted links. Links between neurons used together get stronger;
unused links fade. Relevant neurons are injected automatically with every user
prompt inside `<plasticity-memories>`. Your session ID was given at session
start ("This session's ID is ..."); pass it as `--session` below.

All commands print JSON. The CLI is `plasticity` (on PATH in Bash).

## 1. Store durable knowledge

Store something when it will matter in a future session:

| type | store when |
|---|---|
| `decision` | a choice was made between alternatives — always with the rationale |
| `fact` | a non-obvious fact was discovered: root cause, API quirk, constraint, environment detail |
| `procedure` | a sequence of steps worked (build, deploy, debug recipe) |
| `rationale` | a standalone reason that supports other knowledge |
| `concept` | a project-specific concept or term and what it means here |
| `question` | an open question worth revisiting |

```sh
plasticity add -t decision --title "Use SQLite WAL for storage" \
  -c "Single file, many readers, one writer" \
  -r "Hooks run as separate processes; Badger allows one process per DB" \
  --tags storage,db --session <SESSION_ID> [--link <ID>:supports]
```

- **Title**: one specific, self-contained line (it is searched with the highest weight).
- **Rationale** (`-r`): why — the most valuable part; never omit it for decisions.
- **One idea per neuron.** Several ideas → several `add` calls; link them with
  `--link ID:kind` (`supports`, `derives_from`, `contradicts`, `part_of`, `explicit`).
- `--session` links the neuron to what this session already used or created.

**If the result has `duplicate_of`**, nothing was inserted: an equivalent neuron
exists. Refine it instead (`plasticity update ID -c "..." -r "..."`), or link to it.
Use `--force` only if it is genuinely different.

**Do not store**: transient task state, things obvious from the code or git
history, secrets/credentials, or guesses you have not verified.

## 2. Find knowledge

```sh
plasticity recall "how do hooks access the database" --no-record   # associative: follows links
plasticity search "WAL"                                             # keywords only
plasticity get <ID>                                                  # a neuron with its links
```

Injected memories are *possibly* relevant: verify before relying on them.
If one is wrong or outdated, fix it with `plasticity update ID ...` (or
`plasticity delete ID` if it is simply false).

## 3. Report what helped

When memories helped — injected ones or ones you found yourself — report them
together; this is what teaches the graph which knowledge belongs together:

```sh
plasticity used --session <SESSION_ID> <ID> [<ID> ...]
```

Report only memories that actually influenced your work, not everything shown.
