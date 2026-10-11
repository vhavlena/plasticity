---
name: memory
description: Long-term memory (plasticity knowledge graph). Use when a decision is made, a root cause or non-obvious fact is discovered, a procedure works, or the user states a preference — to store it as a neuron; before design decisions, in unfamiliar code, when stuck on an error, or when past decisions, rationale or project context may exist — to recall from memory; and when memories helped — to report them as used.
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

## 2. Recall knowledge when you need it

Each user prompt gets an automatic recall, but the knowledge you need often
only becomes clear while working. Recall yourself:

- **before a design decision** or choosing between alternatives: was this decided before, and why?
- **when starting work in an unfamiliar area**, module or file: conventions, pitfalls, past decisions;
- **when stuck on an error** or unexpected behavior: recall with the error message, since root causes are stored;
- **before re-investigating** something that may have been solved in an earlier session;
- when the user refers to earlier work ("like last time", "as we decided").

```sh
plasticity recall "why does the hook time out on large repos" --session <SESSION_ID>   # associative: follows links
plasticity search "WAL"                                                                 # keywords only
plasticity get <ID>                                                                      # a neuron with its links
```

- Ask a **specific question** in natural language, or paste the key line of an
  error. A vague query ("storage") seeds weakly and returns little.
- Always pass `--session`: your recall then takes part in learning and is
  measured. Use `--no-record` only for browsing that has nothing to do with the task.
- An empty result is an answer too: nothing is known, so proceed and consider
  storing what you find.

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
