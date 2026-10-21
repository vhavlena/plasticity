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

All commands print JSON. The CLI is `plasticity` (on PATH in Bash). Run it with
the `--db` prefix given at session start (`plasticity --db '...'`): Bash commands
do not see the plugin's scope option, so without it they may use another graph.
Every `plasticity` below stands for that prefix.

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
  --importance 0.7 --tags storage,db --session <SESSION_ID> [--link <ID>:supports]
```

- **Title**: one specific, self-contained line (it is searched with the highest weight).
- **Rationale** (`-r`): why — the most valuable part; never omit it for decisions.
- **Importance** (`--importance`, 0–1, default 0): how much it matters if this is
  forgotten. It is the only signal for a new neuron (usage is learned later), and
  it raises the neuron in recall ranking. Always set it:

  | importance | for |
  |---|---|
  | 0.8–1.0 | user preferences and corrections; decisions or root causes that were costly to reach or costly to get wrong |
  | 0.4–0.7 | useful project facts, working procedures, ordinary decisions |
  | 0.1–0.3 | minor details, open questions, context that is nice to have |

  Adjust it later with `plasticity update ID --importance X` when its weight changes.
- **One idea per neuron.** Several ideas → several `add` calls; link them with
  `--link ID:kind` (`supports`, `derives_from`, `contradicts`, `part_of`, `explicit`).
- `--session` links the neuron to what this session already used or created.
- **Cross-project knowledge**: memories from other projects are recalled less
  readily. Tag knowledge that holds everywhere (your preferences, tool tips,
  shared infrastructure such as servers) with `--tags global`.

**If the result has `duplicate_of`**, nothing was inserted: an equivalent neuron
exists. Refine it instead (`plasticity update ID -c "..." -r "..."`), or link to it.
Use `--force` only if it is genuinely different.

**When new knowledge replaces old** (a refuted fix, a revised decision, the
final version of an iterated design), mark the old neuron superseded. It is then
never injected again; recall finds the new one in its place, and the history
stays searchable:

```sh
plasticity add -t decision --title "..." -c "..." --supersedes OLD_ID --session <SESSION_ID>
plasticity update OLD_ID --superseded-by NEW_ID     # after the fact; --clear-superseded undoes it
```

The `add` result lists `similar` existing neurons: check whether one is now
outdated. Supersede only what is **wholly** replaced. If an old neuron is still
partly true, update it instead. Example: a neuron diagnosing why disequations
were dropped, whose proposed fix was later refuted, keeps its valid diagnosis;
update it to point to the neuron with the correct fix rather than superseding it.

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
If one is wrong or outdated, fix it with `plasticity update ID ...`, mark it
`--superseded-by` its replacement, or `plasticity delete ID` if it is simply false.

## 3. Report what helped

When memories helped — injected ones or ones you found yourself — report them
right after they helped, once per task; this is what teaches the graph which
knowledge belongs together:

```sh
plasticity used --session <SESSION_ID> <ID> [<ID> ...]
```

Report only memories that actually influenced your work, not everything shown.
Do not save reports up for the end of a session: memories reported within a
few prompts of each other are linked as parts of the same task, while one
batch for a whole session would link unrelated topics.
