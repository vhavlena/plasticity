---
name: remember
description: Store something in plasticity long-term memory, e.g. /plasticity:remember we deploy with make release because CI caches break otherwise
argument-hint: <what to remember>
disable-model-invocation: true
allowed-tools: Bash(plasticity *)
---

Run every `plasticity` command below with the `--db` prefix given at session
start (`plasticity --db '...'`), so it uses this session's memory.

Store the following in long-term memory: $ARGUMENTS

1. Decide the neuron type(s) (decision, fact, procedure, rationale, concept,
   question) and split it into one idea per neuron.
2. For each, write a specific title, the content, and the rationale (ask the
   user only if the why is essential and cannot be inferred).
3. Run `plasticity add -t <type> --title "..." -c "..." -r "..." --session <SESSION_ID>`
   (your session ID was given at session start). Link related neurons with
   `--link ID:kind`.
4. If the result reports `duplicate_of`, update the existing neuron instead.
5. Confirm briefly what was stored, with the neuron IDs.
