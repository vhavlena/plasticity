---
name: flag
description: Record an observation about plasticity memory for evaluation, e.g. /plasticity:flag miss it forgot we pinned schema 6
argument-hint: miss|noise|helped|other <what happened>
disable-model-invocation: true
allowed-tools: Bash(plasticity *)
---

The user is evaluating the long-term memory and flags an observation: $ARGUMENTS

1. Kind: the first word if it is `miss`, `noise`, `helped` or `other`; otherwise
   infer it:
   - `miss`: something stored was not recalled when it should have been;
   - `noise`: an injected memory was irrelevant;
   - `helped`: a memory clearly helped;
   - `other`: anything else.
2. Neurons concerned: IDs of the memories involved, if you can tell them from the
   injected `<plasticity-memories>` or your recalls in this session. For a miss,
   find the memory that should have been recalled with
   `plasticity search "<keywords>"` and name it too.
3. Run:
   ```sh
   plasticity note -k <kind> --session <SESSION_ID> [--id <ID> ...] "<the user's observation, plus one line of context: the prompt or task it was about>"
   ```
   (the session ID was given at session start).

Confirm in one line. Do not change the memory itself: the note is evidence for
tuning. If the user also wants a wrong memory fixed, do that separately.
