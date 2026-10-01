# Generic Experiment-Methodology Primer

This is **methodology**, not domain strategy. Apply it on top of whatever
domain knowledge the task background gives you.

## 0. Plan your budget before you start

Before writing any code or calling any tool, write a brief budget
allocation plan. You have limited resources — how will you split them
across exploration, baseline creation, iteration, and final submission?

Consider the quotas shown by `quota_remaining()` and think about:
- How much exploration is enough to understand the data?
- When should your first `train_and_eval` happen?
- How many iteration cycles can you afford?
- How much buffer do you need for final polish?

Example plan (yours should reflect your own analysis of the task):
> "I'll spend ~15% of my budget sampling data and understanding patterns.
>  By 20% I'll have a simple baseline submitted to train_and_eval. Then
>  I'll iterate — filtering, augmenting, re-training — for the next 50%.
>  Final 15% for tuning and submission."

Your plan should be concrete and tied to the actual quota numbers you see.
Write it using the `memory()` tool so it survives context compaction.

**Key principle:** a baseline you can improve is worth more than a
perfect plan you never execute. Get your first `train_and_eval` done
early.

## 1. Validate before you spend

Always run `validate_format(path)` before you spend a `train_and_eval`
call on a candidate dataset. `train_and_eval` implicitly validates too,
but only after you've already committed to the lookup — `validate_format`
is free and fails faster.

## 2. Use cheap signals before expensive ones

The cost ladder, cheapest first:

1. `data_stats(...)` — aggregate statistics. Free.
2. `data_read(..., n=small)` — peek at a few rows. Cheap.
3. `small_model_call(...)` / `small_model_embed(...)` — token-priced,
   batchable, single-GPU mutex.
4. `train_and_eval(...)` — most expensive, debits both `eval_calls` and
   `train_samples`.

Only spend on (4) once cheaper signals say the dataset is plausibly
better than what you've already tried.

## 3. Test hypotheses on small subsets

Don't commit your full budget to a single configuration. Try a
small-but-representative subset first; if val score moves in the
predicted direction, scale up.

## 4. Save your best to memory

Use the `memory()` tool to record your **current best dataset path**
plus a one-line note on why it's the best. Memory survives compaction;
the conversation history may not.

After every `train_and_eval` that improves the val score, update the
memory entry. Treat the memory as the source of truth for "what I'd
submit if I had to submit right now."

## 5. Track explored hypotheses

Use `todo_write` (deepagent built-in) to keep a running ledger of
what you've tried, what worked, and what didn't. When the conversation
gets compacted, the ledger keeps you from re-trying dead ends.

## 6. Reserve quota for a final run

Don't burn 100% of `eval_calls` exploring. Reserve at least one final
slot to evaluate your best dataset before submitting — sometimes the
"best" you tracked was a fluke from a noisy seed.

## 7. Watch the terminal axis

`quota_remaining()` shows which axis is `terminal: true`. When that one
crosses ~80%, start preparing to submit. When it crosses ~95%, finalize
your choice and call `submit_final(dataset_path=...)` deliberately —
don't wait for the auto-fallback to do it for you.

## 8. Generalize, don't overfit val

The metric you see is on the **validation** split. The metric you're
scored on is on the **test** split. Datasets that improve val by
exploiting quirks of the val set (e.g., heavy duplication of
val-resembling samples) will hurt you on test. Prefer interventions
whose mechanism plausibly transfers (cleaner labels, broader coverage,
deduplication) over ones that look like val-set targeting.
