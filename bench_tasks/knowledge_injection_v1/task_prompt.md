# Knowledge Injection Data Construction Task

You are a knowledge injection training data construction agent. Your goal is not to answer evaluation questions directly, but to use the publicly accessible context to construct high-quality offline OPSD (Offline Privileged-context Supervised Distillation) training data for Talkie-13B. Talkie's pretraining corpus cuts off around 1930; you need to help it learn new facts from after 1930 while avoiding damage to its existing historical knowledge.

## Accessible Resources

- `{SOURCES_PATH}`: 1,000 Wikipedia summaries directly corresponding to target new knowledge, without evaluation questions, answers, or options. These are your core target set when constructing knowledge injection data.
- `{POOL_PATH}`: 542,970 post-1930 Wikipedia source contexts, forming the complete retrieval corpus. Each record mainly contains `qid`, `subject`, `wikipedia_title`, `summary_extract`, temporal and domain metadata; the 1,000 core sources are also included. You can retrieve supplementary background material about the same entity, event, relationship, or related topics. A small number of records have no usable summary and should be skipped.
- Your working directory and local tools, models, and retrieval components permitted by the Harness.

You cannot access or infer private evaluation files, cannot modify training, validation, and evaluation scripts, and cannot modify the base model or fixed hyperparameters.

## Evaluation Objective and Primary Metric

After your submission is trained, it will be evaluated on two independent private benchmarks:

- **Novel benchmark**: 1,000 post-1930 new knowledge questions, measuring whether the model truly acquired the target new knowledge; balanced across ten decades from 1930s–2020s, with 100 questions per decade.
- **Retention benchmark**: 4,400 pre-1930 historical knowledge questions, measuring whether the model retains its original capabilities after knowledge injection.

Both parts use four-choice evaluation. The evaluator computes teacher-forced conditional likelihood for each complete answer option and selects the option with the highest mean token log-likelihood as the prediction, without relying on whether the model can canonically output A, B, C, or D.

The primary score is not a micro average computed directly over 5,400 questions, but an equal-weight average of novel and retention accuracy:

```text
macro_normalized = (novel_normalized + retention_normalized) / 2
```

Therefore your goal is to improve both new knowledge acquisition and old knowledge retention. Simply pursuing a large number of new knowledge training samples, causing retention to drop significantly, may not improve the final score. You will not see private questions, options, or answers, and can only construct generalizable training data based on public sources.

## Final Deliverable

Submit only one UTF-8 JSONL file, with each row required to be:

```json
{
  "context_qid": "Q123",
  "context": "the complete original text from the context pool",
  "question": "a factual question whose answer can be supported by this context?",
  "teacher_continuation": "a concise, direct, context-grounded answer"
}
```

`context` must exactly match the `summary_extract` of the corresponding QID in the public pool. You may add a unique `id` and `metadata` that does not contain hidden answers, but no other fields are allowed. Submit at most 10,000 rows.

## Suggested Workflow

1. Audit the public data's fields, decades, topics, entities, and domain distributions.
2. Extract reliable entity–relation–value facts from each direct source.
3. Design natural, clear, uniquely-answerable questions for the facts.
4. Retrieve potentially helpful supplementary context from the larger context pool.
5. Generate QA supported by the supplementary context itself; do not force direct source answers onto paragraphs that only mention the same entity.
6. Perform automatic quality filtering, deduplication, and coverage analysis on questions and continuations.
7. Run `scripts/validate_submission.py` and fix all errors.

## Explorable Data Selection and Retrieval Strategies

You can autonomously compare or combine:

- Exact entity name, alias, or normalized name matching;
- Supplementary background for the same person, event, or organization;
- Different facts about the same entity, to strengthen entity representation;
- Materials with different entities but the same relationship type, to provide relational patterns;
- Decade, domain, topic, or document length matching;
- BM25, TF-IDF, and other lexical retrieval;
- External embedding retrievers, such as Qwen3-Embedding-0.6B;
- Cross-encoder reranking or LLM relevance judging;
- Hybrid retrieval composed of entity recall, embedding recall, and reranker precision ranking;
- Allocate quotas by decade, entity, relationship, or domain to avoid high-frequency topics dominating the data;
- Generate multiple linguistically diverse but semantically consistent questions for the same fact;
- Filter samples based on context support strength, answer uniqueness, question naturalness, and teacher signal.

These are explorable directions, not fixed answers to copy. You should choose strategies based on data audit results and save selection rationales and statistics.

## Quality Principles

- "Mentioning the same entity" does not equal "supporting the same fact." Each QA must be supported by the currently paired context.
- Do not fabricate precise numbers, dates, casualties, weapon quantities, or election margins that do not appear in the original text or cannot be reasonably inferred.
- Continuations should be concise, avoiding hallucination, explanatory expansion, sentence fragments, and prompt pollution like `Question:`.
- Avoid Yes/No questions, meta-questions, ambiguous references, and questions with non-unique answers.
- Avoid mechanically copying a single template; linguistic diversity must not change factual meaning.
- Do not optimize only for data volume. Low-quality auxiliary context may dilute direct knowledge and harm retention.
- Check distributions to prevent high-frequency topics like World War II from dominating the training set.
- Do not obtain private questions, options, answers, or evaluation outputs through any leakage path.

## Optional Internal Quality Checks

Without accessing the private evaluation set, you can:

- Check whether key entities, numbers, and dates in answers appear in the context;
- Use NLI, LLM judge, or rules to check whether the context supports the continuation;
- Check entity and lexical correspondence between question and context;
- Verify candidate answer uniqueness and recoverability;
- Use base Talkie to compare `p(answer | context, question)` vs `p(answer | question)`, preferring samples where the context truly gives the Teacher an advantage;
- Save relevance scores, retriever sources, and rejection reasons;
- Report direct/auxiliary context ratios, QA count per QID, decade/domain/relationship distributions, duplication rates, and filter pass rates.

## Pre-Submission Checklist

- All `context_qid` come from the public pool.
- All `context` are frozen original text, without rewriting, truncation, or concatenation.
- All questions end with `?`, have reasonable length, and are semantically complete.
- All continuations are supported by the paired context.
- No duplicate IDs or duplicate `(context_qid, question)`.
- No hidden benchmark information or disallowed fields.
- Total rows do not exceed 10,000.
- `python scripts/validate_submission.py --submission YOUR_FILE.jsonl` passes successfully.

Finally, submit only compliant JSONL to the Harness. Training, LoRA merging, and private evaluation are executed by the trusted Harness after you finish your run.
