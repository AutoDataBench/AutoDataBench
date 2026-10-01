# External resources

The repository does not contain training data, MTEB datasets, or model weights.

Place the source pool at data/retrieval_v1/train.jsonl, or call load_task with a
data_root argument. The expected format is documented in format_spec.md.

Models are downloaded by their Hugging Face IDs:

- nreimers/MiniLM-L6-H384-uncased
- Qwen/Qwen3-4B-Instruct-2507
- Qwen/Qwen3-Embedding-0.6B

MTEB downloads the configured evaluation datasets through the Hugging Face
cache. Set HF_HOME to control their storage location.

