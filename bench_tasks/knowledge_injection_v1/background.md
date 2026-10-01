# Knowledge Injection Task Background

## Task Overview

This task evaluates whether an agent can construct high-quality OPSD (Offline Privileged-context Supervised Distillation) training data to inject post-1930 knowledge into Talkie-13B, a language model pretrained only on text up to approximately 1930. The agent's only control is the training data submission; the base model, training code, evaluation code, and hyperparameters are all fixed.

## Talkie-13B Model

Talkie-13B is a 13-billion parameter causal language model with a custom architecture (`TalkieForCausalLM`). Key characteristics:

- **Pretraining cutoff**: ~1930. The model has strong knowledge of pre-1930 history, literature, science, and culture, but lacks knowledge of post-1930 events, people, and discoveries.
- **Architecture**: 40 layers, 5120 hidden size, 40 attention heads, 65K vocabulary.
- **Context window**: 2048 tokens maximum.
- **Special tokens**: `<|end|>` (EOS), `
