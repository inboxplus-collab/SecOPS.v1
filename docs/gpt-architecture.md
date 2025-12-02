# GPT Decoder-Only Transformer Overview

## Core Architecture
- **Input tokenization and embeddings** convert text (and other modalities in newer models) into dense vectors, with positional encodings to preserve order.
- A deep stack of identical **decoder Transformer blocks** performs masked multi-head self-attention followed by position-wise feed-forward networks (FFNs), each wrapped in residual connections and layer normalization for stable training and deep signal flow.
- **Output generation** applies a final normalization, a projection to the vocabulary, and softmax sampling to produce the next token autoregressively.

## Key Components
- **Masked multi-head self-attention:** Ensures each token attends only to previous tokens while multiple heads capture diverse relationships.
- **Feed-forward networks:** Two-layer MLPs applied per token to enrich representations after attention.
- **Residual + layer norm:** Skip connections and normalization surround attention and FFN sublayers to maintain gradients and stable activations.

## Recent Structural Enhancements
- **Mixture-of-Experts (MoE):** Replaces dense FFNs with gated expert MLPs so only a small subset of parameters activate per token, enabling trillion-parameter capacity with manageable inference cost.
- **Multimodal unification:** Specialized encoders map images or audio into token-like embeddings that the shared Transformer stack can process alongside text, enabling cross-modal attention and generation.
