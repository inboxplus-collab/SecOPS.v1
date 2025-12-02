# SecOPS.v1

## Transformer Decoder with Mixture-of-Experts Overview

### 1. Input and Embedding Layer
* **Tokenization:** Raw text is split into sub-word tokens (e.g., Byte-Pair Encoding).
* **Token Embedding:** Each token is mapped to a learned embedding vector.
* **Positional Encoding:** A positional vector is added so attention can reason about order:

  $$\text{Input Vector}_i = \text{Embedding}_i + \text{Positional\_Encoding}_i$$

### 2. Decoder Stack (L Blocks)
Each block processes the previous hidden state \(\mathbf{x}_l\) to produce \(\mathbf{x}_{l+1}\), with residual connections after both attention and the sparsely gated MoE layer.

1. **Pre-Attention LayerNorm** on \(\mathbf{x}_l\).
2. **Masked Multi-Head Self-Attention (MMHSA):**
   * Derives \(\mathbf{Q}, \mathbf{K}, \mathbf{V}\) projections.
   * Applies a causal mask so position \(i\) only attends to positions \(j \le i\).
   * Produces a context-aware attention output.
3. **Residual Add:** \(\mathbf{x}' = \mathbf{x}_l + \text{Attention}(\dots)\).
4. **Pre-MoE LayerNorm** on \(\mathbf{x}'\).
5. **Sparsely-Gated Mixture-of-Experts (MoE):**
   * **Routing:** A gating network scores experts with a softmax:

     $$\mathbf{g} = \text{Softmax}(\mathbf{x}' \cdot \mathbf{W}_{\text{gate}})$$

   * **Top-k Selection:** Only the top \(k\) experts (commonly \(k=2\)) run per token.
   * **Expert Computation:** Selected experts (small FFNs) transform the token, and their weighted sum forms the MoE output:

     $$\text{MoE\_Output} = \sum_{i \in \text{Top}k} \mathbf{g}_i \cdot \text{Expert}_i(\mathbf{x}')$$

6. **Residual Add:** \(\mathbf{x}_{l+1} = \mathbf{x}' + \text{MoE\_Output}\).

### 3. Output Generation
* **Final LayerNorm** on \(\mathbf{x}_L\).
* **Linear Projection:** Maps the normalized state to vocabulary logits.
* **Softmax:** Converts logits into next-token probabilities:

  $$P(\text{token}\mid\text{context}) = \text{Softmax}(\text{Logits})$$

* **Token Sampling:** Generates the next token via greedy, nucleus, or similar sampling strategies.
