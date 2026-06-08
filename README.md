# BitNet Context Expansion Experiments

This repository contains my research and experiments with expanding the context window of BitNet from 4k to 8k (and attempting 16k), along with some exploration into mechanistic interpretability.

## 🎯 Project Aims

1. **Context Expansion**: Successfully scale the context window of the BitNet model from its default 4k up to 8k (with experiments targeting 16k).
2. **Mechanistic Interpretability**: Analyze the model to understand its internal attention mechanisms and representations more deeply.
3. **Quantization**: Apply quantization to the context-expanded model to reduce its memory footprint and improve efficiency.

## 📉 Results and the "Failure"

While the **context expansion itself worked fine**—the model successfully adapted to the larger context windows during training (as shown in the `8k_training` logs and benchmarking scripts)—the project ultimately hit a major roadblock during the quantization phase.

**The quantization of the expanded model did not work well.** After quantization, the model degraded significantly, and I was unable to get any coherent responses from it. As a result, the fully quantized, long-context model is not functional in its current state. 

Despite this, I am open-sourcing the training recipes, the successful context-expansion scripts, and the mechanistic interpretability tools to showcase the research and the components that did work prior to the quantization failure.

## 📂 Repository Contents

* **`8k_training/`**: Scripts and logs detailing the training process and Yarn RoPE implementation to expand the context to 8k.
* **`16k_attempt/`**: Configurations, training logs, and loss curves from the attempt to push the context window to 16k.
* **`benchmarks_scripts/`**: A suite of benchmarking tools (e.g., Needle in a Haystack, Perplexity evaluations) used to evaluate the context expansion.
* **`interpretability/`**: Scripts (like `attention_profile.py`) used to visualize and understand the model's internal attention profiles.
* **`rope_analysis/`**: Analysis scripts and visualizations investigating the RoPE (Rotary Position Embedding) scaling behavior.
* **`quantize_bitnet.py`**: The quantization script that was used for the final, unsuccessful quantization phase.
