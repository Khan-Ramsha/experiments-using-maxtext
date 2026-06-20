## Experiments using maxtext

Objective of these experiments was to explore and evaluate Large Language Model (LLM) training using MaxText across different hardware backends (CPU, GPU, and TPU) and model architectures (Dense and Mixture-of-Experts).

### About MaxText

MaxText is Google's high-performance, open-source LLM training framework built on JAX and XLA. It is designed to efficiently train and serve large language models on TPUs and GPUs while providing reference implementations of modern architectures such as Qwen, Llama, Mistral, DeepSeek, and Mixture-of-Experts (MoE) models.

The experiments consisted of three major tasks:

Task 1: Understanding MaxText Data Formats

I explored the MaxText documentation and codebase to understand:

- Supported dataset formats and input pipelines
- Differences between dataset types
- Advantages and trade-offs of each approach
- Configuration parameters required for dataset selection

A summary of findings is available in the [document](https://docs.google.com/document/d/1_bWr3cJCFDDScIspYEVpgFat8taDgVD9NEM2zIkvgsY/edit?usp=sharing)

Task 2: Dense Model Experiments (Qwen)

Using synthetic data, I trained models across 3 different backends (CPU, GPU, TPU):
- Qwen 0.6B
- Scaled Qwen (~1B)

Task 3: Mixture-of-Experts (DeepSeek)

Using synthetic data, I configured a DeepSeek-based MoE model, scaled it below 1B parameters and ran experiments across all 3 backends.

Please read the google document for my detailed writeup:
[Document](https://docs.google.com/document/d/1_bWr3cJCFDDScIspYEVpgFat8taDgVD9NEM2zIkvgsY/edit?usp=sharing)

## Repository Structure

```text

├── Deepseek_MoE
│   ├── CPU
│   │   ├── deepseek_on_CPU.ipynb
│   │   └── training_log_deepseek_cpu_v2.txt
│   │
│   ├── GPU
│   │   ├── MoE_scaled_on_GPU (3).ipynb
│   │   └── training_log_deepseek_gpu_v2.txt
│   │
│   └── TPU
│       ├── dense_scaled_on_TPU.ipynb
│       └── training_log_deepseek_tpu_v2.txt
│
├── Qwen3-0.6B-Dense
│   ├── CPU
│   │   ├── qwen3-on-cpu.ipynb
│   │   └── training_log_cpu_0.6.txt
│   │
│   ├── GPU
│   │   ├── original_qwen_0.6
│   │   │   ├── qwen3-0.6B_on_GPU.ipynb
│   │   │   └── training_log_gpu_0.6.txt
│   │   │
│   │   └── scaled_up
│   │       ├── dense_scaled_on_GPU.ipynb
│   │       └── training_log_gpu_1b.txt
│   │
│   └── TPU
│       ├── original_qwen_0.6
│       │   ├── qwen3-on-TPU.ipynb
│       │   └── training_log_tpu_0.6b.txt
│       │
│       └── scaled_up
│           ├── scaled_on_TPU.ipynb
│           └── training_log_tpu_1b (1).txt
│
└── README.md
```
