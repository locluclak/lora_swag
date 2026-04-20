# Easy SWAG-LoRA

A modernized, simplified, and high-performance implementation of **Stochastic Weight Averaging Gaussian (SWAG)** for **Low-Rank Adaptation (LoRA)**. This project enables Bayesian uncertainty estimation for Large Language Models (LLMs) with minimal overhead.

## 🚀 Key Features

- **Modern LLM Support**: Optimized for the **Qwen-2.5-1.5B** family (lightweight and powerful).
- **Efficient Bayesian Inference**: Collects SWAG statistics (mean, variance, low-rank covariance) only for LoRA layers. 
- **Surgical Storage**: Saves only a few MBs of adapter stats instead of multi-GB model checkpoints.
- **Automated Workflow**: 
    - Starts SWAG collection automatically at **75%** of training.
    - Transitions to a **constant learning rate** during the sampling phase.
    - Collects a determined number of samples (default: **30**) across the final 25% of steps.
- **Robust OOD Detection**: Built-in evaluation for In-Distribution (**MNLI**) and Out-of-Distribution (**RTE**) tasks.
- **VM-Friendly**: Simplified Hydra configuration without complex timestamped subdirectories, perfect for Kaggle/Colab/AutoDL automation.

## 🛠️ Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

**Requirements:** `torch`, `transformers`, `peft`, `datasets`, `hydra-core`, `tqdm`, `scikit-learn`, `accelerate`.

## 📈 Methodology

This implementation follows a rigorous Bayesian training schedule:
1. **Pre-training (0% - 75%)**: Standard LoRA fine-tuning with a linear learning rate schedule and warmup.
2. **Bayesian Phase (75% - 100%)**: 
    - The learning rate is frozen at its 75% value (Constant LR).
    - The model explores the local minima.
    - 30 weight samples are collected at regular step intervals to build the SWAG Gaussian posterior.

## 📖 Usage

### 1. Training & Collection
Train the LoRA adapter and build the SWAG posterior stats:
```bash
python train.py
```
*Outputs are saved to `./outputs/`:*
- `lora_adapter/`: The fine-tuned LoRA weights.
- `swag_stats.pt`: The compact SWAG statistics (Mean, SqMean, Cov).

### 2. Evaluation
Evaluate the deterministic LoRA base model vs. the SWAG Ensemble:
```bash
python eval.py
```
*Metrics reported:*
- **ID Accuracy**: Performance on the MNLI matched validation set.
- **OOD AUROC**: Ability to detect distribution shift using RTE via predictive entropy.

## ⚙️ Configuration

All settings are managed in `configs/config.yaml`. 

| Parameter | Description | Default |
|-----------|-------------|---------|
| `dataset_percentage` | Use a subset of data for faster training | `0.5` (50%) |
| `swag_start_ratio` | When to start Bayesian collection | `0.75` |
| `swag_total_samples` | Number of samples to collect | `30` |
| `num_epochs` | Total training epochs | `3` |
| `learning_rate` | Peak learning rate | `2e-4` |

To override settings via CLI:
```bash
python train.py experiment.dataset_percentage=0.1 experiment.num_epochs=1
```

## 🛡️ Compatibility Notes

- **Hardware**: Fully compatible with **NVIDIA P100** (Kaggle) and newer. Includes automatic `float32` casting for `BFloat16` tensors to prevent NumPy/Type errors on older GPUs.
- **Storage**: The `swag_stats.pt` is decoupled from the base model, making it easy to share and deploy.

---
*Based on the original SWAG-LoRA research, modernized for 2026 LLM standards.*
