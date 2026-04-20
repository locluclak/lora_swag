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
- **VM-Friendly**: Simplified Hydra configuration without complex timestamped subdirectories.

## 📊 Datasets & Metrics

### In-Distribution (ID): MNLI
- **Dataset**: `GLUE / MNLI` (Multi-Genre Natural Language Inference).
- **Task**: 3-class classification (Entailment, Neutral, Contradiction).
- **Metric**: **Accuracy**. measures how many labels the model predicts correctly on the `validation_matched` split.

### Out-of-Distribution (OOD): RTE
- **Dataset**: `GLUE / RTE` (Recognizing Textual Entailment).
- **Purpose**: Evaluates how the model behaves on "unknown" distributions. Even though it's NLI, RTE's distribution and binary labels differ from MNLI, providing a standard benchmark for distribution shift.
- **Metric**: **AUROC (Area Under ROC Curve)**.
    - We calculate the **Predictive Entropy** $H(y|x) = -\sum p(y|x) \log p(y|x)$ for every sample in both MNLI and RTE.
    - A well-calibrated Bayesian model should have **higher entropy** (more uncertainty) on the OOD data (RTE).
    - **AUROC** measures the model's ability to distinguish ID from OOD samples based solely on this entropy score. An AUROC of 1.0 means perfect separation.

## 📈 Bayesian Methodology

### The SWAG Algorithm
SWAG approximates the posterior distribution of the LoRA weights as a Gaussian $\mathcal{N}(\theta_{SWA}, \Sigma_{SWAG})$.
1. **Mean ($\theta_{SWA}$)**: The running average of weights collected during the final 25% of training.
2. **Variance ($\Sigma_{SWAG}$)**: A combination of a diagonal variance (from second moments) and a low-rank covariance matrix (storing the last $K$ weight deviations).

### Ensemble Inference
During `eval.py`, we perform **Bayesian Model Averaging**:
- We sample $T=10$ different weight sets from the Gaussian posterior.
- We run the input $x$ through each sampled model to get probabilities $p(y|x, \theta_t)$.
- We compute the **Ensemble Probability**: $\bar{p} = \frac{1}{T} \sum_{t=1}^T p(y|x, \theta_t)$.
- **Accuracy** and **Entropy** are calculated using this averaged probability $\bar{p}$, which is typically more robust and better calibrated than a single point-estimate model.

## 🛠️ Installation & Usage

```bash
pip install -r requirements.txt

# 1. Train on MNLI (Starts SWAG at 75% progress)
python train.py

# 2. Evaluate on MNLI (ID) and RTE (OOD)
python eval.py
```

## ⚙️ Configuration

Managed in `configs/config.yaml`. 

| Parameter | Description | Default |
|-----------|-------------|---------|
| `dataset_percentage` | Subsample ID training data (e.g., 0.5 = 50%) | `0.5` |
| `swag_start_ratio` | Progress ratio to start collection (Bayesian phase) | `0.75` |
| `swag_total_samples` | Exact number of samples to collect for the posterior | `30` |
| `swag_scale` | Scaling factor for the sampled weight noise | `1.0` |

---
*Developed for efficient, reproducible research on virtual machines and Kaggle/AutoDL environments.*
