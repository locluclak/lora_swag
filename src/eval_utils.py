import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import Tuple, List
import torch.nn.functional as F

def evaluate(model, dataloader, device, num_samples=1, scale=1.0, use_cov=True):
    model.eval()
    all_probs = []
    all_labels = []
    
    # Check if we are sampling or just doing a single pass
    is_swag = hasattr(model, 'sample') and num_samples > 1
    
    with torch.no_grad():
        for i in range(num_samples):
            if is_swag:
                model.sample(scale=scale, use_cov=use_cov)
            
            sample_probs = []
            for batch in tqdm(dataloader, desc=f"Eval Sample {i+1}/{num_samples}", leave=False):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                probs = torch.softmax(outputs.logits, dim=-1)
                sample_probs.append(probs.cpu())
                if i == 0:
                    all_labels.extend(batch["labels"].cpu().numpy())
            
            all_probs.append(torch.cat(sample_probs, dim=0))
    
    avg_probs = torch.stack(all_probs).mean(dim=0)
    # Ensure float32 for numpy compatibility (P100 doesn't support bfloat16 numpy well)
    avg_probs_f32 = avg_probs.float()
    preds = avg_probs_f32.argmax(dim=-1).numpy()
    
    acc = accuracy_score(all_labels, preds)
    entropies = -(avg_probs_f32 * torch.log(avg_probs_f32 + 1e-10)).sum(dim=-1).numpy()
    
    # NLL Calculation
    labels_tensor = torch.tensor(all_labels)
    nll_loss = F.nll_loss(torch.log(avg_probs_f32 + 1e-10), labels_tensor)
    nll_value = nll_loss.item()
    return acc, avg_probs_f32.numpy(), all_labels, entropies, nll_value

def compute_ood_metrics(id_entropies, ood_entropies):
    labels = np.zeros(len(id_entropies) + len(ood_entropies))
    labels[len(id_entropies):] = 1 
    scores = np.concatenate([id_entropies, ood_entropies])
    auroc = roc_auc_score(labels, scores)
    return auroc

def compute_prr(accuracies, uq_scores):
    """
    Compute the Prediction-Rejection Ratio (PRR).
    accuracies: list of 1s (correct) and 0s (wrong)
    uq_scores: uncertainty scores (higher = more uncertain)
    """
    n = len(accuracies)
    if n == 0: return 0.0
    
    accuracies = np.array(accuracies)
    uq_scores = np.array(uq_scores)
    
    # Initial accuracy (A_Random baseline is constant accuracy)
    initial_acc = np.mean(accuracies)
    
    def get_auc(sorted_accs):
        # Calculate accuracy at each rejection step
        # Step k means we have rejected k samples and have n-k left
        res = []
        for k in range(n):
            remaining_acc = np.mean(sorted_accs[k:])
            res.append(remaining_acc)
        return np.mean(res)

    # 1. A_UQ: Sort by UQ scores Descending (reject most uncertain first)
    uq_indices = np.argsort(-uq_scores)
    a_uq = get_auc(accuracies[uq_indices])
    
    # 2. A_Random: Constant area
    a_random = initial_acc
    
    # 3. A_Oracle: Sort by accuracies Ascending (reject all 0s first)
    oracle_indices = np.argsort(accuracies)
    a_oracle = get_auc(accuracies[oracle_indices])
    
    # PRR Formula
    if a_oracle == a_random:
        return 0.0
    
    prr = (a_uq - a_random) / (a_oracle - a_random)
    return prr

def calculate_ece(probs, labels, n_bins=10):
    """
    Tính Expected Calibration Error (ECE)
    probs: mảng xác suất (N, Num_Classes)
    labels: mảng nhãn thực tế (N,)
    n_bins: số lượng thùng để chia (mặc định là 10)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # Lấy độ tự tin (max prob) và dự đoán (argmax)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Tìm các mẫu nằm trong thùng hiện tại
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin) # Trọng số của thùng (|Bm| / N)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            # ECE += (|Bm| / N) * |Acc(Bm) - Conf(Bm)|
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)

    return ece

def plot_entropy_dist(id_entropies, ood_entropies, title="Entropy Distribution", path=None):
    plt.figure(figsize=(10, 6))
    
    # Vẽ phân phối cho ID
    sns.kdeplot(id_entropies, fill=True, label='In-Distribution (ID)', color='blue', bw_adjust=0.5)
    
    # Vẽ phân phối cho OOD
    sns.kdeplot(ood_entropies, fill=True, label='Out-of-Distribution (OOD)', color='red', bw_adjust=0.5)
    
    plt.xlabel('Entropy')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    if path:
        plt.savefig(path)
    else:
        plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()

def plot_confidence_dist(id_probs, ood_probs, title="Confidence Distribution", path=None):
    plt.figure(figsize=(10, 6))
    id_conf = np.max(id_probs, axis=1)
    ood_conf = np.max(ood_probs, axis=1)
    
    sns.kdeplot(id_conf, fill=True, label='In-Distribution (ID)', color='blue', bw_adjust=0.5)
    sns.kdeplot(ood_conf, fill=True, label='Out-of-Distribution (OOD)', color='red', bw_adjust=0.5)
    
    plt.xlabel('Confidence (Max Probability)')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    if path:
        plt.savefig(path)
    plt.close()

def plot_combined_reliability_diagram(id_probs, id_labels, n_bins=15, title="Reliability Diagram", path=None):
    """
    Plots a clean reliability diagram for ID data.
    - Top horizontal bars for accuracy
    - Gap offset from the diagonal (Identity line)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_centers = (bin_lowers + bin_uppers) / 2

    confidences = np.max(id_probs, axis=1)
    predictions = np.argmax(id_probs, axis=1)
    accuracies = (predictions == id_labels)
    
    bin_accs = []
    bin_confs = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if np.any(in_bin):
            bin_accs.append(np.mean(accuracies[in_bin]))
            bin_confs.append(np.mean(confidences[in_bin]))
        else:
            bin_accs.append(0.0)
            bin_confs.append(bin_lower + (bin_upper - bin_lower)/2)

    fig, ax = plt.subplots(figsize=(8, 8))

    # 1. Perfect calibration line
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")

    # 2. Accuracy Bars (Top Horizontal Bar style)
    # We draw bars from 0 to accuracy, but often stylized as just the top
    # To show the "Gap", we fill the area between Accuracy and the Diagonal
    for i in range(n_bins):
        # The Accuracy Bar
        ax.bar(bin_lowers[i], bin_accs[i], width=1/n_bins, align='edge', 
               color='blue', alpha=0.3, edgecolor='blue')
        
        # The Gap (Difference between confidence/diagonal and accuracy)
        # Typically, if Acc < Conf, it's overconfident (Red gap)
        # If Acc > Conf, it's underconfident (Green gap)
        if bin_accs[i] > 0 or (bin_lowers[i] > 0.5): # Only draw if bin has data
            gap_color = "red" if bin_centers[i] > bin_accs[i] else "green"
            ax.bar(bin_lowers[i], bin_centers[i] - bin_accs[i], bottom=bin_accs[i], 
                   width=1/n_bins, align='edge', color=gap_color, alpha=0.5, label="Gap" if i == 0 else "")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(loc="upper left")
    
    plt.grid(alpha=0.2)
    if path:
        plt.savefig(path)
    plt.close()

# Cách sử dụng với biến của bạn:
# plot_combined_reliability_diagram(swag_id_probs, swag_id_labels, swag_ood_probs, swag_ood_labels, title="SWAG LoRA: ID vs OOD Calibration")