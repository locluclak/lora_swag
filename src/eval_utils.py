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

def plot_combined_reliability_diagram(id_probs, id_labels, ood_probs, ood_labels, n_bins=10, title="Reliability Diagram: ID vs OOD", path=None):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    def get_bin_stats(probs, labels):
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels)
        
        bin_accs = []
        bin_confs = []
        bin_counts = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            bin_counts.append(np.sum(in_bin))
            if np.any(in_bin):
                bin_accs.append(np.mean(accuracies[in_bin]))
                bin_confs.append(np.mean(confidences[in_bin]))
            else:
                bin_accs.append(0)
                bin_confs.append(0)
        return np.array(bin_accs), np.array(bin_confs), np.array(bin_counts)

    id_accs, id_confs, id_counts = get_bin_stats(id_probs, id_labels)
    ood_accs, ood_confs, ood_counts = get_bin_stats(ood_probs, ood_labels)

    # Chuẩn hóa số lượng mẫu trong mỗi bin để vẽ biểu đồ mật độ (optional)
    id_proportions = id_counts / np.sum(id_counts)
    ood_proportions = ood_counts / np.sum(ood_counts)

    fig, ax1 = plt.subplots(figsize=(10, 8))

    # 1. Vẽ các cột Accuracy cho ID
    ax1.bar(bin_lowers, id_accs, width=1/n_bins, align='edge', alpha=0.4, 
            edgecolor="blue", color="blue", label="ID Accuracy per Bin")
    
    # 2. Đường chéo lý tưởng
    ax1.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")

    # 3. Vẽ phân phối Confidence (Mật độ mẫu rơi vào bin)
    # Dùng trục phụ hoặc vẽ đè lên để thấy sự khác biệt phân phối
    ax2 = ax1.twinx()
    ax2.step(bin_lowers, id_proportions, where='post', color="blue", alpha=0.7, label="ID Confidence Dist.")
    ax2.step(bin_lowers, ood_proportions, where='post', color="red", linestyle="--", alpha=0.7, label="OOD Confidence Dist.")
    
    ax1.set_xlabel("Confidence")
    ax1.set_ylabel("Accuracy (Only for ID)")
    ax2.set_ylabel("Proportion of Samples")
    
    plt.title(title)
    # Hợp nhất legend từ 2 trục
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    
    plt.grid(alpha=0.2)
    if path:
        plt.savefig(path)
    else:
        plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()

# Cách sử dụng với biến của bạn:
# plot_combined_reliability_diagram(swag_id_probs, swag_id_labels, swag_ood_probs, swag_ood_labels, title="SWAG LoRA: ID vs OOD Calibration")