import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
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
