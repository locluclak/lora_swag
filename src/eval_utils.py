import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import Tuple, List

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
    preds = avg_probs.argmax(dim=-1).numpy()
    
    acc = accuracy_score(all_labels, preds)
    entropies = -(avg_probs * torch.log(avg_probs + 1e-10)).sum(dim=-1).numpy()
    
    return acc, avg_probs.numpy(), all_labels, entropies

def compute_ood_metrics(id_entropies, ood_entropies):
    labels = np.zeros(len(id_entropies) + len(ood_entropies))
    labels[len(id_entropies):] = 1 
    scores = np.concatenate([id_entropies, ood_entropies])
    auroc = roc_auc_score(labels, scores)
    return auroc
