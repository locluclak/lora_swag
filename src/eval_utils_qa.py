import torch
import numpy as np
import re
import string
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def evaluate_qa(model, dataloader, device, tokenizer, num_samples=1, scale=1.0, max_gen_len=20):
    model.eval()
    all_em = []
    all_f1 = []
    all_uncertainties = []
    all_results = []

    is_swag = hasattr(model, 'sample')
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating QA"):
            # Format inputs for generation
            questions = batch["question"]
            ground_truths = batch["answer"]
            
            # For each batch item, we generate several times if num_samples > 1
            batch_probs = []
            batch_generations = []
            
            for i in range(num_samples):
                if is_swag and num_samples > 1:
                    model.sample(scale=scale, use_cov=True)
                
                # Encode questions for generation
                # We need to process one by one to avoid padding issues in generation for small models
                for q_idx, q_text in enumerate(questions):
                    prompt = f"Question: {q_text}\nAnswer:"
                    inputs = tokenizer(prompt, return_tensors="pt").to(device)
                    
                    # Generate
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_gen_len,
                        do_sample=False, # Greedily for EM/F1 metrics
                        output_scores=True,
                        return_dict_in_generate=True
                    )
                    
                    gen_tokens = outputs.sequences[0, inputs.input_ids.shape[1]:]
                    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                    
                    # Calculate uncertainty (mean negative log-prob of generated tokens)
                    # scores is a tuple of (max_new_tokens) each (1, vocab_size)
                    logits = torch.stack(outputs.scores, dim=1) # (1, gen_len, vocab_size)
                    log_probs = torch.log_softmax(logits, dim=-1)
                    
                    # Get log-probs of actual generated tokens
                    token_log_probs = log_probs[0, torch.arange(len(gen_tokens)), gen_tokens]
                    uncertainty = -token_log_probs.mean().item()
                    
                    if i == 0: # Store first sample results
                        em = exact_match_score(gen_text, ground_truths[q_idx])
                        f1 = f1_score(gen_text, ground_truths[q_idx])
                        all_em.append(em)
                        all_f1.append(f1)
                        all_uncertainties.append(uncertainty)
                        all_results.append({
                            "question": q_text,
                            "prediction": gen_text,
                            "ground_truth": ground_truths[q_idx],
                            "em": em,
                            "f1": f1,
                            "uncertainty": uncertainty
                        })

    return {
        "em": np.mean(all_em),
        "f1": np.mean(all_f1),
        "uncertainties": all_uncertainties,
        "em_list": all_em,
        "results": all_results
    }

def plot_qa_uncertainty(em_list, uncertainties, path):
    plt.figure(figsize=(10, 6))
    em_list = np.array(em_list)
    uncertainties = np.array(uncertainties)
    
    sns.kdeplot(uncertainties[em_list == 1], fill=True, label='Correct (EM=1)', color='green')
    sns.kdeplot(uncertainties[em_list == 0], fill=True, label='Incorrect (EM=0)', color='red')
    
    plt.xlabel('Uncertainty (Mean Negative Log-Prob)')
    plt.ylabel('Density')
    plt.title('Uncertainty Distribution for QA Task')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(path)
    plt.close()

def compute_prr_qa(accuracies, uq_scores):
    """
    Standard PRR for QA EM.
    """
    n = len(accuracies)
    if n == 0: return 0.0
    
    accuracies = np.array(accuracies).astype(float)
    uq_scores = np.array(uq_scores)
    
    initial_acc = np.mean(accuracies)
    
    def get_auc(sorted_accs):
        res = []
        for k in range(n):
            remaining_acc = np.mean(sorted_accs[k:]) if k < n else 0
            res.append(remaining_acc)
        return np.mean(res)

    # A_UQ: Sort by UQ scores Descending (reject high uncertainty first)
    uq_indices = np.argsort(-uq_scores)
    a_uq = get_auc(accuracies[uq_indices])
    
    # A_Random
    a_random = initial_acc
    
    # A_Oracle
    oracle_indices = np.argsort(accuracies)
    a_oracle = get_auc(accuracies[oracle_indices])
    
    if a_oracle == a_random:
        return 0.0
    
    prr = (a_uq - a_random) / (a_oracle - a_random)
    return prr
