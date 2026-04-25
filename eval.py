import os
import torch
import hydra
from omegaconf import DictConfig
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, set_seed

from src.swag import LoRASWAG
from src.data import get_dataloaders
from src.eval_utils import (
    calculate_ece, evaluate, compute_ood_metrics, compute_prr, 
    plot_combined_reliability_diagram, plot_entropy_dist, plot_confidence_dist
)

def get_binary_acc(probs, labels):
    preds = probs.argmax(axis=-1)
    return (preds == labels).astype(int)

@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    set_seed(cfg.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use the clean save path from config
    save_path = cfg.experiment.save_path
    print(f"Loading models from: {os.path.abspath(save_path)}")

    # 1. Load Data
    _, _, test_id_loader, test_ood_loader, tokenizer = get_dataloaders(
        model_name=cfg.model.model_name,
        task=cfg.experiment.task,
        ood_task=cfg.experiment.ood_task,
        batch_size=cfg.experiment.batch_size,
        max_length=cfg.experiment.max_length,
        dataset_percentage=cfg.experiment.dataset_percentage
    )

    # 2. Setup Base Model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.model_name, num_labels=3
    )
    if tokenizer.pad_token_id is not None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    # 3. Evaluate Baseline (Base LoRA Adapter)
    base_adapter_path = os.path.join(save_path, "base_lora_adapter")
    if not os.path.exists(base_adapter_path):
        print(f"Error: Base adapter not found at {base_adapter_path}. Run train.py first.")
        return
    
    print("\n--- Evaluating Baseline (Standard LoRA) ---")
    model = PeftModel.from_pretrained(base_model, base_adapter_path)
    model.to(device)

    base_id_acc, base_id_probs, base_id_labels, id_entropies_base, base_nll = evaluate(model, test_id_loader, device)
    base_ood_acc, base_ood_probs, base_ood_labels, ood_entropies_base, ood_nll = evaluate(model, test_ood_loader, device)
    
    base_auroc = compute_ood_metrics(id_entropies_base, ood_entropies_base)
    base_prr = compute_prr(get_binary_acc(base_id_probs, base_id_labels), id_entropies_base)
    base_prr_ood = compute_prr(get_binary_acc(base_ood_probs, base_ood_labels), ood_entropies_base)
    base_ece_id = calculate_ece(base_id_probs, base_id_labels)
    base_ece_ood = calculate_ece(base_ood_probs, base_ood_labels)
    
    print(f"Base ID Acc: {base_id_acc:.4f} | ECE: {base_ece_id:.4f} | PRR: {base_prr:.4f}")
    print(f"Base OOD Acc: {base_ood_acc:.4f} | ECE: {base_ece_ood:.4f} | AUROC: {base_auroc:.4f}")

    # Plotting Baseline
    plot_entropy_dist(id_entropies_base, ood_entropies_base, 
                      title="Base Model Entropy Distribution", 
                      path=os.path.join(save_path, "base_entropy_dist.png"))
    plot_confidence_dist(base_id_probs, base_ood_probs,
                         title="Base Model Confidence Distribution",
                         path=os.path.join(save_path, "base_confidence_dist.png"))
    plot_combined_reliability_diagram(base_id_probs, base_id_labels, 
                                        title="Base LoRA Calibration (ID)", 
                                        path=os.path.join(save_path, "base_reliability_diagram.png"))

    # 4. Evaluate SWAG (Base LoRA + Stats)
    print("\n--- Evaluating SWAG (Bayesian LoRA) ---")
    swag_model = LoRASWAG(model, max_num_models=cfg.experiment.max_num_models)
    swag_path = os.path.join(save_path, "swag_stats.pt")
    if not os.path.exists(swag_path):
        print("Error: SWAG stats not found. Skipping SWAG eval.")
    else:
        swag_stats = torch.load(swag_path, map_location=device)
        swag_model.load_swag_stats(swag_stats)
        swag_model.to(device)

        swag_id_acc, swag_id_probs, swag_id_labels, id_entropies_swag, swag_nll = evaluate(
            swag_model, test_id_loader, device, num_samples=cfg.experiment.swag_eval_samples, scale=cfg.experiment.swag_scale
        )
        swag_ood_acc, swag_ood_probs, swag_ood_labels, ood_entropies_swag, ood_nll = evaluate(
            swag_model, test_ood_loader, device, num_samples=cfg.experiment.swag_eval_samples, scale=cfg.experiment.swag_scale
        )
        
        swag_auroc = compute_ood_metrics(id_entropies_swag, ood_entropies_swag)
        swag_prr = compute_prr(get_binary_acc(swag_id_probs, swag_id_labels), id_entropies_swag)
        swag_ece_id = calculate_ece(swag_id_probs, swag_id_labels)
        swag_ece_ood = calculate_ece(swag_ood_probs, swag_ood_labels)

        print(f"SWAG ID Acc: {swag_id_acc:.4f} | ECE: {swag_ece_id:.4f} | PRR: {swag_prr:.4f}")
        print(f"SWAG OOD Acc: {swag_ood_acc:.4f} | ECE: {swag_ece_ood:.4f} | AUROC: {swag_auroc:.4f}")

        # Plotting SWAG
        plot_entropy_dist(id_entropies_swag, ood_entropies_swag, 
                        title="SWAG Model Entropy Distribution", 
                        path=os.path.join(save_path, "swag_entropy_dist.png"))
        plot_confidence_dist(swag_id_probs, swag_ood_probs,
                            title="SWAG Model Confidence Distribution",
                            path=os.path.join(save_path, "swag_confidence_dist.png"))
        plot_combined_reliability_diagram(swag_id_probs, swag_id_labels, 
                                          title="SWAG LoRA Calibration (ID)", 
                                          path=os.path.join(save_path, "swag_reliability_diagram.png"))

if __name__ == "__main__":
    main()
