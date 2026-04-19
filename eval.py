import os
import torch
import hydra
from omegaconf import DictConfig
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, set_seed

from src.swag import LoRASWAG
from src.data import get_dataloaders
from src.eval_utils import evaluate, compute_ood_metrics

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

    # 2. Setup Base Model + Load LoRA Adapter
    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.model_name, num_labels=3
    )
    if tokenizer.pad_token_id is not None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    adapter_path = os.path.join(save_path, "lora_adapter")
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"LoRA adapter not found at {adapter_path}. Run train.py first.")
    
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.to(device)

    # 3. Setup SWAG wrapper + Load SWAG stats
    swag_model = LoRASWAG(model, max_num_models=cfg.experiment.max_num_models)
    swag_path = os.path.join(save_path, "swag_model.pt")
    if os.path.exists(swag_path):
        swag_model.load_state_dict(torch.load(swag_path, map_location=device))
        swag_model.to(device)
        print("Loaded SWAG stats.")
    else:
        print("Warning: SWAG model.pt not found. Only Base Model evaluation will be performed.")

    print("\n--- Evaluation Results ---")
    
    # 4. Evaluate Base LoRA Model
    print("Evaluating Base LoRA Model...")
    base_id_acc, _, _, id_entropies_base = evaluate(model, test_id_loader, device)
    _, _, _, ood_entropies_base = evaluate(model, test_ood_loader, device)
    base_auroc = compute_ood_metrics(id_entropies_base, ood_entropies_base)
    print(f"Base ID Acc: {base_id_acc:.4f}")
    print(f"Base OOD AUROC: {base_auroc:.4f}")

    # 5. Evaluate SWAG Model
    if swag_model.n_models.item() > 0:
        print("\nEvaluating SWAG LoRA Model (10 Samples)...")
        swag_id_acc, _, _, id_entropies_swag = evaluate(
            swag_model, test_id_loader, device, num_samples=10, scale=cfg.experiment.swag_scale
        )
        _, _, _, ood_entropies_swag = evaluate(
            swag_model, test_ood_loader, device, num_samples=10, scale=cfg.experiment.swag_scale
        )
        swag_auroc = compute_ood_metrics(id_entropies_swag, ood_entropies_swag)
        print(f"SWAG ID Acc: {swag_id_acc:.4f}")
        print(f"SWAG OOD AUROC: {swag_auroc:.4f}")

if __name__ == "__main__":
    main()
