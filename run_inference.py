import os
import torch
import random
import hydra
from omegaconf import DictConfig
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.swag import LoRASWAG
from src.data import get_dataloaders

@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    save_path = cfg.experiment.save_path
    stats_path = os.path.join(save_path, "swag_stats.pt")
    adapter_path = os.path.join(save_path, "base_lora_adapter")
    
    if not os.path.exists(stats_path) or not os.path.exists(adapter_path):
        print(f"Error: Missing files in {save_path}. Run train.py first.")
        return

    # 1. Load Tokenizer and Data
    # We use a batch size of 1 to easily pick random samples
    _, val_loader, _, ood_loader, tokenizer = get_dataloaders(
        model_name=cfg.model.model_name,
        task=cfg.experiment.task,
        ood_task=cfg.experiment.ood_task,
        batch_size=1,
        max_length=cfg.experiment.max_length
    )

    # 2. Setup Model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.model_name, num_labels=3
    )
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        target_modules=list(cfg.model.target_modules),
    )
    model = get_peft_model(base_model, peft_config)
    model.load_adapter(adapter_path, "default")
    model.to(device)

    # 3. Load SWAG
    swag_model = LoRASWAG(model)
    swag_stats = torch.load(stats_path, map_location=device)
    swag_model.load_swag_stats(swag_stats)

    # Label maps
    mnli_labels = {0: "entailment", 1: "neutral", 2: "contradiction"}
    rte_labels = {0: "entailment", 1: "not_entailment", 2: "N/A"}

    def run_examples(dataloader, name, label_map, num_examples=3):
        print(f"\n{'='*20} {name} Examples {'='*20}")
        dataset = dataloader.dataset
        indices = random.sample(range(len(dataset)), num_examples)
        
        for idx in indices:
            item = dataset[idx]
            # Convert tensors to device and add batch dim
            input_ids = item['input_ids'].unsqueeze(0).to(device)
            attention_mask = item['attention_mask'].unsqueeze(0).to(device)
            label = item['labels'].item()

            # Decode text
            text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            
            # 1. Base Model Prediction
            model.eval()
            with torch.no_grad():
                base_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                base_probs = torch.softmax(base_logits, dim=-1).float().cpu().numpy()[0]
                base_pred = np.argmax(base_probs)

            # 2. SWAG Ensemble Prediction (BMA)
            swag_probs_list = []
            for _ in range(cfg.experiment.swag_samples):
                swag_model.sample(scale=cfg.experiment.swag_scale, use_cov=True)
                with torch.no_grad():
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                    swag_probs_list.append(torch.softmax(logits, dim=-1).float())
            
            avg_probs_tensor = torch.stack(swag_probs_list).mean(dim=0)
            avg_swag_probs = avg_probs_tensor.cpu().numpy()[0]
            swag_pred = np.argmax(avg_swag_probs)
            swag_conf = np.max(avg_swag_probs)

            print(f"\n[Sample {idx}]")
            print(f"Input Text: {text[:200]}...")
            print(f"True Label: {label} ({label_map.get(label, 'Unknown')})")
            print(f"Base Model: Pred={base_pred} ({label_map.get(base_pred, 'Unknown')}), Probs={base_probs.round(3)}")
            print(f"SWAG Model: Pred={swag_pred} ({label_map.get(swag_pred, 'Unknown')}), Probs={avg_swag_probs.round(3)}, Conf={swag_conf:.3f}")

    import numpy as np
    run_examples(val_loader, "MNLI (ID)", mnli_labels)
    run_examples(ood_loader, "RTE (OOD)", rte_labels)

if __name__ == "__main__":
    main()
