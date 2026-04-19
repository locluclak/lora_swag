import os
import torch
from torch.optim import AdamW
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, set_seed

from src.swag import LoRASWAG
from src.data import get_dataloaders
from src.eval_utils import evaluate
from hydra.core.hydra_config import HydraConfig

@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    set_seed(cfg.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine save path (respect Hydra's output dir if not absolute)
    save_path = cfg.experiment.save_path
    if not os.path.isabs(save_path):
        save_path = os.path.join(HydraConfig.get().runtime.output_dir, save_path)
    print(f"Saving results to: {save_path}")

    # 1. Load Data

    train_loader, val_loader, _, _, tokenizer = get_dataloaders(
        model_name=cfg.model.model_name,
        task=cfg.experiment.task,
        ood_task=cfg.experiment.ood_task,
        batch_size=cfg.experiment.batch_size,
        max_length=cfg.experiment.max_length,
        dataset_percentage=cfg.experiment.dataset_percentage
    )


    # 2. Setup Model with LoRA
    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.model_name, num_labels=3
    )
    if tokenizer.pad_token_id is not None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        target_modules=list(cfg.model.target_modules),
    )
    model = get_peft_model(base_model, peft_config)
    model.to(device)
    model.print_trainable_parameters()

    # 3. Setup SWAG wrapper
    swag_model = LoRASWAG(model, max_num_models=cfg.experiment.max_num_models)
    swag_model.to(device)

    # 4. Optimizer
    optimizer = AdamW(model.parameters(), lr=cfg.experiment.learning_rate)

    # 5. Training Loop
    for epoch in range(cfg.experiment.num_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        print(f"Epoch {epoch} Avg Train Loss: {total_loss / len(train_loader):.4f}")

        # SWAG Collection
        if epoch >= cfg.experiment.swag_start_epoch:
            if (epoch - cfg.experiment.swag_start_epoch) % cfg.experiment.swag_collect_freq == 0:
                print(f"Collecting model for SWAG at epoch {epoch}")
                swag_model.collect_model()

        # Simple Validation during training
        val_acc, _, _, _ = evaluate(model, val_loader, device)
        print(f"Validation Accuracy: {val_acc:.4f}")

    # 6. Save Outputs
    os.makedirs(save_path, exist_ok=True)
    # Save the LoRA adapter (used to load the model architecture later)
    model.save_pretrained(os.path.join(save_path, "lora_adapter"))
    # Save the SWAG stats
    torch.save(swag_model.state_dict(), os.path.join(save_path, "swag_model.pt"))
    
    print(f"Training finished. Model saved to {save_path}")

if __name__ == "__main__":
    main()
