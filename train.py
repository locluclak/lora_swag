import os
import torch
from torch.optim import AdamW
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, set_seed, get_linear_schedule_with_warmup

from src.swag import LoRASWAG
from src.data import get_dataloaders
from src.eval_utils import evaluate

@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    set_seed(cfg.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use a clean, predictable save path from config
    save_path = cfg.experiment.save_path
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving results to: {os.path.abspath(save_path)}")

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

    # 4. Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=cfg.experiment.learning_rate)
    
    total_steps = len(train_loader) * cfg.experiment.num_epochs
    swag_start_step = int(total_steps * cfg.experiment.swag_start_ratio)
    swag_steps_remaining = total_steps - swag_start_step
    # Calculate interval to collect exactly cfg.experiment.swag_total_samples
    swag_collect_interval = max(1, swag_steps_remaining // cfg.experiment.swag_total_samples)
    
    # Standard scheduler for the first 75%
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * 0.05), 
        num_training_steps=total_steps
    )

    print(f"Total steps: {total_steps} | SWAG starts at step: {swag_start_step} | Collection interval: {swag_collect_interval}")

    global_step = 0
    swag_collected_count = 0
    
    # 5. Training Loop
    train_losses = []
    for epoch in range(cfg.experiment.num_epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            # Step scheduler only BEFORE swag_start_step
            if global_step < swag_start_step:
                scheduler.step()
            
            loss_val = loss.item()
            epoch_loss += loss_val
            train_losses.append(loss_val)
            global_step += 1
            
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({"loss": f"{loss_val:.4f}", "lr": f"{current_lr:.2e}", "step": global_step})

            # SWAG Logic
            if global_step == swag_start_step:
                print(f"\n[SWAG] Reached start step {global_step}. Setting LR to constant {cfg.experiment.swag_lr_ratio} * peak.")
                # Set LR to ratio * peak (cfg.experiment.learning_rate)
                swag_lr = cfg.experiment.swag_lr_ratio * cfg.experiment.learning_rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = swag_lr
                
                print(f"[SWAG] Saving base LoRA adapter...")
                base_adapter_path = os.path.join(save_path, "base_lora_adapter")
                model.save_pretrained(base_adapter_path)
                print(f"[SWAG] Base adapter saved to {base_adapter_path}")

            # 2. Collect statistics
            if global_step >= swag_start_step:
                # Collect every N steps to reach the target sample count
                if (global_step - swag_start_step) % swag_collect_interval == 0 and swag_collected_count < cfg.experiment.swag_total_samples:
                    print(f"\n[Step {global_step}] Collecting SWAG sample {swag_collected_count+1}/{cfg.experiment.swag_total_samples} (LR: {current_lr:.2e})")
                    swag_model.collect_model()
                    swag_collected_count += 1
        
        print(f"Epoch {epoch} Avg Train Loss: {epoch_loss / len(train_loader):.4f}")
        
        # Periodic validation
        val_acc, _, _, _, _ = evaluate(model, val_loader, device)
        print(f"Validation Accuracy: {val_acc:.4f}")

    # 6. Save Outputs
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Global Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "loss_curve.png"))
    plt.close()

    model.save_pretrained(os.path.join(save_path, "last_lora_adapter"))
    # Save only the SWAG statistics (much smaller)
    swag_stats = swag_model.get_swag_stats()
    torch.save(swag_stats, os.path.join(save_path, "swag_stats.pt"))
    print(f"Training finished. {swag_collected_count} SWAG samples collected. Model saved to {save_path}")

if __name__ == "__main__":
    main()
