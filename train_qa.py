import os
import torch
from torch.optim import AdamW
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, set_seed, get_linear_schedule_with_warmup

from src.swag import LoRASWAG
from src.data_qa import get_qa_dataloaders

@hydra.main(config_path="configs", config_name="qa", version_base="1.1")
def main(cfg: DictConfig):
    set_seed(cfg.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    save_path = cfg.experiment.save_path
    os.makedirs(save_path, exist_ok=True)

    # 1. Load Data
    train_loader, val_loader, tokenizer = get_qa_dataloaders(
        model_name=cfg.model.model_name,
        task=cfg.experiment.task,
        subset=cfg.experiment.subset,
        batch_size=cfg.experiment.batch_size,
        max_length=cfg.experiment.max_length,
        dataset_percentage=cfg.experiment.dataset_percentage
    )

    # 2. Setup Model for Causal LM
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    if tokenizer.pad_token_id is not None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
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
    swag_collect_interval = max(1, swag_steps_remaining // cfg.experiment.swag_total_samples)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps
    )

    print(f"Total steps: {total_steps} | SWAG starts at step: {swag_start_step}")

    global_step = 0
    swag_collected_count = 0
    
    # 5. Training Loop
    for epoch in range(cfg.experiment.num_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            if global_step < swag_start_step:
                scheduler.step()
            
            total_loss += loss.item()
            global_step += 1
            
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})

            # SWAG Phase Shift
            if global_step == swag_start_step:
                print(f"\n[SWAG] Setting LR to constant {cfg.experiment.swag_lr_ratio} * peak.")
                swag_lr = cfg.experiment.swag_lr_ratio * cfg.experiment.learning_rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = swag_lr
                
                print(f"[SWAG] Saving base LoRA adapter...")
                model.save_pretrained(os.path.join(save_path, "base_lora_adapter"))

            # Collection
            if global_step >= swag_start_step:
                if (global_step - swag_start_step) % swag_collect_interval == 0 and swag_collected_count < cfg.experiment.swag_total_samples:
                    swag_model.collect_model()
                    swag_collected_count += 1
        
        print(f"Epoch {epoch} Avg Train Loss: {total_loss / len(train_loader):.4f}")

    # 6. Save Outputs
    model.save_pretrained(os.path.join(save_path, "last_lora_adapter"))
    swag_stats = swag_model.get_swag_stats()
    torch.save(swag_stats, os.path.join(save_path, "swag_stats.pt"))
    print(f"Training finished. Model saved to {save_path}")

if __name__ == "__main__":
    main()
