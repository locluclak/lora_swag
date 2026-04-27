import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, set_seed
from src.swag import LoRASWAG
from src.data_qa import get_qa_dataloaders

def get_loss(model, dataloader, device, max_batches=5):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            count += 1
    return total_loss / count if count > 0 else 0

@hydra.main(config_path="configs", config_name="qa", version_base="1.1")
def main(cfg: DictConfig):
    set_seed(cfg.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    save_path = cfg.experiment.save_path
    stats_path = os.path.join(save_path, "swag_stats.pt")
    adapter_path = os.path.join(save_path, "base_lora_adapter")
    
    if not os.path.exists(stats_path) or not os.path.exists(adapter_path):
        print(f"Error: Missing files in {save_path}. Run train_qa.py first.")
        return

    # 1. Load Data (Small subset for loss calculation)
    _, val_loader, tokenizer = get_qa_dataloaders(
        model_name=cfg.model.model_name,
        task=cfg.experiment.task,
        subset=cfg.experiment.subset,
        batch_size=cfg.experiment.batch_size,
        max_length=cfg.experiment.max_length,
        dataset_percentage=0.01 
    )

    # 2. Setup Model
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        target_modules=list(cfg.model.target_modules),
    )
    model = get_peft_model(base_model, peft_config)
    model.load_adapter(adapter_path, "default")
    model.to(device)

    # 3. Load SWAG
    swag_model = LoRASWAG(model, max_num_models=cfg.experiment.max_num_models)
    swag_stats = torch.load(stats_path, map_location=device)
    swag_model.load_swag_stats(swag_stats)

    # 4. Extract Trajectory and compute PCA
    all_means = []
    all_covs = [] 
    param_info = [] 
    
    for module, param_name, full_path in swag_model.params:
        mean = getattr(module, f"{param_name}_mean")
        cov_sqrt = getattr(module, f"{param_name}_cov_mat_sqrt") 
        all_means.append(mean.view(-1))
        all_covs.append(cov_sqrt)
        param_info.append((module, param_name, mean.shape))

    theta_swa = torch.cat(all_means)
    D_mat = torch.cat(all_covs, dim=1) 
    
    U, S, Vh = torch.linalg.svd(D_mat, full_matrices=False)
    
    # Project Training Points (Red dots)
    train_coords = (U * S).cpu().numpy()

    # Generate and Project Posterior Samples (Blue x)
    num_samples = 30
    print(f"Drawing {num_samples} samples from the posterior...")
    sampled_coords = []
    for _ in range(num_samples):
        swag_model.sample(scale=cfg.experiment.swag_scale, use_cov=True)
        current_params = []
        for module, param_name, _ in swag_model.params:
            current_params.append(getattr(module, param_name).data.view(-1))
        w_sampled = torch.cat(current_params)
        coords = (w_sampled - theta_swa) @ Vh.T
        sampled_coords.append(coords.cpu().numpy())
    sampled_coords = np.array(sampled_coords)

    # 5. Grid Visualization
    res = 12 # Slightly lower resolution for faster generation on large LLMs
    u_range = np.linspace(-8, 8, res)
    v_range = np.linspace(-8, 8, res)
    U_grid, V_grid = np.meshgrid(u_range, v_range)

    def plot_landscape(v1_idx, v2_idx, filename):
        e1 = Vh[v1_idx]
        e2 = Vh[v2_idx]
        loss_grid = np.zeros_like(U_grid)
        
        print(f"Generating loss landscape for PC {v1_idx+1} & {v2_idx+1}...")
        for i in tqdm(range(res)):
            for j in range(res):
                u, v = U_grid[i, j], V_grid[i, j]
                new_weights_flat = theta_swa + u * e1 + v * e2
                
                offset = 0
                for module, param_name, shape in param_info:
                    numel = np.prod(shape)
                    w_layer = new_weights_flat[offset : offset + numel].view(shape)
                    getattr(module, param_name).data.copy_(w_layer)
                    offset += numel
                
                loss_grid[i, j] = get_loss(model, val_loader, device, max_batches=5)

        plt.figure(figsize=(10, 8))
        cp = plt.contourf(U_grid, V_grid, loss_grid, levels=20, cmap='Spectral_r')
        plt.colorbar(cp, label='Loss')
        
        # Plot markers
        plt.scatter(train_coords[:, v1_idx], train_coords[:, v2_idx], color='red', s=20, alpha=0.5, label='Training Trajectory')
        plt.scatter(sampled_coords[:, v1_idx], sampled_coords[:, v2_idx], color='blue', s=40, alpha=0.5, label='Posterior Samples', marker='x')
        plt.scatter(0, 0, marker='*', color='black', s=200, label='SWA Center', zorder=5)
        
        plt.xlabel(f'PC {v1_idx+1}')
        plt.ylabel(f'PC {v2_idx+1}')
        plt.title(f'QA Posterior Landscape\nAxes: PC {v1_idx+1} vs PC {v2_idx+1}')
        plt.legend()
        
        plot_path = os.path.join(save_path, filename)
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        plt.close()

    plot_landscape(0, 1, "qa_posterior_pc1_2.png")
    if Vh.size(0) >= 4:
        plot_landscape(2, 3, "qa_posterior_pc3_4.png")

if __name__ == "__main__":
    main()
