import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, set_seed
from src.swag import LoRASWAG
from src.data import get_dataloaders

def get_loss(model, dataloader, device, max_batches=5):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            count += 1
    return total_loss / count if count > 0 else 0

@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    set_seed(cfg.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    save_path = cfg.experiment.save_path
    stats_path = os.path.join(save_path, "swag_stats.pt")
    adapter_path = os.path.join(save_path, "base_lora_adapter")
    
    if not os.path.exists(stats_path) or not os.path.exists(adapter_path):
        print(f"Error: Missing files in {save_path}. Run train.py first.")
        return

    # 1. Load Data
    _, val_loader, _, _, tokenizer = get_dataloaders(
        model_name=cfg.model.model_name,
        task=cfg.experiment.task,
        ood_task=cfg.experiment.ood_task,
        batch_size=cfg.experiment.batch_size,
        max_length=cfg.experiment.max_length,
        dataset_percentage=0.1 # Use small subset for faster plotting
    )

    # 2. Setup Model
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
    # Load the base adapter weights
    model.load_adapter(adapter_path, "default")
    model.to(device)

    # 3. Load SWAG stats
    swag_model = LoRASWAG(model, max_num_models=cfg.experiment.max_num_models)
    swag_stats = torch.load(stats_path, map_location=device)
    swag_model.load_swag_stats(swag_stats)

    # 4. Prepare for PCA-based projection
    # Collect all mean parameters and cov_mat_sqrt into large tensors
    all_means = []
    all_covs = [] # This will be (K, D_total)
    
    param_info = [] # To map back to model
    
    for module, param_name, full_path in swag_model.params:
        mean = getattr(module, f"{param_name}_mean")
        cov_sqrt = getattr(module, f"{param_name}_cov_mat_sqrt") # (K, D_layer)
        
        all_means.append(mean.view(-1))
        all_covs.append(cov_sqrt)
        param_info.append((module, param_name, mean.shape))

    theta_swa = torch.cat(all_means)
    D_mat = torch.cat(all_covs, dim=1) # (K, D_total)
    
    print(f"Total SWAG parameters: {theta_swa.numel()}")
    print(f"Covariance rank: {D_mat.size(0)}")

    # Compute SVD to get top eigenvectors
    U, S, Vh = torch.linalg.svd(D_mat, full_matrices=False)
    
    # Project the original samples (rows of D_mat) onto the PCA axes
    # Coordinates of sample i on PC j is simply (D_mat @ Vh.T)_{i,j}
    # Since we have U, S, Vh, and D_mat = U S Vh, then D_mat @ Vh.T = U S
    sample_coords = U * S
    sample_coords = sample_coords.cpu().numpy()

    # Grid definition
    res = 15 
    u_range = np.linspace(-5, 5, res)
    v_range = np.linspace(-5, 5, res)
    U_grid, V_grid = np.meshgrid(u_range, v_range)

    # Function to generate and save a plot for a given pair of eigenvectors
    def plot_landscape(v1_idx, v2_idx, filename):
        e1 = Vh[v1_idx]
        e2 = Vh[v2_idx]
        loss_grid = np.zeros_like(U_grid)
        
        print(f"Generating loss landscape grid for PC {v1_idx+1} & {v2_idx+1}...")
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
                    
                loss_grid[i, j] = get_loss(model, val_loader, device, max_batches=10)

        plt.figure(figsize=(10, 8))
        cp = plt.contourf(U_grid, V_grid, loss_grid, levels=20, cmap='Spectral_r')
        plt.colorbar(cp, label='Loss')
        
        # Plot SWAG samples
        plt.scatter(sample_coords[:, v1_idx], sample_coords[:, v2_idx], 
                    color='red', s=30, alpha=0.6, label='SWAG Samples', edgecolors='white')
        
        # Plot SWA center
        plt.scatter(0, 0, marker='*', color='black', s=200, label='SWA Center', zorder=5)
        
        plt.xlabel(f'PC {v1_idx+1}')
        plt.ylabel(f'PC {v2_idx+1}')
        plt.title(f'Posterior Loss Landscape (LoRA-SWAG)\nAxes: PC {v1_idx+1} vs PC {v2_idx+1}')
        plt.legend()
        
        plot_path = os.path.join(save_path, filename)
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        plt.close()

    # Generate the two requested plots
    plot_landscape(0, 1, "posterior_loss_pc1_2.png")
    if Vh.size(0) >= 4:
        plot_landscape(2, 3, "posterior_loss_pc3_4.png")
    else:
        print("Warning: Not enough components for PC 3 & 4 plot. Increase max_num_models/total_samples.")

if __name__ == "__main__":
    main()
