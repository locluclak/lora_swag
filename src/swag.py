import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional

class LoRASWAG(nn.Module):
    """
    SWAG wrapper for LoRA parameters. 
    It collects mean and second moments of LoRA weights.
    """
    def __init__(self, model: nn.Module, max_num_models: int = 20, var_clamp: float = 1e-30):
        super().__init__()
        self.model = model
        self.max_num_models = max_num_models
        self.var_clamp = var_clamp
        
        self.register_buffer("n_models", torch.zeros([1], dtype=torch.long))
        
        # Identify LoRA parameters
        self.params: List[Tuple[nn.Module, str, str]] = []
        for name, module in self.model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if 'lora_A' in param_name or 'lora_B' in param_name:
                    # Store module, param_name, and the full prefix for saving/loading
                    self.params.append((module, param_name, name))
        
        self._init_swag_buffers()

    def _init_swag_buffers(self):
        for module, param_name, _ in self.params:
            param = getattr(module, param_name)
            module.register_buffer(f"{param_name}_mean", torch.zeros_like(param.data))
            module.register_buffer(f"{param_name}_sq_mean", torch.zeros_like(param.data))
            module.register_buffer(f"{param_name}_cov_mat_sqrt", 
                                 torch.zeros((0, param.numel()), device=param.device))

    def collect_model(self):
        n = self.n_models.item()
        for module, param_name, _ in self.params:
            param = getattr(module, param_name).data
            mean = getattr(module, f"{param_name}_mean")
            sq_mean = getattr(module, f"{param_name}_sq_mean")
            
            new_mean = (mean * n + param) / (n + 1)
            new_sq_mean = (sq_mean * n + param**2) / (n + 1)
            
            cov_mat_sqrt = getattr(module, f"{param_name}_cov_mat_sqrt")
            dev = (param - new_mean).view(1, -1)
            new_cov_mat_sqrt = torch.cat([cov_mat_sqrt, dev], dim=0)
            
            if new_cov_mat_sqrt.size(0) > self.max_num_models:
                new_cov_mat_sqrt = new_cov_mat_sqrt[1:]
                
            setattr(module, f"{param_name}_mean", new_mean)
            setattr(module, f"{param_name}_sq_mean", new_sq_mean)
            setattr(module, f"{param_name}_cov_mat_sqrt", new_cov_mat_sqrt)
            
        self.n_models.add_(1)

    def sample(self, scale: float = 1.0, use_cov: bool = True):
        if self.n_models.item() == 0:
            return

        for module, param_name, _ in self.params:
            mean = getattr(module, f"{param_name}_mean")
            sq_mean = getattr(module, f"{param_name}_sq_mean")
            
            var = torch.clamp(sq_mean - mean**2, self.var_clamp)
            sample = mean + torch.randn_like(mean) * torch.sqrt(var) * scale
            
            if use_cov:
                cov_mat_sqrt = getattr(module, f"{param_name}_cov_mat_sqrt")
                if cov_mat_sqrt.size(0) > 0:
                    eps = torch.randn((cov_mat_sqrt.size(0), 1), device=mean.device)
                    cov_sample = (cov_mat_sqrt.t() @ eps).view_as(mean)
                    sample = sample + (cov_sample * scale / (max(1, self.n_models.item() - 1))**0.5)
            
            getattr(module, param_name).data.copy_(sample)

    def get_swag_stats(self) -> Dict[str, torch.Tensor]:
        """Extract only the SWAG buffers for efficient saving."""
        stats = {"n_models": self.n_models}
        for module, param_name, full_name in self.params:
            for suffix in ["_mean", "_sq_mean", "_cov_mat_sqrt"]:
                key = f"{full_name}.{param_name}{suffix}"
                stats[key] = getattr(module, f"{param_name}{suffix}")
        return stats

    def load_swag_stats(self, stats: Dict[str, torch.Tensor]):
        """Load SWAG buffers from a filtered dictionary."""
        self.n_models.copy_(stats["n_models"])
        for module, param_name, full_name in self.params:
            for suffix in ["_mean", "_sq_mean", "_cov_mat_sqrt"]:
                key = f"{full_name}.{param_name}{suffix}"
                if key in stats:
                    # Resize the cov_mat_sqrt buffer if necessary to match loaded rank
                    if suffix == "_cov_mat_sqrt":
                        loaded_val = stats[key]
                        current_buffer = getattr(module, f"{param_name}_cov_mat_sqrt")
                        if current_buffer.shape != loaded_val.shape:
                            # Re-register buffer with correct shape
                            module.register_buffer(f"{param_name}_cov_mat_sqrt", 
                                                 torch.zeros(loaded_val.shape, device=loaded_val.device))
                    
                    getattr(module, f"{param_name}{suffix}").copy_(stats[key])

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
