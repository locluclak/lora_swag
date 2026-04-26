import math

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
        
        # Identify LoRA parameters correctly
        # In PEFT, lora_A/B are submodules, and the parameter is named 'weight'
        self.params: List[Tuple[nn.Module, str, str]] = []
        
        for name, param in self.model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                # Split name into module path and parameter name
                # e.g., "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
                parts = name.split('.')
                module_path = '.'.join(parts[:-1])
                param_name = parts[-1]
                
                module = self.model.get_submodule(module_path)
                self.params.append((module, param_name, module_path))
        
        if len(self.params) == 0:
            print("WARNING: No LoRA parameters found! SWAG statistics will be empty.")
        else:
            print(f"LoRASWAG: Found {len(self.params)} LoRA parameters to track.")
        
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
            
            # Use float64 for intermediate calculation if precision is an issue, 
            # but usually float32 is fine for SWA
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

        # K trong công thức chính là số lượng model đã collect
        K = self.n_models.item()

        for module, param_name, _ in self.params:
            mean = getattr(module, f"{param_name}_mean")
            sq_mean = getattr(module, f"{param_name}_sq_mean")
            
            # 1. Diagonal variance part: (1/sqrt(2)) * sqrt(var) * z1
            var = torch.clamp(sq_mean - mean**2, self.var_clamp)
            # Hệ số 1/sqrt(2) rất quan trọng để khớp với phân phối Gaussian của SWAG
            sample = mean + (1.0 / math.sqrt(2.0)) * torch.randn_like(mean) * torch.sqrt(var) * scale
            
            # 2. Low-rank covariance part: (1/sqrt(2*(K-1))) * D * z2
            if use_cov:
                cov_mat_sqrt = getattr(module, f"{param_name}_cov_mat_sqrt") # Đây chính là ma trận D_hat
                if cov_mat_sqrt.size(0) > 0:
                    # z2 ~ N(0, I_K)
                    z2 = torch.randn((cov_mat_sqrt.size(0), 1), device=mean.device)
                    
                    # D_hat * z2
                    cov_sample = (cov_mat_sqrt.t() @ z2).view_as(mean)
                    
                    # Nhân với hệ số 1 / sqrt(2 * (K - 1))
                    scale_low_rank = scale / math.sqrt(2.0 * max(1, K - 1))
                    sample = sample + (cov_sample * scale_low_rank)
            
            # Cập nhật trọng số vào model để chuẩn bị forward pass
            getattr(module, param_name).data.copy_(sample)

    def get_swag_stats(self) -> Dict[str, torch.Tensor]:
        """Extract only the SWAG buffers for efficient saving."""
        stats = {"n_models": self.n_models}
        for module, param_name, full_module_path in self.params:
            for suffix in ["_mean", "_sq_mean", "_cov_mat_sqrt"]:
                key = f"{full_module_path}.{param_name}{suffix}"
                stats[key] = getattr(module, f"{param_name}{suffix}")
        return stats

    def load_swag_stats(self, stats: Dict[str, torch.Tensor]):
        """Load SWAG buffers from a filtered dictionary."""
        self.n_models.copy_(stats["n_models"])
        for module, param_name, full_module_path in self.params:
            for suffix in ["_mean", "_sq_mean", "_cov_mat_sqrt"]:
                key = f"{full_module_path}.{param_name}{suffix}"
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
