import os
import torch
import hydra
from omegaconf import DictConfig
from peft import PeftModel
from transformers import AutoModelForCausalLM, set_seed, AutoTokenizer

from src.swag import LoRASWAG
from src.data_qa import get_qa_dataloaders
from src.eval_utils_qa import evaluate_qa, plot_qa_uncertainty, compute_prr_qa

@hydra.main(config_path="configs", config_name="qa", version_base="1.1")
def main(cfg: DictConfig):
    set_seed(cfg.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    save_path = cfg.experiment.save_path
    
    # 1. Load Data
    _, val_loader, tokenizer = get_qa_dataloaders(
        model_name=cfg.model.model_name,
        task=cfg.experiment.task,
        subset=cfg.experiment.subset,
        batch_size=1, # Eval batch size 1 for generation
        dataset_percentage=cfg.experiment.dataset_percentage
    )

    # 2. Setup Base Model
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    if tokenizer.pad_token_id is not None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    # 3. Evaluate Baseline (Standard LoRA)
    base_adapter_path = os.path.join(save_path, "base_lora_adapter")
    if os.path.exists(base_adapter_path):
        print("\n--- Evaluating Baseline (Standard LoRA) ---")
        model = PeftModel.from_pretrained(base_model, base_adapter_path)
        model.to(device)
        
        base_results = evaluate_qa(model, val_loader, device, tokenizer, max_gen_len=cfg.experiment.max_new_tokens)
        base_prr = compute_prr_qa(base_results["em_list"], base_results["uncertainties"])
        
        print(f"Baseline EM: {base_results['em']:.4f}")
        print(f"Baseline F1: {base_results['f1']:.4f}")
        print(f"Baseline PRR: {base_prr:.4f}")
        
        plot_qa_uncertainty(base_results["em_list"], base_results["uncertainties"], 
                            os.path.join(save_path, "base_qa_uncertainty.png"))
        
        # Show Examples
        print("\n--- Baseline Examples ---")
        for res in base_results["results"][:3]:
            print(f"Q: {res['question']}")
            print(f"Pred: {res['prediction']} | GT: {res['ground_truth']} | EM: {res['em']}")
    else:
        print(f"Error: {base_adapter_path} not found.")
        return

    # 4. Evaluate SWAG (Bayesian LoRA)
    swag_path = os.path.join(save_path, "swag_stats.pt")
    if os.path.exists(swag_path):
        print("\n--- Evaluating SWAG (Bayesian LoRA) ---")
        swag_model = LoRASWAG(model, max_num_models=cfg.experiment.max_num_models)
        swag_stats = torch.load(swag_path, map_location=device)
        swag_model.load_swag_stats(swag_stats)
        swag_model.to(device)
        
        swag_results = evaluate_qa(swag_model, val_loader, device, tokenizer, 
                                   num_samples=cfg.experiment.swag_eval_samples, 
                                   scale=cfg.experiment.swag_scale,
                                   max_gen_len=cfg.experiment.max_new_tokens)
        swag_prr = compute_prr_qa(swag_results["em_list"], swag_results["uncertainties"])
        
        print(f"SWAG EM: {swag_results['em']:.4f}")
        print(f"SWAG F1: {swag_results['f1']:.4f}")
        print(f"SWAG PRR: {swag_prr:.4f}")
        
        plot_qa_uncertainty(swag_results["em_list"], swag_results["uncertainties"], 
                            os.path.join(save_path, "swag_qa_uncertainty.png"))

        # Show Examples
        print("\n--- SWAG Examples ---")
        for res in swag_results["results"][:3]:
            print(f"Q: {res['question']}")
            print(f"Pred: {res['prediction']} | GT: {res['ground_truth']} | EM: {res['em']}")
    else:
        print("SWAG stats not found.")

if __name__ == "__main__":
    main()
