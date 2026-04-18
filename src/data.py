from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
import numpy as np

def get_dataloaders(model_name: str, task: str = "mnli", ood_task: str = "rte", batch_size: int = 32, max_length: int = 128, dataset_percentage: float = 1.0):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Keys for GLUE tasks
    task_to_keys = {
        "mnli": ("premise", "hypothesis"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
    }

    def tokenize_function(examples, current_task):
        key1, key2 = task_to_keys[current_task]
        args = (examples[key1],) if key2 is None else (examples[key1], examples[key2])
        return tokenizer(*args, truncation=True, padding=False, max_length=max_length)

    # ID Dataset: MNLI
    id_dataset = load_dataset("glue", task)
    
    # Subsampling ID training data
    if dataset_percentage < 1.0:
        for split in ["train"]:
            num_samples = int(len(id_dataset[split]) * dataset_percentage)
            indices = np.random.choice(len(id_dataset[split]), num_samples, replace=False)
            id_dataset[split] = id_dataset[split].select(indices)
            print(f"Subsampled {split} to {num_samples} samples ({dataset_percentage*100}%).")

    tokenized_id = id_dataset.map(lambda x: tokenize_function(x, task), batched=True)
    tokenized_id = tokenized_id.rename_column("label", "labels")
    tokenized_id.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # OOD Dataset: RTE
    ood_dataset = load_dataset("glue", ood_task)
    tokenized_ood = ood_dataset.map(lambda x: tokenize_function(x, ood_task), batched=True)
    tokenized_ood = tokenized_ood.rename_column("label", "labels")
    tokenized_ood.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(tokenized_id["train"], batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(tokenized_id["validation_matched"], batch_size=batch_size, collate_fn=data_collator)
    test_id_loader = DataLoader(tokenized_id["validation_matched"], batch_size=batch_size, collate_fn=data_collator)
    test_ood_loader = DataLoader(tokenized_ood["validation"], batch_size=batch_size, collate_fn=data_collator)

    return train_loader, val_loader, test_id_loader, test_ood_loader, tokenizer
