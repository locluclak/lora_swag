import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

class TriviaQADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        # In TriviaQA rc.nocontext, the answer is in 'answer' dict
        answer = item["answer"]["normalized_value"]
        
        prompt = f"Question: {question}\nAnswer:"
        
        # Tokenize prompt and answer
        inputs = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # For training, we need labels. In CausalLM, labels are shifted version of input_ids.
        # But for simple fine-tuning, we often just train on the whole sequence.
        # Here we'll concatenate prompt + answer for training.
        full_text = f"Question: {question}\nAnswer: {answer}"
        full_enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # We only want to train on the answer part, so we mask the prompt in labels
        labels = full_enc["input_ids"].clone()
        prompt_enc = self.tokenizer(prompt, max_length=self.max_length, truncation=True)
        prompt_len = len(prompt_enc["input_ids"])
        
        labels[0, :prompt_len] = -100 # Mask prompt tokens in loss calculation
        
        return {
            "input_ids": full_enc["input_ids"].squeeze(0),
            "attention_mask": full_enc["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "question": question,
            "answer": answer
        }

def get_qa_dataloaders(model_name, task="trivia_qa", subset="rc.nocontext", batch_size=4, max_length=512, dataset_percentage=0.1):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset(task, subset)
    
    train_data = dataset["train"]
    val_data = dataset["validation"]
    
    if dataset_percentage < 1.0:
        train_size = int(len(train_data) * dataset_percentage)
        val_size = int(len(val_data) * dataset_percentage)
        train_data = train_data.select(range(train_size))
        val_data = val_data.select(range(val_size))

    train_dataset = TriviaQADataset(train_data, tokenizer, max_length)
    val_dataset = TriviaQADataset(val_data, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader, tokenizer
