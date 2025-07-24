import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

import warnings
warnings.filterwarnings("ignore")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

NUM_PROMPTS = 5

def preprocess_function(articles,summaries):
    model_inputs = tokenizer(articles, max_length=512, padding="max_length", truncation=True)
    labels = tokenizer(summaries, max_length=512, padding="max_length", truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    input_ids = model_inputs['input_ids']
    attention_mask = model_inputs['attention_mask']
    label_ids = model_inputs['labels']
    return input_ids, attention_mask, label_ids

class FineTuningDataset(Dataset):
    def __init__(self, articles, summaries):
        self.input_ids, self.attention_mask, self.label_ids = preprocess_function(articles, summaries)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return torch.tensor(self.input_ids[idx], dtype=torch.long), torch.tensor(self.attention_mask[idx], dtype=torch.long), torch.tensor(self.label_ids[idx], dtype=torch.long)

def main ():
    df = pd.read_csv('cnn_dailymail/train.csv')
    df = df.dropna()

    # Select a subset of 30,000 samples
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df[:30000]

    # Split the dataset into train, validation, and test sets
    train_df = df[:21000]
    val_df = df[21000:27000]
    test_df = df[27000:]

    # Extract articles and summaries for each split
    train_articles = train_df['article'].tolist()
    train_summaries = train_df['highlights'].tolist()
    train_data = FineTuningDataset(train_articles,train_summaries)
    torch.save(train_data, "train_data.pt")

    val_articles = val_df['article'].tolist()
    val_summaries = val_df['highlights'].tolist()
    val_data = FineTuningDataset(val_articles,val_summaries)
    torch.save(val_data, "val_data.pt")

    test_articles = test_df['article'].tolist()
    test_summaries = test_df['highlights'].tolist()
    test_data = FineTuningDataset(test_articles,test_summaries)
    torch.save(test_data, "test_data.pt")

if __name__ == "__main__":
    main()