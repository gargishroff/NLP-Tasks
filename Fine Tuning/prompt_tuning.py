import time
import torch
from tqdm import tqdm  
import torch.nn as nn
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW

import evaluate
rouge_metric = evaluate.load("rouge")

from data_preprocessing import preprocess_function, FineTuningDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
BATCH_SIZE = 4
NUM_PROMPTS = 5
LR = 5e-5
EPOCHS = 10

def count_parameters(model):
    return sum(p.numel() for p in model.prompt_embeddings.parameters() if p.requires_grad)

def pad_labels (label_ids):
    # Create a padding tensor filled with -100, same batch size as labels
    padding = torch.full((label_ids.size(0), NUM_PROMPTS), -100, dtype=label_ids.dtype, device=label_ids.device)
    # Concatenate padding and original labels along the last dimension
    padded_labels = torch.cat((padding, label_ids), dim=1)
    return padded_labels

class PromptTunedModel (nn.Module):
    def __init__(self):
        super(PromptTunedModel, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        for param in self.gpt2.parameters():
            param.requires_grad = False

        self.embd_size = self.gpt2.transformer.wte.embedding_dim
        self.prompt_embeddings = nn.Embedding(NUM_PROMPTS, self.embd_size)

    def forward (self, input_ids, attention_mask=None):
        input_embds = self.gpt2.transformer.wte(input_ids)
        prompt_embds = self.prompt_embeddings.weight.unsqueeze(0).expand(input_ids.size(0), -1, -1)
        final_input = torch.cat([prompt_embds, input_embds], dim=1)
        if attention_mask is not None:
            batch_size = input_ids.shape[0]
            attention_mask_prompt = torch.ones(batch_size,NUM_PROMPTS).to(device)
            final_attention_mask = torch.cat([attention_mask_prompt,attention_mask], dim=1)
        else :
            final_attention_mask = None
        return self.gpt2(inputs_embeds=final_input, attention_mask=final_attention_mask)

def generate_predictions(model, data_loader):
    model.eval()
    predictions = []
    references = []

    with torch.no_grad():
        for input_ids, attention_mask, label_ids in tqdm(data_loader, desc="Generating Predictions", unit="batch"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = pad_labels(label_ids)

            # Generate predictions
            outputs = model(input_ids, attention_mask)
            logits = outputs.logits
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # Decode predictions and references
            for pred, ref in zip(predicted_ids, labels):
                predictions.append(tokenizer.decode(pred, skip_special_tokens=True))
                references.append(tokenizer.decode(ref, skip_special_tokens=True))
    
    return predictions, references

def calculate_rouge(predictions, references):
    # Compute ROUGE scores using the evaluate library
    results = rouge_metric.compute(predictions=predictions, references=references)
    # Format and return average ROUGE scores
    avg_rouge = {key: results[key].mid.fmeasure for key in results.keys()}
    return avg_rouge

def evaluate_model (model,data_loader):
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    model.eval()

    total_loss = 0
    with torch.no_grad():
        progress_bar = tqdm (data_loader, desc=f"Evaluating Model", unit="batch", dynamic_ncols=True)
        for input_ids, attention_mask, label_ids in progress_bar:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label_ids = label_ids.to(device)
            
            outputs = model(input_ids,attention_mask)
            logits = outputs.logits
            labels = pad_labels(label_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss += loss.item()

        # Compute average loss
        avg_loss = total_loss / len(data_loader)
    return avg_loss


def finetuningloop (model, train_dataloader,val_dataloader):
    optimizer = AdamW(model.prompt_embeddings.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    initial_memory = torch.cuda.memory_allocated(device)

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        progress_bar = tqdm (train_dataloader, desc=f"Training Epoch {epoch + 1}", unit="batch", dynamic_ncols=True)
        total_loss = 0
        for input_ids, attention_mask, label_ids in progress_bar:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label_ids = label_ids.to(device)
            
            outputs = model(input_ids,attention_mask)
            labels = pad_labels(label_ids)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()  
            shift_labels = labels[..., 1:].contiguous()    
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_time = time.time()-start_time
        torch.cuda.synchronize()
        epoch_memory = torch.cuda.memory_allocated(device) - initial_memory
        print(f"Epoch {epoch + 1}, Average Training Loss: {total_loss/len(train_dataloader)}, Training Time: {epoch_time: .2f}, GPU Memory Usage: {epoch_memory / (1024**2):.2f} MB")
        val_loss = evaluate_model(model,val_dataloader)
        print(f"Epoch {epoch + 1}, Average Validation Loss: {val_loss}")
        torch.save(model.state_dict(),'prompt_tuned_model.pt')
        torch.cuda.empty_cache()
    return model


def main ():
    train_data = torch.load('train_data.pt')
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_data = torch.load('val_data.pt')
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

    model = PromptTunedModel()
    model = model.to(device)
    print(f"Number of Added Parameters: {count_parameters(model)}")
    # model = finetuningloop(model,train_dataloader,val_dataloader)
    model.load_state_dict(torch.load('prompt_tuned_model.pt'))

    test_data = torch.load('test_data.pt')
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    train_loss = evaluate_model(model,train_dataloader)
    print(f"Average Training Loss: {train_loss}")
    val_loss = evaluate_model(model,val_dataloader)
    print(f"Average Validation Loss: {val_loss}")
    test_loss = evaluate_model(model,test_dataloader)
    print(f"Average Testing Loss: {test_loss}")

    # predictions, references = generate_predictions(model, test_dataloader)
    # rouge_scores = calculate_rouge(predictions, references)
    # print(f"ROUGE Score on Test Data: {rouge_scores}")
    return


if __name__ == "__main__":
    main()
