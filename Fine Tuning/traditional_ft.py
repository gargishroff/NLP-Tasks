import time
import torch
import evaluate
from tqdm import tqdm  
import torch.nn as nn
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW

from data_preprocessing import preprocess_function, FineTuningDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
BATCH_SIZE = 4
NUM_PROMPTS = 5
LR = 5e-5
EPOCHS = 10

class TraditionalFineTunedModel (nn.Module):
    def __init__(self):
        super(TraditionalFineTunedModel, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        for param in self.gpt2.parameters():
            param.requires_grad = False

        for param in self.gpt2.lm_head.parameters():
            param.requires_grad = True

    def forward (self, input_ids, attention_mask, labels):
        return self.gpt2(input_ids=input_ids, attention_mask=attention_mask, labels = labels)

def calculate_rouge(model, data_loader):
    model.eval()
    rouge = evaluate.load('rouge')
    
    generated_texts = []
    reference_texts = []

    with torch.no_grad():
        for input_ids, attention_mask, label_ids in tqdm(data_loader, desc="Calculating ROUGE Score"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label_ids = label_ids.to(device)

            # Generate model output
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,max_length=label_ids.size(1))
            
            # Decode generated and reference sequences
            generated_texts.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
            reference_texts.extend(tokenizer.batch_decode(label_ids, skip_special_tokens=True))

    # Calculate ROUGE scores for the entire dataset at once
    rouge_scores = rouge.compute(predictions=generated_texts, references=reference_texts)
    
    # Return average ROUGE scores
    return rouge_scores

def evaluate_model (model,data_loader):
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    model.eval()

    total_loss = 0
    with torch.no_grad():
        progress_bar = tqdm (data_loader, desc=f"Evaluating Model", unit="batch", dynamic_ncols=True)
        for input_ids, attention_mask, label_ids in progress_bar:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label_ids = label_ids.to(device)
            
            outputs = model(input_ids,attention_mask,label_ids)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = label_ids[..., 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss += loss.item()

        # Compute average loss
        avg_loss = total_loss / len(data_loader)
    return avg_loss


def finetuningloop (model, train_dataloader,val_dataloader):
    optimizer = AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
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
            
            outputs = model(input_ids,attention_mask,label_ids)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()  
            shift_labels = label_ids[..., 1:].contiguous()    
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
        torch.save(model.state_dict(),'traditional_ft_model.pt')
        torch.cuda.empty_cache()
    return model


def main ():
    train_data = torch.load('train_data.pt')
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_data = torch.load('val_data.pt')
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

    model = TraditionalFineTunedModel()
    model = model.to(device)
    # model = finetuningloop(model,train_dataloader,val_dataloader)
    model.load_state_dict(torch.load('traditional_ft_model.pt'))

    test_data = torch.load('test_data.pt')
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    train_loss = evaluate_model(model,train_dataloader)
    print(f"Average Training Loss: {train_loss}")
    val_loss = evaluate_model(model,val_dataloader)
    print(f"Average Validation Loss: {val_loss}")
    test_loss = evaluate_model(model,test_dataloader)
    print(f"Average Testing Loss: {test_loss}")

    # rouge_scores = calculate_rouge(model, test_dataloader)
    # print(f"Average ROUGE Scores on Test Dataset: {rouge_scores}")

    return


if __name__ == "__main__":
    main()
