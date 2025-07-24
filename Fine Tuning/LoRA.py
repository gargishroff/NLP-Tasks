import time
import torch
import evaluate
from tqdm import tqdm  
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW

from peft import LoraConfig, get_peft_model

from data_preprocessing import preprocess_function, FineTuningDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
BATCH_SIZE = 4
LR = 5e-5
EPOCHS = 10

def generate_predictions(model, data_loader):
    model.eval()
    predictions, references = [], []
    
    with torch.no_grad():
        for input_ids, attention_mask, label_ids in tqdm(data_loader, desc="Generating Predictions"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Generate predictions using the model
            outputs = model.generate(input_ids, attention_mask=attention_mask) # Adjust max_length if needed
            generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            reference_texts = [tokenizer.decode(label, skip_special_tokens=True) for label in label_ids]
            
            predictions.extend(generated_texts)
            references.extend(reference_texts)
            
    return predictions, references

def compute_rouge(predictions, references):
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=predictions, references=references)
    return results

def evaluate_model (model, data_loader):
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    model.eval()
    total_loss = 0
    with torch.no_grad():
        progress_bar = tqdm (data_loader, desc=f"Evaluating Model", unit="batch", dynamic_ncols=True)
        for input_ids, attention_mask, label_ids in progress_bar:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label_ids = label_ids.to(device)
            outputs = model(input_ids,labels=label_ids)   
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = label_ids[..., 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss += loss.item()

        # Compute average loss
        avg_loss = total_loss / len(data_loader)
    return avg_loss

def finetuningloop(LoRA_model, train_dataloader, val_dataloader):
    optimizer = AdamW(LoRA_model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    initial_memory = torch.cuda.memory_allocated(device)

    for epoch in range(EPOCHS) :
        start_time = time.time()
        LoRA_model.train()
        progress_bar = tqdm (train_dataloader, desc=f"Training Epoch {epoch + 1}", unit="batch", dynamic_ncols=True)
        total_loss = 0
        for input_ids, attention_mask, label_ids in progress_bar:
            input_ids = input_ids.to(device)
            label_ids = label_ids.to(device)
            
            outputs = LoRA_model(input_ids,labels=label_ids)   
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()  
            shift_labels = label_ids[..., 1:].contiguous() 
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(LoRA_model.parameters(), 1.0) 
            optimizer.step()
            total_loss += loss.item()

        epoch_time = time.time()-start_time
        torch.cuda.synchronize()
        epoch_memory = torch.cuda.memory_allocated(device) - initial_memory
        print(f"Epoch {epoch + 1}, Average Training Loss: {total_loss/len(train_dataloader)}, Training Time: {epoch_time: .2f}, GPU Memory Usage: {epoch_memory / (1024**2):.2f} MB")
        torch.save(LoRA_model.state_dict(),'LoRA_model.pt')
        
    val_loss = evaluate_model(LoRA_model,val_dataloader)
    print(f"Average Validation Loss: {val_loss}")
    return LoRA_model

def main():
    train_data = torch.load('train_data.pt')
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_data = torch.load('val_data.pt')
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    for param in model.parameters():
        param.requires_grad() = False

    lora_config = LoraConfig(
        r=128,
        lora_alpha=32,
        lora_dropout=0.2,
        target_modules=["attn.c_attn", "attn.c_proj","mlp.c_fc", "mlp.c_proj"],
    )

    LoRA_model = get_peft_model(model, lora_config)
    LoRA_model.to(device)
    # LoRA_model = finetuningloop(LoRA_model,train_dataloader,val_dataloader)
    LoRA_model.load_state_dict(torch.load('LoRA_model.pt'))

    test_data = torch.load('test_data.pt')
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    train_loss = evaluate_model(model,train_dataloader)
    print(f"Average Training Loss: {train_loss}")
    val_loss = evaluate_model(model,val_dataloader)
    print(f"Average Validation Loss: {val_loss}")
    test_loss = evaluate_model(model,test_dataloader)
    print(f"Average Testing Loss: {test_loss}")

    # predictions, references = generate_predictions(model, test_dataloader)
    # rouge_scores = compute_rouge(predictions, references)
    # print(f"Average ROUGE Scores on Test Dataset: {rouge_scores}")

    return

if __name__ == "__main__":
    main()