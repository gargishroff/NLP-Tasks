import torch
import numpy as np
from time import time
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BitsAndBytesConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

memory_usage = []
latency = []
perplexity_scores = []

def plot_results():
    quantization_methods = ['Original Model', 'Bits and Bytes - 8 bits', 'Bits and Bytes - 4 bits', 'NF4']

    # Bar positions
    x = np.arange(len(quantization_methods))  # the label locations
    width = 0.25  # the width of the bars

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width, memory_usage, width, label='Memory Usage (MB)')
    ax.bar(x, latency, width, label='Latency (ms)')
    ax.bar(x + width, perplexity_scores, width, label='Perplexity')

    # Labels and title
    ax.set_xlabel('Quantization Methods')
    ax.set_ylabel('Metrics')
    ax.set_title('Comparison of Metrics Across Quantization Methods')
    ax.set_xticks(x)
    ax.set_xticklabels(quantization_methods)
    ax.legend()

    plt.tight_layout()
    plt.show()

class ModelEvaluator:
    def __init__(self, model):
        self.model = model
        self.results = {}

    def compute_perplexity(self, dataset, max_samples: int = 3000) -> float:
        total_loss = 0
        total_tokens = 0
        latencies = []
        with torch.no_grad():
            for i, example in enumerate(dataset):
                if i >= max_samples:
                    break
                    
                inputs = tokenizer(
                    example['sentence'],
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                
                start_time = time()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                latency = time() - start_time
                latencies.append(latency)
                
                # Calculate loss only for non-padding tokens
                loss = outputs.loss
                if not torch.isnan(loss) and not torch.isinf(loss):
                    num_tokens = attention_mask.sum().item()
                    total_loss += loss.item() * num_tokens
                    total_tokens += num_tokens
        
        mean_latency = np.mean(latencies[10:])
        if total_tokens == 0:
            return float('nan')
            
        avg_loss = total_loss / total_tokens
        return np.exp(avg_loss), mean_latency

    def measure_model_memory(self):
        total_memory = 0
        for param in self.model.parameters():
            total_memory += param.nelement() * param.element_size()
        return total_memory/(1024**2)

    def run_evaluation(self, dataset, experiment_name: str):
        memory_mb = self.measure_model_memory()
        perplexity, latency_ms = self.compute_perplexity(dataset)
        latency_ms = latency_ms * 1000
        self.results[experiment_name] = {
            'memory_mb': memory_mb,
            'latency_ms': latency_ms,
            'perplexity': perplexity
        }

        memory_usage.append(memory_mb)
        latency.append(latency_ms)
        perplexity_scores.append(perplexity)

        print(f"\nUsing - {experiment_name}")
        print(f"Memory Usage: {memory_mb:.2f} MB")
        print(f"Inference Latency: {latency_ms:.2f} ms")
        print(f"Perplexity: {perplexity:.2f}")

def main ():
    dataset = load_dataset('ptb_text_only', split='test')

    model_original = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    evaluator_original = ModelEvaluator(model_original)
    evaluator_original.run_evaluation(dataset, 'Original Model')
    del model_original
    torch.cuda.empty_cache()

    model_8bit = GPT2LMHeadModel.from_pretrained(
        'gpt2',
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        ),
        device_map = 'auto',
    )
    evaluator_8bit = ModelEvaluator(model_8bit)
    evaluator_8bit.run_evaluation(dataset, 'Bits and Bytes - 8 bits')
    del model_8bit
    torch.cuda.empty_cache()

    model_4bit = GPT2LMHeadModel.from_pretrained(
        'gpt2',
        device_map = 'auto',
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    )
    evaluator_4bit = ModelEvaluator(model_4bit)
    evaluator_4bit.run_evaluation(dataset, 'Bits and Bytes - 4 bits')
    del model_4bit
    torch.cuda.empty_cache()

    model_nf4 = GPT2LMHeadModel.from_pretrained(
        'gpt2',
        device_map = 'auto',
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    )
    evaluator_nf4 = ModelEvaluator(model_nf4)
    evaluator_nf4.run_evaluation(dataset, 'NF4')
    del model_nf4
    torch.cuda.empty_cache()

    plot_results()

if __name__ == "__main__":
    main()