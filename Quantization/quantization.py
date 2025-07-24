import torch
import numpy as np
from time import time
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelQuantizer:
    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.original_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        self.quantized_bits = None
        self.quantized_params = set()  # Track which parameters are quantized

    def get_tensor_memory(self, tensor: torch.Tensor, bits: Optional[int] = None) -> int:
        if bits is None:
            # Use actual tensor memory
            return tensor.nelement() * tensor.element_size()
        else:
            # Calculate memory after quantization
            return tensor.nelement() * (bits // 8)

    def quantize_tensor(self, tensor: torch.Tensor, bits: int = 8) -> torch.Tensor:
        if tensor.numel() == 0:
            return tensor
        
        # Store original dtype for proper conversion
        original_dtype = tensor.dtype
        tensor = tensor.float()

        # Calculate quantization range
        qmin, qmax = -(2**(bits-1)), 2**(bits-1)-1
        scale = (torch.max(tensor) - torch.min(tensor)) / (qmax - qmin)
        zero_point = qmin - torch.min(tensor) / scale
        
        quantized = torch.clamp(torch.round(tensor / scale + zero_point), qmin, qmax) # Quantize
        dequantized = (quantized - zero_point) * scale # Dequantize
        return dequantized.to(original_dtype) # Convert back to original dtype

    def quantize_whole_model(self, bits: int = 8):
        self.quantized_bits = bits
        self.quantized_params = set()  
        
        quantized_state_dict = {}
        for name, param in self.model.state_dict().items():
            if param.dtype in [torch.float32, torch.float16]:
                quantized_state_dict[name] = self.quantize_tensor(param, bits)
                self.quantized_params.add(name)
            else:
                quantized_state_dict[name] = param

        self.model.load_state_dict(quantized_state_dict)

    def quantize_selective_component(self, component_names: List[str], bits: int = 8):
        self.quantized_bits = bits
        self.quantized_params = set() 
        quantized_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
        for name, param in self.model.state_dict().items():
            if any(component in name for component in component_names):
                if param.dtype in [torch.float32, torch.float16]:
                    quantized_state_dict[name] = self.quantize_tensor(param, bits)
                    self.quantized_params.add(name)

        self.model.load_state_dict(quantized_state_dict)

    def reset_model(self):
        self.model.load_state_dict({k: v.to(device) for k, v in self.original_state_dict.items()})
        self.quantized_bits = None
        self.quantized_params = set()

class ModelEvaluator:
    def __init__(self, model_quantizer: ModelQuantizer):
        self.model_quantizer = model_quantizer
        self.results = {}

    def measure_memory(self) -> float:
        total_bytes = 0
        for name, param in self.model_quantizer.model.named_parameters():
            if name in self.model_quantizer.quantized_params:
                # Calculate memory for quantized parameters
                total_bytes += self.model_quantizer.get_tensor_memory(
                    param, self.model_quantizer.quantized_bits)
            else:
                # Calculate memory for non-quantized parameters
                total_bytes += self.model_quantizer.get_tensor_memory(param)      
        return total_bytes / (1024 * 1024)  

    def compute_perplexity(self, dataset, max_samples: int = 3000) -> float:
        total_loss = 0
        total_tokens = 0
        latencies = []
        with torch.no_grad():
            for i, example in enumerate(dataset):
                if i >= max_samples:
                    break
                    
                inputs = self.model_quantizer.tokenizer(
                    example['sentence'],
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                
                start_time = time()
                outputs = self.model_quantizer.model(
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

    def run_evaluation(self, dataset, experiment_name: str):
        memory_mb = self.measure_memory()
        perplexity, latency_ms = self.compute_perplexity(dataset)
        latency_ms = latency_ms * 1000
        
        self.results[experiment_name] = {
            'memory_mb': memory_mb,
            'latency_ms': latency_ms,
            'perplexity': perplexity
        }

    def plot_results(self):
        metrics = ['memory_mb', 'latency_ms', 'perplexity']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            values = [self.results[exp][metric] for exp in self.results]
            sns.barplot(x=list(self.results.keys()), y=values, ax=axes[i])
            axes[i].set_title(metric.replace('_', ' ').title())
            axes[i].tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        return fig

def main():
    quantizer = ModelQuantizer()
    evaluator = ModelEvaluator(quantizer)
    dataset = load_dataset('ptb_text_only', split='test')
    
    # Run original model
    evaluator.run_evaluation(dataset, 'Original Model')

    # Run whole model quantization
    quantizer.quantize_whole_model(bits=8)
    evaluator.run_evaluation(dataset, 'Whole Model Quantization')

    # Run selective quantization
    quantizer.reset_model()
    attention_components = ['attn', 'c_attn', 'c_proj']
    quantizer.quantize_selective_component(attention_components, bits=8)
    evaluator.run_evaluation(dataset, 'Selective Quantization')

    print("\nFinal Results:")
    for exp_name, metrics in evaluator.results.items():
        print(f"\n{exp_name}:")
        print(f"Memory Usage: {metrics['memory_mb']:.2f} MB")
        print(f"Inference Latency: {metrics['latency_ms']:.2f} ms")
        print(f"Perplexity: {metrics['perplexity']:.2f}")

    fig = evaluator.plot_results()
    plt.show()

if __name__ == "__main__":
    main()
