# Advanced Natural Language Processing
## Assignment 3 - Report

### Theory Questions 
1. **Concept of Soft Prompts - How does the introduction of ”soft prompts” address the limitations of discrete text prompts in large language models? Why might soft prompts be considered a more flexible and efficient approach for task-specific conditioning?**
    - Traditional discrete text prompts have several limitations like discreteness constraint i.e, text prompts are restricted to tokens from the model's vocabulary and each token must be a valid word/subword unit which limits the expressiveness and optimization potential.
    - Discrete prompts often require many tokens to specify the task. This consumes significant context window space and processing longer prompts increases computational overhead.
    - Soft prompts address these limitations of Discrete Text Prompts by providing Continuous Representations i.e, Instead of discrete tokens, soft prompts use learned continuous vectors, these vectors can be optimized directly in the embedding space and they are not constrained by vocabulary or grammatical rules. Hence they can encode task-specific information more densely. 
    - Soft prompt vectors are trained via gradient descent so they can adapt to capture precise task requirements since the optimization process can find more effective task specifications than manual prompt engineering.
    - Soft Prompts typically require fewer vectors that equivalent text prompts which save the context window space for actual inputs, reducing the computational costs.


2. **Scaling and Efficiency in Prompt Tuning - How does the efficiency of prompt tuning relate to the scale of the language model? Discuss the implications of this relationship for future developments in large-scale language models and their adaptability to specific tasks.**
    - Prompt tuning typically requires optimizing only a small subset of parameters—usually the prompt embeddings—rather than the entire model. This makes it feasible to adapt even very large models to new tasks with minimal added computational overhead. 
    - Prompt tuning enables more effective use of LLMs by learning nuanced task-specific representations in the prompt embeddings without altering the model’s internal weights. As a result, LLMs can be quickly and efficiently adapted to new tasks by learning new prompts rather than re-optimizing the entire model. Therefore, prompt tuning supports scalable and more frequent adaptation of LLMs across multiple tasks in production settings.
    - The efficiency gains of prompt tuning encourage future architectures to support modular task adaptation, enabling LLMs to be more flexible across domains and applications by simply swapping prompts rather than entire models. The models can therefore be more easily personalized for individual users or tasks, even allowing for real-time responsiveness in multi-task environments.


3. **Understanding LoRA - What are the key principles behind Low-Rank Adaptation (LoRA) in fine-tuning large language models? How does LoRA improve upon traditional fine-tuning methods regarding efficiency and performance?**
    - Low-Rank Adaptation (LoRA) is an efficient apprach to fine-tune large language models by adapting only a subset of their parameters, through low-rank representations of model weight updates.
    - In LoRA, weight updates are approximated as low-rank matrices. Rather than adjusting the entire set of model parameters, LoRA inserts low-rank matrices that capture the essential information needed for task adaptation. These low-rank matrices have a much smaller parameter count than the original full-rank model weights, leading to substantial memory savings and computational efficiency.
    - LoRA introduces these low-rank matrices in parallel to the pre-trained model's weight matrices, which allows it to achieve task adaptation without slowing down inference. This is in contrast to traditional fine-tuning, where updating all weights can introduce inference overhead and latency.

4. **Theoretical Implications of LoRA - Discuss the theoretical implications of introducing low-rank adaptations to the parameter space of large language models. How does this affect the expressiveness and generalization capabilities of the model compared to standard fine-tuning?**
    - LoRA fine-tunes models by updating only a small set of low-rank matrices, leaving the main model parameters intact.
    - Large language models typically have a high level of redundancy in their parameter spaces, which implies that even complex adjustments can often be approximated in a low-dimensional subspace. Low-rank adaptations exploit this redundancy, effectively condensing the parameter space by focusing on lower-dimensional embeddings.
    - Traditional fine-tuning potentially offers higher expressiveness by adjusting all parameters, LoRA achieves efficient task-specific adaptation within a constrained subspace. This minor expressiveness trade-off is generally offset by the practical benefits in efficiency and generalization.
    - Since LoRA keeps the original model weights frozen, the model's general-purpose knowledge remains intact, enhancing its ability to generalize across tasks. LoRA acts as a form of regularization, limiting the extent to which the model can “specialize” to a particular task, which theoretically reduces the risk of catastrophic forgetting.

### Prompt Tuning
- Hyperparameters Used - Batch Size = 4, Learning Rate = 5e-5, Num Prompts = 5, Epochs = 5
1. **Evaluation Loss**
    - Training Data - 7.2044
    - Validation Data - 7.2568
    - Testing Data - 7.2432
2. **GPU Memory Usage** - 1027.58 MB
3. **Time For Training** - 1449.67 per epoch
4. **ROUGE Score on Testing Dataset** - 0.157
5. **Additional Parameters** - 768*5 = 3840

### Traditional Fine-Tuning (Last Layer Only)
- Hyperparameters Used - Batch Size = 4, Learning Rate = 5e-5, Epochs = 10
1. **Evaluation Loss**
    - Training Data - 6.9779
    - Validation Data - 7.4764
    - Testing Data - 7.4672
2. **GPU Memory Usage** - 1850.67 MB
3. **Time For Training** - 1553.40 per epoch
4. **ROUGE Score on Testing Dataset** - 0.112
5. **Additional Parameters** - No additional Parameters


### LoRA Fine_Tuning 
- Hyperparameters Used - Batch Size = 4, Learning Rate = 5e-5, Epochs = 10, alpha = 32, dropout rate = 0.2, r = 128
1. **Evaluation Loss**
    - Training Data - 7.1104
    - Validation Data - 7.1818
    - Testing Data - 7.1789
2. **GPU Memory Usage** - 1641.31 MB
3. **Time For Training** - 1601.46 per epoch
4. **ROUGE Score on Testing Dataset** - 0.186
5. **Additional Parameters** - 4*(768*128+768*128)

### Analysis 
1. **Evaluation Loss** 
- LoRA Fine-Tuning achieves the best balance across datasets, while Traditional Fine-Tuning has the lowest training loss but suffers from a gap in validation and testing, indicating possible overfitting. Prompt Tuning, while stable, has higher losses, suggesting it’s a simpler approach with limited capacity to learn complex patterns.
2. **GPU Memory Usage** 
- Prompt Tuning utilizes the least GPU memory (1027.58 MB), making it the most efficient in terms of resource usage. This lightweight approach is likely due to the limited parameter tuning, with only prompt parameters adjusted rather than model weights.
- Traditional Fine-Tuning (Last Layer Only) uses 1850.67 MB, likely because fine-tuning even a single layer requires the model to maintain gradients for a larger set of parameters than Prompt Tuning.
- LoRA Fine-Tuning uses a moderate 1641.31 MB, as LoRA tunes only specific low-rank matrices, making it more memory-efficient than full fine-tuning but more intensive than Prompt Tuning.
3. **Time for Training per Epoch**
- Prompt Tuning takes only 1449.67 seconds per epoch, the fastest of the three, consistent with its limited parameter updates.
- Traditional Fine-Tuning (Last Layer Only) takes slightly longer at 1553.40 seconds per epoch, due to gradient updates across a larger subset of parameters in the last layer.
- LoRA Fine-Tuning requires the longest per epoch (1601.46 seconds), due to the computational complexity of the low-rank matrix adjustments in LoRA.
4. **ROUGE Score on Testing Dataset**
- LoRA Fine-Tuning shows the best performance on ROUGE, making it the most effective approach in terms of quality output. Prompt Tuning performs moderately well, while Traditional Fine-Tuning underperforms on this metric.

