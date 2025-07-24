## Advanced Natural Language Processing

- The 'dataset_preprocessing.py' file contains the code for spliting the dataset, tokenizing and it and saving it in 'train_data.pt', 'test_data.pt' and 'val_data.pt' files.

- The command to evaluate the Prompt-Tuned GPT2 model:
    > python3 prompt_tuning.py
    - This outputs the Evaluation Loss for the Training, Testing and Validation Dataset and the ROUGE Score on the testing data.
    - By default, it uses the saved model 'prompt_tuned_model.pt'. If the model is to be Prompt Tuned again, uncomment the lines from main function. 

- The command to evaluate the Prompt-Tuned GPT2 model:
    > python3 traditional_ft.py
    - This outputs the Evaluation Loss for the Training, Testing and Validation Dataset and the ROUGE Score on the testing data.
    - By default, it uses the saved model 'traditional_ft_model.pt'. If the model is to be Fine-Tuned again, uncomment the lines from main function. 

- The command to evaluate the Prompt-Tuned GPT2 model:
    > python3 LoRA.py
    - This outputs the Evaluation Loss for the Training, Testing and Validation Dataset and the ROUGE Score on the testing data.
    - By default, it uses the saved model 'LoRA_model.pt'. If the model is to be Fine Tuned using LoRA again, uncomment the lines from main function. 


- Analysis of performance of different models is present in the Report.
- The pretrained models ".pt" files and the dataset files after splitting then into test, train and validation are uploaded on one drive. 
https://iiitaphyd-my.sharepoint.com/:f:/g/personal/gargi_shroff_research_iiit_ac_in/EiabI8iN6jVFiiWjWZJBIuMBwf9r3i8st1HnSk6GjmAiRA?e=Nr5PE5
