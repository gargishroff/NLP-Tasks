## Intro to Natural Language Processing

- The command for running the python script using Feed Forward Neural Network Model for POS Tagging
    > python3 pos_tagger.py -f
    - This outputs the result of different Evaluation Metrics (f1 score, accuracy, recall) on the test data loaded from a pretrained model save in 'ffnn_model.pt'. 
    - It futher generates a prompt on the command line, for an input sentence and predicts the Part of Speech tags for the words in the sentence.

- The command for running the python script using Recurrent Neural Network Model for POS Tagging
    > python3 pos_tagger.py -r
    - This outputs the result of different Evaluation Metrics (f1 score, accuracy, recall) on the test data loaded from a pretrained model save in 'ffnn_model.pt'. 
    - It futher generates a prompt on the command line, for an input sentence and predicts the Part of Speech tags for the words in the sentence.

- The code by default uses the pretrained model present in the zip file, if we want to train the model again then, uncomment that part from the main function.
- A analysis.ipynb file is also added in the folder which contains the results of various evalutaion metrics and graphs as outputs for different hyperparameters. The graphs and the outputs are analysed in the report.
