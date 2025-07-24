## Advanced Natural Language Processing

- The command for running the python script for Neural Language Model
    > python3 nnlm.py
    - This outputs the Average Perplexity on the test data and also a file "2022114009-LM1-test.txt" which contains sentence wise perplexity across sentences of testing data.

- The command for running the python script for RNN based Language Model
    > python3 rnn.py
    - This outputs the Average Perplexity on the test data and also a file "2022114009-LM2-test.txt" which contains sentence wise perplexity across sentences of testing data.

- The command for running the python script for Transformer based Language Model
    > python3 rnn.py
    - This outputs the Average Perplexity on the test data and also a file "2022114009-LM3-test.txt" which contains sentence wise perplexity across sentences of testing data.

- The code by default uses the pretrained model present in the zip file, if we want to train the model again then, uncomment that part from the main function.
- The original dataset is split into train, test and validation datasets using a python script 'data_preprocess.py' which is present in the dataset folder. This data is then used by the language models for training and testing purpose.
- Analysis of performance of different models is present in the Report.
- 100 dimension pretrained glove embeddings are used.
- The pretrained models ".pt" files, ".pkl" files (containing word2idx dictionary, tokenized training data and glove embedding files) and the dataset files after splitting then into test, train and validation are uploaded on one drive. The link is attached -
https://iiitaphyd-my.sharepoint.com/:f:/g/personal/gargi_shroff_research_iiit_ac_in/Eo-AeusboLlIkuPmGQbN1QUBLo3pHWpnZyuoMiQ8MUtyEA?e=Vo3qC6

