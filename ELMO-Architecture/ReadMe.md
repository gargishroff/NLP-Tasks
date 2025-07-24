## Intro to Natural Language Processing

- Note : Only 15000 sentences from the training data are used for classification and generating word embeddings
- The command for generating word embeddings using ELMO -
    > python3 ELMO.py
    - This results in generation of a file 'bilstm.pt' which contains embeddings for all the words in the vocab.
    - It also stores the vocab (word to index dictionary) in a json file. 
- The command for performing Downstream classification task using word embeddings generated using ELMO-
    > python3 classification.py
    - This is generates the scores of various Evaluation metrics on the testing data after training the model.

- The code by default uses the pretrained model present in the zip files, the link for the folder containing pretrained models is present in the ReadMe.
https://iiitaphyd-my.sharepoint.com/:f:/g/personal/gargi_shroff_research_iiit_ac_in/EgkWQ0bPRt5DrrtMXJP9kAIBdD0uErJ9ApvaYrldH8TAng?e=MYK1f7

