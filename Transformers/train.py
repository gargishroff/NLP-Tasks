import re
import nltk
import utils
import encoder
import decoder
import torch
import pickle
import test
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def preprocess_text(file_path):
    tokenized_data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            rm_punct = re.sub(r'[^\w\s]','',line)
            tokens = nltk.word_tokenize(rm_punct)
            tokenized_data.append(tokens)
    return tokenized_data

def get_vocab(data):
    vocab = {'<pad>', '<unk>'}
    for sentence in data:
        vocab.update(word.lower() for word in sentence)
    vocab_dict = {word: idx for idx, word in enumerate(sorted(vocab))}
    return vocab_dict    

def generate_model_data(data,vocab):
    model_data = []
    for sentence in data:
        i = 0
        sent_idx = []
        for word in sentence:
            i += 1
            if word in vocab:
                sent_idx.append(vocab[word])
            else:
                sent_idx.append(vocab['<unk>'])
            if i == utils.SEQ_LEN:
                break
        while i != utils.SEQ_LEN:
            sent_idx.append(vocab['<pad>'])
            i += 1
        model_data.append(torch.tensor(sent_idx))
    model_data_tensor = torch.stack(model_data)
    return model_data_tensor

def get_transformer(vocab_en_size,vocab_fr_size):
    pos = utils.PositionalEncoding()
    input_embd = utils.Embeddings(vocab_en_size)
    output_embd = utils.Embeddings(vocab_fr_size)
    encoder_blocks = []
    decoder_blocks = []
    for i in range (0,utils.NUM_BLOCKS):
        attention = utils.MultiHeadAttention()
        ffl = utils.FeedForward()
        encd = encoder.Encoder(attention,ffl)
        encoder_blocks.append(encd)

        self_attention = utils.MultiHeadAttention()
        cross_attention = utils.MultiHeadAttention()
        ffld = utils.FeedForward()
        decd = decoder.Decoder(self_attention,cross_attention,ffld)
        decoder_blocks.append(decd)

    EncoderFinal = encoder.CombinedEncoder(nn.ModuleList(encoder_blocks))
    DecoderFinal = decoder.CombinedDecoder(nn.ModuleList(decoder_blocks))
    Projection_Layer = decoder.ProjectOutputs(vocab_fr_size)

    transformer = utils.Transformer(EncoderFinal,DecoderFinal,pos,vocab_fr_size,input_embd,output_embd,Projection_Layer).to(utils.device)
    for parameter in transformer.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)
    return transformer

def training_loop (encd_ipt,decd_ipt,vocab_eng,vocab_fr,test_data,test_fr):
    model = get_transformer(len(vocab_eng),len(vocab_fr)).to(utils.device)
    optimizer = optim.Adam(model.parameters(),lr=utils.LR, eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab_eng['<pad>'], label_smoothing=0.1).to(utils.device)

    training_data = TensorDataset(encd_ipt,decd_ipt)
    train_loader = DataLoader(training_data, batch_size=utils.BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(utils.EPOCHS):
        torch.cuda.empty_cache()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        
        for encd,decd in progress_bar:
            encd = encd.to(utils.device)
            decd = decd.to(utils.device)
            encd_mask = (encd != vocab_eng['<pad>']).unsqueeze(1).unsqueeze(2)
            decd_padding = (decd != vocab_fr['<pad>']).unsqueeze(1).unsqueeze(2)
            mask = generate_mask(decd.size(1))
            decd_mask = decd_padding & mask

            encoder_output = model.encode(encd, encd_mask).to(utils.device)
            decoder_output = model.decode(encoder_output, encd_mask, decd, decd_mask).to(utils.device)
            proj_output = model.final_projection(decoder_output).to(utils.device)
            # print(proj_output.shape)

            loss = criterion(proj_output.view(-1, proj_output.size(-1)), decd.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

    model.eval()
    predictions = []

    with torch.no_grad():
        for encd,decd in test_data:
            encd = encd.unsqueeze(0).to(utils.device)
            # print(encd.shape)   [(1,45)]
            encd_mask = (encd != vocab_fr['<pad>']).unsqueeze(1).unsqueeze(2)
            decd = decd.unsqueeze(0).to(utils.device)
            # print(decd.shape)   [(1,45)]
            decd_padding = (decd != vocab_fr['<pad>']).unsqueeze(1).unsqueeze(2)
            mask = generate_mask(decd.size(1))
            decd_mask = decd_padding & mask
            encoder_output = model.encode(encd, encd_mask).to(utils.device)
            decoder_output = model.decode(encoder_output, encd_mask, decd, decd_mask).to(utils.device)
            proj_output = model.final_projection(decoder_output).to(utils.device)
            predicted_indices = proj_output.argmax(dim=-1)
            # print(predicted_indices.shape)   [(1,45)]
            predictions.append(predicted_indices.squeeze(0).tolist())

    decoded_predictions = []
    for pred in predictions:
        decoded_sentence = [idx for idx in pred if idx != vocab_fr['<pad>']]
        # Convert indices back to words using the vocabulary
        decoded_sentence_words = [list(vocab_fr.keys())[list(vocab_fr.values()).index(idx)] for idx in decoded_sentence]
        decoded_predictions.append(" ".join(decoded_sentence_words))

    decoded_references = [" ".join(sentence) for sentence in test_fr]
    bleu_scores = test.calculate_bleu(decoded_predictions, decoded_references)
    rouge_scores = test.calculate_rouge(decoded_predictions, decoded_references)

    average_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    print(f"Average Bleu Score : {average_bleu}")

    with open("testbleu.txt", "w") as file:
        for i, (bleu, rouge) in enumerate(zip(bleu_scores, rouge_scores)):
            file.write(f"Sentence {i+1}: {bleu:.2f} ROUGE-L: {rouge['rougeL']:.2f}\n")

    print("Results written to evaluation_results.txt")
    torch.save(model.state_dict(),'model.pt')
    return model

def generate_mask(mask_size):
    mask = torch.tril(torch.ones((mask_size,mask_size), device=utils.device)).unsqueeze(0).unsqueeze(0).to(torch.bool)
    return mask

def main ():
    data_eng = preprocess_text("dataset/train.en")
    vocab_eng = get_vocab(data_eng)
    data_fr = preprocess_text("dataset/train.fr")
    vocab_fr = get_vocab(data_fr)

    # encd_ipt = generate_model_data(data_eng,vocab_eng)
    # torch.save(encd_ipt,'encd_ipt.pt')
    # decd_ipt = generate_model_data(data_fr,vocab_fr)
    # torch.save(decd_ipt,'decd_ipt.pt')

    encd_ipt = torch.load('encd_ipt.pt')
    decd_ipt = torch.load('decd_ipt.pt')

    test_eng = preprocess_text("dataset/test.en")
    test_fr = preprocess_text("dataset/test.fr")
    test_ipt = generate_model_data(test_eng,vocab_eng)
    test_opt = generate_model_data(test_fr,vocab_fr)
    zipped_data = list(zip(test_ipt, test_opt))

    transformer_model = training_loop(encd_ipt,decd_ipt,vocab_eng,vocab_fr,zipped_data,test_fr)
    torch.save(transformer_model.state_dict(),'transformer_model.pt')

    return

if __name__ == "__main__":
    main()