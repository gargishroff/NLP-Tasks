import torch
import torch.nn as nn
import train
import utils

import sacrebleu
from rouge_score import rouge_scorer

def calculate_bleu(predictions, references):
    """Calculates BLEU score for each prediction."""
    bleu_scores = []
    for pred, ref in zip(predictions, references):
        bleu = sacrebleu.sentence_bleu(pred, [ref])
        bleu_scores.append(bleu.score)
    return bleu_scores

def calculate_rouge(predictions, references):
    """Calculates ROUGE scores for each prediction."""
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = []
    for pred, ref in zip(predictions, references):
        scores = rouge_scorer_obj.score(ref, pred)
        rouge_scores.append({
            'rougeL': scores['rougeL'].fmeasure
        })
    return rouge_scores

def generate_predictions(model, test_data, vocab_fr):
    """Generates translations for the test dataset using the trained model."""
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
            mask = train.generate_mask(decd.size(1))
            decd_mask = decd_padding & mask
            encoder_output = model.encode(encd, encd_mask).to(utils.device)
            decoder_output = model.decode(encoder_output, encd_mask, decd, decd_mask).to(utils.device)
            proj_output = model.final_projection(decoder_output).to(utils.device)
            predicted_indices = proj_output.argmax(dim=-1)
            # print(predicted_indices.shape)   [(1,45)]
            predictions.append(predicted_indices.squeeze(0).tolist())
    return predictions

def main ():
    data_eng = train.preprocess_text("dataset/train.en")
    vocab_eng = train.get_vocab(data_eng)
    data_fr = train.preprocess_text("dataset/train.fr")
    vocab_fr = train.get_vocab(data_fr)
    model = train.get_transformer(len(vocab_eng),len(vocab_fr))
    model.load_state_dict(torch.load('transformer_model.pt'))

    test_eng = train.preprocess_text("dataset/test.en")
    test_fr = train.preprocess_text("dataset/test.fr")
    test_ipt = train.generate_model_data(test_eng,vocab_eng)
    test_opt = train.generate_model_data(test_fr,vocab_fr)
    zipped_data = list(zip(test_ipt, test_opt))

    predictions = generate_predictions(model, zipped_data, vocab_fr)

    # Convert predictions and references from indices to words
    decoded_predictions = []
    for pred in predictions:
        decoded_sentence = [idx for idx in pred if idx != vocab_fr['<pad>']]
        # Convert indices back to words using the vocabulary
        decoded_sentence_words = [list(vocab_fr.keys())[list(vocab_fr.values()).index(idx)] for idx in decoded_sentence]
        decoded_predictions.append(" ".join(decoded_sentence_words))

    decoded_references = [" ".join(sentence) for sentence in test_fr]
    bleu_scores = calculate_bleu(decoded_predictions, decoded_references)
    rouge_scores = calculate_rouge(decoded_predictions, decoded_references)

    average_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    print(f"Average Bleu Score : {average_bleu}")

    with open("testbleu.txt", "w") as file:
        for i, (bleu, rouge) in enumerate(zip(bleu_scores, rouge_scores)):
            file.write(f"Sentence {i+1}: {bleu:.2f} ROUGE-L: {rouge['rougeL']:.2f}\n")

    print("Results written to evaluation_results.txt")
    return

if __name__ == "__main__":
    main()