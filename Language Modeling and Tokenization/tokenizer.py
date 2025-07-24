import re
import argparse

def Identify_URLS(tokenized_words):
    pattern = r'^http|www'
    for i, sentence_words in enumerate(tokenized_words):
        for j, word in enumerate(sentence_words):
            if re.match(pattern, word):
                tokenized_words[i][j] = '<URL>'
    return tokenized_words

def Identify_Mails(tokenized_words):
    pattern = r'\w[^\s{}()\[\]]*@[^\s{}()\[\]]*\w'
    for i, sentence_words in enumerate(tokenized_words):
        for j, word in enumerate(sentence_words):
            if re.match(pattern, word):
                tokenized_words[i][j] = '<MAILID>'
    return Identify_URLS(tokenized_words)

def Identify_Mentions(tokenized_words):
    pattern = r'@[^\s{}()\[\]]*\w'
    for i, sentence_words in enumerate(tokenized_words):
        for j, word in enumerate(sentence_words):
            if re.match(pattern, word):
                tokenized_words[i][j] = '<MENTION>'
    return Identify_Mails(tokenized_words)

def Identify_Hashtags(tokenized_words):
    pattern = r'#[^\s{}()\[\]]*\w'
    for i, sentence_words in enumerate(tokenized_words):
        for j, word in enumerate(sentence_words):
            if re.match(pattern, word):
                tokenized_words[i][j] = '<HASHTAG>'
    return Identify_Mentions(tokenized_words)

def Identify_Num (tokenized_words):
    pattern = r'\d+\.\d+(?=\s|$)|\d[\d+,]*(?=\s|$)|\d+(?=\s|$)'
    for i, sentence_words in enumerate(tokenized_words):
        for j, word in enumerate(sentence_words):
            if re.match(pattern, word):
                tokenized_words[i][j] = '<NUM>'
    return Identify_Hashtags(tokenized_words)

def Identify_Punct(tokenized_words):
    for i, sentence_words in enumerate(tokenized_words):
        for j, word in enumerate(sentence_words):
            if len(word) == 1:
                if re.match(r'^[^\w(){}\[\]]', word):
                    tokenized_words[i][j] = '<PUNCT>'
    return Identify_Num(tokenized_words)

def words_tokenize(sentences):
    tokenized_words = []
    for sentence in sentences:
        words = re.findall(r"'s|@[^\s{}()\[\]]*\w|#[^\s{}()\[\]]*\w|\w[^'\s{}()\[\]]*\w|\w+|\S",sentence)
        tokenized_words.append(words)
    # return Identify_Punct(tokenized_words)
    return Identify_Num(tokenized_words)

def sentence_tokenize(doc):
    pattern = re.compile(r'(?<![A-Z]\.|[0-9]\.)(?<!Mr\.|Dr\.)(?<!Mrs\.)(?<!\.\.)(?<=\.|\?|\!)\s' + '|' + r'(?<=\.\"|\?\"|\!\")\s' + '|' + r'\n\n+')
    ignore = r'\n'
    snts = pattern.split(doc)
    sentences = []
    for snt in snts:
        sentence = re.sub(ignore,' ',snt)
        if sentence:
            sentences.append(sentence)
    return words_tokenize(sentences)

def tokenizer(doc):
    return sentence_tokenize(doc)

def tokenize_file(input_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()

    tokenized_text = tokenizer(text)
    return tokenized_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    args = parser.parse_args()
    input_file = args.input_file

    try:
        tokenized_text = tokenize_file(input_file)
        print(tokenized_text)
    except FileNotFoundError:
        print("Error: The input file is not found.")
    except Exception as e:
        print(f"An unexpected error occured.{e}")


if __name__ == "__main__":
    main()
