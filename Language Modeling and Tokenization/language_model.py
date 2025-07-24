import re
import argparse
from scipy.stats import linregress
import numpy as np
import math

def Identify_URLS(tokenized_words):
    pattern = r'^http|www'
    for i, sentence_words in enumerate(tokenized_words):
        for j, word in enumerate(sentence_words):
            if re.match(pattern, word):
                tokenized_words[i][j] = '<URL>'
    return Rem_Punct(tokenized_words)

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

def Rem_Punct(tokenized_words):
    filtered_words = []
    for sentence_words in tokenized_words:
        list = []
        for word in sentence_words:
            if len(word) == 1:
                if re.match(r'^[^\w(){}\[\]]', word):
                    continue
                else:
                    list.append(word)
            else:
                list.append(word)
        if list:
            filtered_words.append(list)
    return filtered_words

def words_tokenize(sentences):
    tokenized_words = []
    for sentence in sentences:
        sent = sentence
        words = re.findall(r"'s|@[^\s{}()\[\]]*\w|#[^\s{}()\[\]]*\w|\w[^'\s{}()\[\]]*\w|\w+|\S",sent)
        tokenized_words.append(words)
    return Identify_Num(tokenized_words)

def sentence_tokenize(doc):
    pattern = re.compile(r'(?<![A-Z]\.|[0-9]\.)(?<!Mr\.|Dr\.)(?<!Mrs\.)(?<!\.\.)(?<=\.|\?)\s' + '|' + r'(?<=\.\"|\?\")\s')
    ignore = r'\n'
    snts = pattern.split(doc)
    sentences = []
    for snt in snts:
        sentence = re.sub(ignore,' ',snt)
        if sentence:
            sentences.append(sentence)
    return sentences

def tokenizer(doc):
    return sentence_tokenize(doc)

def tokenize_file(input_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()
    tokenized_text = tokenizer(text)
    return tokenized_text

##################################################################################################################

def generate_ngrams (tokenized_text,n):
    ngrams = {}
    n_1grams = {}
    for sentence_token in tokenized_text:
        sentence_token = ['<s>']*(n-1) + sentence_token + ['</s>']
        for i in range (len(sentence_token) - n + 1):
            ng = tuple(sentence_token[i:i+n])
            if ng in ngrams:
                ngrams[ng] += 1
            else:
                ngrams[ng] = 1

        for i in range (len(sentence_token) - n + 2):
            ng = tuple(sentence_token[i:i+n-1])
            if ng in n_1grams:
                n_1grams[ng] += 1
            else:
                n_1grams[ng] = 1

    return ngrams,n_1grams

##################################################################################################################

# NORMAL PROBABLITIES(CONDITIONAL FOR N = 3) STORED FOR UNIGRAMS,BIGRAMS AND TRIGRAMS
def pnormal_trigram(trigrams,bigrams):
    P = {}
    for trigram in trigrams:
        P[trigram] = trigrams[trigram]/bigrams[trigram[:2]]
    return P

def pnormal_bigrams(bigrams,unigrams):
    P = {}
    for bigram in bigrams:
        P[bigram] = bigrams[bigram]/unigrams[unigrams[:1]]
    return P

def pnormal_unigrams(unigrams,N):
    P = {}
    for unigram in unigrams:
        P[unigram] = unigrams[unigram]/N
    return P

##################################################################################################################

def Good_Turing (ngrams):
    freq = {}
    for tuples in ngrams:
        if (ngrams[tuples]) in freq:
            freq[ngrams[tuples]] += 1
        else :
            freq[ngrams[tuples]] = 1
    sorted_freq = dict(sorted(freq.items()))

    q = []
    t = []
    q.append(0)
    N = 0
    for frequencies in sorted_freq:
        q.append(frequencies)
        t.append(frequencies)
        N += frequencies*sorted_freq[frequencies]
    del q[-1]
    q[-1] *= 2
    del t[0]
    t.append(2*t[-1])

    r = []
    Zr = []
    i = 0
    for frequencies in sorted_freq:
        Zr.append((2*sorted_freq[frequencies])/(t[i]-q[i]))
        r.append(frequencies)
        i += 1

    slope, intercept, r_value, p_value, std_err = linregress(np.log(r),np.log(Zr))

    SNr = []
    for i in range (1,r[-1]+2):
        SNr.append(np.exp(intercept + (slope * np.log(i))))

    R = []
    for i in range(1,r[-1]+1):
        val = (i+1)*(SNr[i]/SNr[i-1])
        R.append(val)

    return R,N

##################################################################################################################

def probablity_GoodTuring(ngrams,n_1grams,R,N):
    sentence = input("input sentence: ")
    sent = []
    sent.append(sentence)
    tokens = words_tokenize(sent)
    tokens = ['<s>']*(2) + tokens[0] + ['</s>']
    P = 0
    for i in range (2,len(tokens)):
        trigram = tuple(tokens[i-2:i+1])
        bigram = tuple(tokens[i-2:i])
        if bigram in n_1grams:
            if trigram in ngrams:
                P += math.log(R[ngrams[trigram]-1]/n_1grams[bigram])
            else:
                P += math.log(1/n_1grams[bigram])
        else:
            P += math.log(1/N)
    P = math.exp(P)
    print(f"Score: {P}")

##################################################################################################################
    
def perplexity_GoodTuring (ngrams,n_1grams,R,N,sentence):
    sent = []
    sent.append(sentence)
    tokens = words_tokenize(sent)
    if tokens:
        tokens = ['<s>']*(2) + tokens[0] + ['</s>']
        P = 0
        for i in range (2,len(tokens)):
            trigram = tuple(tokens[i-2:i+1])
            bigram = tuple(tokens[i-2:i])
            if bigram in n_1grams:
                if trigram in ngrams:
                    P += math.log(R[ngrams[trigram]-1]/n_1grams[bigram])
                else:
                    P += math.log(1/n_1grams[bigram])
            else:
                P += math.log(1/N)

        avg_P = P/len(tokens)

        if P != 0:
            perplex = math.exp(-1*avg_P)
        else :
            perplex = 15000
        return perplex
    return None

##################################################################################################################

def LinearInterpolation (ngrams,n_1grams,unigrams):
    N = sum(unigrams.values())
    l1 = 0
    l2 = 0
    l3 = 0
    for trigram in ngrams:
        v1 = 0
        v2 = 0
        v3 = 0
        if n_1grams[trigram[:2]] > 1:
            v1 = (ngrams[trigram]-1)/(n_1grams[trigram[:2]]-1)
            max = v1
            add = 3

        if unigrams[trigram[1:2]] > 1:
            v2 = (n_1grams[trigram[-2:]]-1)/(unigrams[trigram[1:2]]-1)
            if v2 > max:
                add = 2
                max = v2
        
        if N > 1:
            v3 = (unigrams[trigram[-1:]]-1)/(N-1)
            if v3 > max:
                add = 1
                max = v3

        if add == 1:
            l1 += ngrams[trigram]
        elif add == 2:
            l2 += ngrams[trigram]
        else:
            l3 += ngrams[trigram]
    
    s = l1 + l2 + l3
    l1 = l1/s
    l2 = l2/s
    l3 = l3/s

    return l1,l2,l3

##################################################################################################################

def probablity_Interpolation(ngrams,n_1grams,unigrams,l1,l2,l3):
    N = sum(unigrams.values())
    sentence = input("input sentence: ")
    sent = []
    sent.append(sentence)
    tokens = words_tokenize(sent)
    tokens = ['<s>']*(2) + tokens[0] + ['</s>']

    P = 0
    for i in range (2,len(tokens)):
        trigram = tuple(tokens[i-2:i+1])
        val = 0
        if trigram[:2] in n_1grams and trigram in ngrams:
            val += l3*((ngrams[trigram])/(n_1grams[trigram[:2]]))
        if trigram[1:2] in unigrams and trigram[-2:] in n_1grams:
            val += l2*((n_1grams[trigram[-2:]])/(unigrams[trigram[1:2]]))
        if trigram[-1:] in unigrams:
            val += l1*(unigrams[trigram[-1:]]/N)
        if val != 0:
            P += math.log(val)
    P = math.exp(P)
    print(f"Score: {P}")

##################################################################################################################

def perplexity_Interpolation(ngrams,n_1grams,unigrams,l1,l2,l3,sentence):
    N = sum(unigrams.values())
    sent = []
    sent.append(sentence)
    tokens = words_tokenize(sent)
    if tokens:
        tokens = ['<s>']*(2) + tokens[0] + ['</s>']
        P = 0
        for i in range (2,len(tokens)):
            trigram = tuple(tokens[i-2:i+1])
            val = 0
            if trigram[:2] in n_1grams and trigram in ngrams:
                val += l3*((ngrams[trigram])/(n_1grams[trigram[:2]]))
            if trigram[1:2] in unigrams and trigram[-2:] in n_1grams:
                val += l2*((n_1grams[trigram[-2:]])/(unigrams[trigram[1:2]]))
            if trigram[-1:] in unigrams:
                val += l1*(unigrams[trigram[-1:]]/N)
            if val != 0:
                P += math.log(val)

        avg_P = P/len(tokens)

        if P != 0:
            perplex = math.exp(-1*avg_P)
        else :
            perplex = 15000
        return perplex
    return None

##################################################################################################################

def train_test (sentences):
    test = sentences[-1000:]
    sh = len(sentences)
    len_dev = int(sh*0.1)
    dev = sentences[:len_dev]
    train = sentences[(len_dev):-1000]
    length_train = len(train)/2
    length_test = len(test)*.5
    dev_tokens = words_tokenize(dev)

    ngrams,n_1grams = generate_ngrams(dev_tokens,3)
    bigrams,unigrams = generate_ngrams(dev_tokens,2)

    l1,l2,l3 = LinearInterpolation (ngrams,n_1grams,unigrams)

    train_tokens = words_tokenize(train)
    train_tri, train_bi = generate_ngrams(train_tokens,3)
    bg,train_uni = generate_ngrams(train_tokens,2)

    # LINEAR INTERPOLATION
    with open('2022114009_LM4_train-perplexity.txt', 'w') as file:
        sum_perplex = 0
        for sent in train:
            value = perplexity_Interpolation(train_tri,train_bi,train_uni,l1,l2,l3,sent)
            if value :
                val = f'{sent} {value}\n'
                sum_perplex += value
                file.write(val)
        avg_perplex = sum_perplex/length_train
        file.write(f'\nAverage Perpexitiy = {avg_perplex}')

    with open('2022114009_LM4_test-perplexity.txt', 'w') as file:
        sum_perplex = 0
        for sent in test:
            value = perplexity_Interpolation(train_tri,train_bi,train_uni,l1,l2,l3,sent)
            if value:
                val = f'{sent} {value}\n'
                sum_perplex += value
                file.write(val)
        avg_perplex = sum_perplex/length_test
        file.write(f'\nAverage Perpexitiy = {avg_perplex}')

    # GOOD TURING
    R,N = Good_Turing(train_tri)

    with open('2022114009_LM3_train-perplexity.txt', 'w') as file:
        sum_perplex = 0
        for sent in train:
            value = perplexity_GoodTuring(train_tri,train_bi,R,N,sent)
            if value :
                val = f'{sent} {value}\n'
                sum_perplex += value
                file.write(val)
        avg_perplex = sum_perplex/length_train
        file.write(f'\nAverage Perpexitiy = {avg_perplex}')

    with open('2022114009_LM3_test-perplexity.txt', 'w') as file:
        sum_perplex = 0
        for sent in test:
            value = perplexity_GoodTuring(train_tri,train_bi,R,N,sent)
            if value:
                val = f'{sent} {value}\n'
                sum_perplex += value
                file.write(val)
        avg_perplex = sum_perplex/len(test)
        file.write(f'\nAverage Perpexitiy = {avg_perplex}')
    
##################################################################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('lm_type')
    parser.add_argument('input_file')
    args = parser.parse_args()
    lm_type = args.lm_type
    input_file = args.input_file

    try:
        sentences = tokenize_file(input_file)
        tokenized_text = words_tokenize(sentences)
        ngrams,n_1grams = generate_ngrams(tokenized_text,3)
        bigrams,unigrams = generate_ngrams(tokenized_text,2)
        # train_test(sentences)
        if lm_type == 'g':
            R,N = Good_Turing(ngrams)
            probablity_GoodTuring(ngrams,n_1grams,R,N)
        elif lm_type == 'i':
            l1,l2,l3 = LinearInterpolation(ngrams,n_1grams,unigrams)
            probablity_Interpolation(ngrams,n_1grams,unigrams,l1,l2,l3)
        else:
            print("Error: Invalid lm_type")

    except FileNotFoundError:
        print("Error: The input file is not found.")
    except Exception as e:
        print(f"An unexpected error occured.{e}")

if __name__ == "__main__":
    main()
