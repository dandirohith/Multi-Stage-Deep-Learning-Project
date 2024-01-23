# %%
import os, sys

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 64
EPOCHS = 20
LSTM_NODES =256
NUM_SENTENCES = 10000
MAX_SENTENCE_LENGTH = 50
MAX_NUM_WORDS = 10000
EMBEDDING_SIZE = 300

import re
import string
from pickle import dump
from unicodedata import normalize
from numpy import array
from numpy.random import shuffle
from pickle import load

# load doc into memory
def load_doc(filename):
  file = open(filename, mode='rt', encoding='utf-8')
# read all text
  text = file.read()
# close the file
  file.close()
  return text

def to_pairs(doc):
  lines = doc.strip().split('\n')
  pairs = [line.split('\t') for line in lines]
  return pairs

def clean_pairs(lines):
  cleaned = list()
  re_punc = re.compile('[%s]' % re.escape(string.punctuation))
  re_print = re.compile('[^%s]' % re.escape(string.printable))
  for pair in lines:
    clean_pair = list()
    for line in pair:
      # normalize unicode characters
      line = normalize('NFD', line).encode('ascii', 'ignore')
      line = line.decode('UTF-8')
      # tokenize on white space
      line = line.split()
      # convert to lowercase
      line = [word.lower() for word in line]
      # remove punctuation from each token
      line = [re_punc.sub('', w) for w in line]
      # remove non-printable chars form each token
      line = [re_print.sub('', w) for w in line]
      # remove tokens with numbers in them
      line = [word for word in line if word.isalpha()]
      # store as string
      clean_pair.append(' '.join(line))
    cleaned.append(clean_pair)
  return array(cleaned)
# save a list of clean sentences to file
def save_clean_data(sentences, filename):
  dump(sentences, open(filename, 'wb'))
  print('Saved: %s' % filename)
# load dataset
filename = 'deu.txt'
doc = load_doc(filename)
# split into english-german pairs
pairs = to_pairs(doc)
# clean sentences
clean_pairs = clean_pairs(pairs)
# save clean pairs to file
save_clean_data(clean_pairs, 'english-german.pkl')

raw_dataset = load(open('english-german.pkl', 'rb'))

n_sentences = 10000
dataset = raw_dataset[:n_sentences, :]
input_sentences = [item[1] for item in dataset]
output_sentences = [item[0] for item in dataset]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(input_sentences, output_sentences, test_size=0.2, random_state=42)

# Print 100th sentence in original script for source and target language
print("\n100th senetnce in Source Language : ",X_train[1]," , Target Language : ",y_train[1])


# Dataset : http://www.manythings.org/anki/  Download the file fra-eng.zip and extract it. You will then see the fra.txt file.
input_sentences = []          # list containing inputs of encoder
output_sentences = []         # list containing output of decoder
output_sentences_inputs = []  # list containing input of decoder

def preprocess(seq):          # Utility function to preprocess strings
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~''' # initializing punctuations string 
    # Removing punctuations in string
    for ele in seq: 
        if ele in punc: 
            seq = seq.replace(ele, "") 
    line = seq.split()
    line = [word.lower() for word in line]
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # remove non-printable chars form each token
    line = [re_print.sub('', w) for w in line]
    return " ".join(line)

for i in range(len(X_train)):
    input_sentence = preprocess(X_train[i])
    output = y_train[i]
    output_sentence = output + ' <eos>'
    output_sentence_input = '<sos> ' + output

    input_sentences.append(input_sentence)
    output_sentences.append(output_sentence)
    output_sentences_inputs.append(output_sentence_input)


print(input_sentences[1])
print(output_sentences[1])
print(output_sentences_inputs[1])

# %%
# TOKENIZATION
input_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
input_tokenizer.fit_on_texts(input_sentences)
input_integer_seq = input_tokenizer.texts_to_sequences(input_sentences)

word2idx_inputs = input_tokenizer.word_index
print('Total unique words in the input: %s' % len(word2idx_inputs))

max_input_len = max(len(sen) for sen in input_integer_seq)
print("Length of longest sentence in input: %g" % max_input_len)


output_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
output_tokenizer.fit_on_texts(output_sentences + output_sentences_inputs)
output_integer_seq = output_tokenizer.texts_to_sequences(output_sentences)
output_input_integer_seq = output_tokenizer.texts_to_sequences(output_sentences_inputs)

word2idx_outputs = output_tokenizer.word_index
print('Total unique words in the output: %s' % len(word2idx_outputs))

num_words_output = len(word2idx_outputs) + 1
max_out_len = max(len(sen) for sen in output_integer_seq)
print("Length of longest sentence in the output: %g" % max_out_len)



encoder_input_sequences = pad_sequences(input_integer_seq, maxlen=max_input_len)
print("encoder_input_sequences.shape:", encoder_input_sequences.shape)
print("encoder_input_sequences[1]:", encoder_input_sequences[1])
#print(word2idx_inputs["pack"])


decoder_input_sequences = pad_sequences(output_input_integer_seq, maxlen=max_out_len, padding='post')
print("decoder_input_sequences.shape:", decoder_input_sequences.shape)
print("decoder_input_sequences[1]:", decoder_input_sequences[1])



decoder_output_sequences = pad_sequences(output_integer_seq, maxlen=max_out_len, padding='post')

# %%
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Assuming you have NLTK installed, if not:
# pip install nltk

# Make sure to download the German language data for NLTK
import nltk
nltk.download('punkt')

all_german_sentences = [item[1] for item in raw_dataset]
# Tokenize the sentences into words
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in all_german_sentences]

# Set the size of the word vectors
vector_size = EMBEDDING_SIZE

# Create Word2Vec model
word2vec_model = Word2Vec(sentences=tokenized_sentences, vector_size=vector_size, window=5, min_count=1, workers=8)

# Training the Word2Vec model
word2vec_model.train(tokenized_sentences, total_examples=len(tokenized_sentences), epochs=20)

# Vocabulary size
num_words = min(MAX_NUM_WORDS,len(word2vec_model.wv) + 1)

# Create an embedding matrix
embedding_matrix = np.zeros((num_words, EMBEDDING_SIZE))
for word, i in input_tokenizer.word_index.items():
    if i < num_words:
        embedding_vector = word2vec_model.wv[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(num_words, EMBEDDING_SIZE, weights=[embedding_matrix], input_length=max_input_len)

# %%
decoder_targets_one_hot = np.zeros((
        len(input_sentences),
        max_out_len,
        num_words_output
    ),
    dtype='float32'
)
    
decoder_targets_one_hot.shape


for i, d in enumerate(decoder_output_sequences):
    for t, word in enumerate(d):
        decoder_targets_one_hot[i, t, word] = 1
        
        
# ENCODERS
encoder_inputs_placeholder = Input(shape=(max_input_len,))
x = embedding_layer(encoder_inputs_placeholder)
encoder = LSTM(LSTM_NODES, return_state=True)

encoder_outputs, h, c = encoder(x)
encoder_states = [h, c]



decoder_inputs_placeholder = Input(shape=(max_out_len,))

decoder_embedding = Embedding(num_words_output, LSTM_NODES)
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)

decoder_lstm = LSTM(LSTM_NODES, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs_x, initial_state=encoder_states)\




decoder_dense = Dense(num_words_output, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)


model = Model([encoder_inputs_placeholder,
  decoder_inputs_placeholder], decoder_outputs)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)



r = model.fit(
    [encoder_input_sequences, decoder_input_sequences],
    decoder_targets_one_hot,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
)


encoder_model = Model(encoder_inputs_placeholder, encoder_states)

decoder_state_input_h = Input(shape=(LSTM_NODES,))
decoder_state_input_c = Input(shape=(LSTM_NODES,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]


decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)


decoder_outputs, h, c = decoder_lstm(decoder_inputs_single_x, initial_state=decoder_states_inputs)

decoder_states = [h, c]
decoder_outputs = decoder_dense(decoder_outputs)



decoder_model = Model(
    [decoder_inputs_single] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)


idx2word_input = {v:k for k, v in word2idx_inputs.items()}
idx2word_target = {v:k for k, v in word2idx_outputs.items()}



def translate_sentence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2idx_outputs['<sos>']
    eos = word2idx_outputs['<eos>']
    output_sentence = []

    for _ in range(max_out_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        idx = np.argmax(output_tokens[0, 0, :])

        if eos == idx:
            break

        word = ''

        if idx > 0:
            word = idx2word_target[idx]
            output_sentence.append(word)

        target_seq[0, 0] = idx
        states_value = [h, c]

    return ' '.join(output_sentence)

from nltk.translate.bleu_score import corpus_bleu

# Example handling unknown words
UNK_IDX = word2idx_inputs.get('<UNK>', None)
if UNK_IDX is None:
    UNK_IDX = len(word2idx_inputs) + 1
    word2idx_inputs['<UNK>'] = UNK_IDX

def encode_to_input(s):
    x  = []
    for w in s.split():
        if w.lower() not in word2idx_inputs:
            print("{w} is not in the vocabulary.")
            x.append(word2idx_inputs['<UNK>'])
        else:
            x.append(word2idx_inputs[w.lower()])
    return pad_sequences([x], maxlen=max_input_len)

with open('output.txt', 'w') as file:
    def evaluate_model(X,Y):
        actual, predicted = list(), list()
        i = 0
        for x,y in zip(X,Y):
            encoded_x = encode_to_input(x)
            translated_x =translate_sentence(encoded_x)
            reference = [y.split()]
            candidate = translated_x.split()
            if i < 10:
                print('src=[%s], target=[%s], predicted=[%s]' % (x, str(reference), str(candidate)), file=file)
                i = i+1
            actual.append(reference)
            predicted.append(candidate)
        print(actual)
        print(predicted)
        print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)), file=file)
        print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)), file=file)
        print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)), file=file)
        print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)), file=file)

    print('train', file=file)
    evaluate_model(X_train, y_train)

    print('test', file=file)
    evaluate_model(X_test, y_test)
