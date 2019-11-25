import os, json, time, pickle, time, gc
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from build_model import build_model
from utils import train_model_cv, index_word_embeddings, prepare_embedding_matrix 
from utils import clean_contractions, clean_special_chars, correct_spelling, CustomAveraging, index_keyed_embeddings
from pathlib import Path
from gensim import models

'''
This script runs a single experiment with the current model from build_model.py file by 
using cross validation. The number of CV folds can be set in N_FOLDS variable.
Results of the experiment are saved in experiments/<experiment_id> directory, where <experiment_id> 
is unique id created during experiment. Some metric reults are also saved to tcc_val_results.txt file in csv format. 
'''

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


N_FOLDS = 5
RANDOM_STATE = 42
USE_TENSORBOARD = False
MODELS_DATA_PATH = 'models/'

TRAIN_DATA_PATH = 'train.csv'
TEST_DATA_PATH = 'test.csv'

GLOVE_PATH = Path('./embeddings/glove.840B.300d/glove.840B.300d.txt')
PARAGRAM_PATH = Path('./embeddings/paragram_300_sl999/paragram_300_sl999.txt')
WIKI_NEWS_PATH = Path('./embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec')
GOOGLE_NEWS_PATH = Path('./embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin')

LABEL_COLUMNS = ["target"]

def load_train_data(file_path, train_column, label_columns):
    print('Loading train data...')
    data_frame = pd.read_csv(file_path)
    return data_frame[train_column].tolist(), data_frame[label_columns].values

def load_test_data(file_path, train_column):
    print('Loading test data...')
    data_frame = pd.read_csv(file_path)
    return data_frame[train_column].tolist(), data_frame['qid']

# baseline model -> all zeros
def evaluate_baselines(y_true, average='binary'):
    y_pred = np.zeros_like(y_true)
    f1_baseline = f1_score(y_true, y_pred, average=average)
    print('All-zeros', average, 'F1 baseline score:', f1_baseline)
    print('Accuracy baseline score:', accuracy_score(y_true, y_pred))

# can be used for repeated cross validation, n_repeats=1 by default
def split_dataset_to_train_val_folds(X_data, y_data, n_folds=N_FOLDS, n_repeats=1, random_state=RANDOM_STATE):
    rmskf = StratifiedKFold(n_splits=n_folds, random_state=random_state)
    train_val_indices = []
    for train_indices, val_indices in rmskf.split(X_data, y_data):
        train_val_indices.append((train_indices, val_indices))
    return train_val_indices

# creates Tokenizer instance and fits it
def fit_tokenizer(X_texts, hparams):
    print('Fitting tokenizer...')
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=hparams['max_words'], filters=hparams['tokenizer_filters'], 
                            lower=hparams['tokenizer_lower'], split=hparams['tokenizer_split'], 
                            char_level=hparams['tokenizer_char_level'], oov_token=hparams['tokenizer_oov_token'])
    tokenizer.fit_on_texts(X_texts)
    return tokenizer

def create_padded_sequences(X_texts, tokenizer, hparams):
    print('Converting texts to sequences...')
    X_sequences = tokenizer.texts_to_sequences(X_texts)
    print('Padding sequences...')
    X_padded = tf.keras.preprocessing.sequence.pad_sequences(X_sequences, maxlen=hparams['max_length'], padding=hparams['padding'], truncating=hparams['truncating'])
    return X_padded

# main
if __name__ == '__main__':
    X_train_texts, y_train = load_train_data(file_path=TRAIN_DATA_PATH, 
                                                train_column='question_text', 
                                                label_columns=LABEL_COLUMNS)

    X_test_texts, _ = load_test_data(TEST_DATA_PATH, train_column='question_text') # test data will be used only for tokenizer fitting

    # main hyperparameters can be set here
    hparams = {
        'max_words': None, # for Tokenizer
        'max_length': 48,
        'batch_size': 512,

        'emb_out_dim': 600, 
        
        'dropout_rate': 0.4,
        'activation': 'relu',
        
        'epochs': 4, # leave 100 or more if using early stopping - true by default
        
        'optimizer': 'nadam',
        'tokenizer_filters': '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        'tokenizer_lower': False,
        'tokenizer_split': " ",
        'tokenizer_char_level': False,
        'padding': 'post',
        'truncating': 'post',

        'tokenizer_oov_token': '<UNK>', # not a real hyperparameter
        'n_classes': y_train.shape[1], # not a real hyperparameter
        'random_state': RANDOM_STATE, # not a real hyperparameter
        'avg_epochs_metric': 'val_loss', # not a real hyperparameter
    }
    
    print('Cleaning contractions...')
    X_train_texts = [clean_contractions(text) for text in X_train_texts]
    X_test_texts = [clean_contractions(text) for text in X_test_texts]

    print('Cleaning special characters...')
    X_train_texts = [clean_special_chars(text) for text in X_train_texts]
    X_test_texts = [clean_special_chars(text) for text in X_test_texts]

    print('Correcting spelling...')
    X_train_texts = [correct_spelling(text) for text in X_train_texts]
    X_test_texts = [correct_spelling(text) for text in X_test_texts]

    print('Evaluating all-zeros baseline...')
    evaluate_baselines(y_train)
        
    glove_index_filename = 'glove_index.p'
    paragram_index_filename = 'paragram_index.p'
    wiki_news_index_filename = 'wikinews_index.p'
    google_news_index_filename = 'googlenews_index.p'
    
    print('Loading embeddings index...')
    if not os.path.isfile(glove_index_filename):
        embeddings_index_glove = index_word_embeddings(GLOVE_PATH, 300)
        with open(glove_index_filename, 'wb') as f:
            pickle.dump(embeddings_index_glove, f)
            print('Embedding index cached to', glove_index_filename)
    else:
        with open(glove_index_filename, 'rb') as f:
            embeddings_index_glove = pickle.load(f)
            print('GloVe embeddings loaded from', glove_index_filename)

    if not os.path.isfile(paragram_index_filename):
        embeddings_index_paragram = index_word_embeddings(PARAGRAM_PATH, 300)
        with open(paragram_index_filename, 'wb') as f:
            pickle.dump(embeddings_index_paragram, f)
            print('Embedding index cached to', paragram_index_filename)
    else:
        with open(paragram_index_filename, 'rb') as f:
            embeddings_index_paragram = pickle.load(f)
            print('Paragram embeddings loaded from', paragram_index_filename)

    tokenizer = fit_tokenizer(X_train_texts + X_test_texts, hparams)
    VOCAB_SIZE = len(tokenizer.word_index)
    if hparams['max_words'] is None:
        hparams['max_words'] = VOCAB_SIZE + 1 # 1 for padding value 0
    else:
        hparams['max_words'] += 1
    print('Found', VOCAB_SIZE, 'unique train tokens.')
    print('MAX WORDS:', hparams['max_words'])
    
    embedding_matrix_glove = prepare_embedding_matrix(hparams['max_words'], 300, 
                                                      tokenizer.word_index, embeddings_index_glove, hparams)

    embedding_matrix_paragram = prepare_embedding_matrix(hparams['max_words'], 300, 
                                                      tokenizer.word_index, embeddings_index_paragram, hparams, lower_only=True)
    
    embedding_matrix_1 = 0.7 * embedding_matrix_glove + 0.3 * embedding_matrix_paragram

    del embeddings_index_glove, embeddings_index_paragram, embedding_matrix_glove, embedding_matrix_paragram
    gc.collect()
    time.sleep(10)

    X_train_padded = create_padded_sequences(X_train_texts, tokenizer, hparams)

    print('Loading embeddings index...')
    if not os.path.isfile(wiki_news_index_filename):
        embeddings_index_wiki = index_word_embeddings(WIKI_NEWS_PATH, 300, True)
        with open(wiki_news_index_filename, 'wb') as f:
            pickle.dump(embeddings_index_wiki, f)
            print('Embedding index cached to', wiki_news_index_filename)
    else:
        with open(wiki_news_index_filename, 'rb') as f:
            embeddings_index_wiki = pickle.load(f)
            print('Wiki News embeddings loaded from', wiki_news_index_filename)

    if not os.path.isfile(google_news_index_filename):
        print('Loading keyed vectors...')
        keyed_vectors_google = models.KeyedVectors.load_word2vec_format(GOOGLE_NEWS_PATH, binary=True) #index_word_embeddings(GOOGLE_NEWS_PATH, 300, True)
        embeddings_index_google = index_keyed_embeddings(keyed_vectors_google)
        with open(google_news_index_filename, 'wb') as f:
            pickle.dump(embeddings_index_google, f)
            print('Embedding index cached to', google_news_index_filename)
    else:
        with open(google_news_index_filename, 'rb') as f:
            embeddings_index_google = pickle.load(f)
            print('Google News embeddings loaded from', google_news_index_filename)

    embedding_matrix_wiki = prepare_embedding_matrix(hparams['max_words'], 300, 
                                                      tokenizer.word_index, embeddings_index_wiki, hparams)

    embedding_matrix_google = prepare_embedding_matrix(hparams['max_words'], 300, 
                                                      tokenizer.word_index, embeddings_index_google, hparams)
    
    embedding_matrix_2 = 0.7 * embedding_matrix_wiki + 0.3 * embedding_matrix_google

    embedding_matrix_final = np.concatenate((embedding_matrix_1, embedding_matrix_2), axis=1)

    del embeddings_index_wiki, embeddings_index_google, embedding_matrix_wiki, embedding_matrix_google
    gc.collect()
    time.sleep(10)

    print('Splitting train set to', N_FOLDS, 'cross validation folds...')
    train_val_indices = split_dataset_to_train_val_folds(X_train_padded, y_train)

    data = {'X': X_train_padded, 'y': y_train, 'cv_indices': train_val_indices}

    start = time.time()
    results = train_model_cv(data, hparams, build_model, tokenizer, MODELS_DATA_PATH, embedding_matrix_final,
                             [tf.keras.callbacks.EarlyStopping(monitor=hparams['avg_epochs_metric'], mode='auto', patience=1, verbose=1, restore_best_weights=True),
                              CustomAveraging(3)
                             ], 
                             RANDOM_STATE, USE_TENSORBOARD, model_to_json=False)
    end = time.time()
    diff = end - start
    hours = diff / 3600
    mins = (hours - int(hours)) * 60
    print('\nExperiment', results[0], 'finished in {:.0f} h {:.0f} mins.'.format(hours, mins))

