import tensorflow as tf
import numpy as np
import os, csv, time, json, pickle, random
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.pipeline import Pipeline
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer

'''
This file contains utility functions for training, saving and loading models, 
as well as custom metrics / callbacks and other stuff. 

Sources:
https://www.kaggle.com/wowfattie/3rd-place
https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings
https://www.kaggle.com/theoviel/improve-your-score-with-text-preprocessing-v2
https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80568
'''

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", 
                       "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", 
                       "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", 
                       "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", 
                       "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", 
                       "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", 
                       "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", 
                       "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                       "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", 
                       "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 
                       "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", 
                       "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", 
                       "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", 
                       "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", 
                       "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", 
                       "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", 
                       "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", 
                       "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", 
                       "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", 
                       "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", 
                       "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&' # for adding space before and after in words containing punct

punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", 
                 "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', 
                 "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', 
                 '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', } # for glove

mispell_mapping = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 
                'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 
                'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 
                'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 
                'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', 
                "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', 
                '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 
                'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization', 
                'pokémon': 'pokemon'}

def clean_contractions(text, mapping=contraction_mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text_new = []
    for t in text.split(' '):
        if t.lower() in mapping:
            t_new = mapping[t.lower()]
            if t.isupper():
                t_new = t_new.upper()
            elif t[0].isupper() and len(t_new) > 1:
                t_new = t_new[0].upper() + t_new[1:]
            text_new.append(t_new)
        else:
            text_new.append(t)
    return ' '.join(text_new)

def clean_special_chars(text, punct=punct, mapping=punct_mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    for p in punct:
        text = text.replace(p, f' {p} ')
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters
    for s in specials:
        text = text.replace(s, specials[s])
    return text

def correct_spelling(text, mapping=mispell_mapping):
    for word in mapping.keys():
        text = text.replace(word, mapping[word])
    return text

class F1(tf.keras.metrics.Metric):
    def __init__(self, name='f1', **kwargs):
        super(F1, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight=sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight=sample_weight)
    
    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        pr = tf.math.multiply(p, r)
        pr = tf.math.scalar_mul(2, pr)
        pr_sum = tf.math.add(p, r)
        result = tf.math.divide_no_nan(pr, pr_sum)
        return result


def index_word_embeddings(embeddings_path, embedding_dim, skip_first_line=False):
    print('Indexing word embeddings from', str(embeddings_path), '...')
    embeddings_index = {}
    skipped = False
    with open(embeddings_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            if skip_first_line and not skipped:
                skipped = True
                continue
            splitted = line.split(' ')
            phrase_len = len(splitted)-embedding_dim
            coefs = ' '.join(splitted[phrase_len:len(splitted)])
            phrase_parts = splitted[0:phrase_len]
            if set(phrase_parts) == set(['']):
                phrase_parts.append('')
            word = ' '.join(phrase_parts)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            assert coefs.size == embedding_dim
            embeddings_index[word] = coefs
    print('Found', len(embeddings_index), 'word vectors.')
    return embeddings_index

def index_keyed_embeddings(keyed_vectors):
    print('Indexing embeddings from keyed vectors...')
    embedding_index = {}
    for word, vector in zip(keyed_vectors.index2word, keyed_vectors.vectors):
        embedding_index[word] = vector
    return embedding_index

def prepare_embedding_matrix(max_words, embedding_dim, word_index, embeddings_index, hparams, lower_only=False):
    print('Preparing embedding matrix...')
    np.random.seed(hparams['random_state'])
    porter = PorterStemmer()
    lancaster = LancasterStemmer()
    snowball = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()
    count = 0
    embedding_matrix = np.zeros((max_words, embedding_dim))
    random_vector = np.random.random(embedding_dim)
    for word, i in word_index.items():
        if i >= max_words:
            continue
        
        if word in embeddings_index and word.lower() not in embeddings_index:
            embeddings_index[word.lower()] = embeddings_index[word]

        embedding_vector = embeddings_index.get(word.lower()) if lower_only else embeddings_index.get(word)

        # https://www.kaggle.com/wowfattie/3rd-place
        if embedding_vector is None:
            embedding_vector = embeddings_index.get(word.lower())

        if embedding_vector is None:
            embedding_vector = embeddings_index.get(word.upper())

        if embedding_vector is None:
            embedding_vector = embeddings_index.get(word.capitalize())

        if embedding_vector is None:
            embedding_vector = embeddings_index.get(porter.stem(word)) 

        if embedding_vector is None:
            embedding_vector = embeddings_index.get(lancaster.stem(word)) 

        if embedding_vector is None:
            embedding_vector = embeddings_index.get(snowball.stem(word)) 

        if embedding_vector is None:
            embedding_vector = embeddings_index.get(lemmatizer.lemmatize(word))
         

        if word == hparams['tokenizer_oov_token'] or embedding_vector is None:
            embedding_matrix[i] = random_vector
        else:    
            embedding_matrix[i] = embedding_vector
            count += 1
        
    print('Word vectors coverage:', count / max_words)
    print('Embedding matrix shape:', embedding_matrix.shape)
    return embedding_matrix


def get_best_threshold(y_true, predictions):
    print('Finding best threshold...')
    max_f1, best_thresh = 0, 0
    for thresh in np.arange(0.1, 0.501, 0.01):
        f1 = f1_score(y_true, predictions > thresh)
        if f1 > max_f1:
            max_f1 = f1
            best_thresh = thresh
    print('Best F1 is {:.5f} for thresh {:.4f}'.format(max_f1, best_thresh))
    return max_f1, best_thresh

class ROCCallback(tf.keras.callbacks.Callback):
    def __init__(self, training_data, validation_data):
        x = training_data[0]
        y = training_data[1]
        x_val = validation_data[0]
        y_val = validation_data[1]

        self.x_all = np.concatenate((x, x_val))
        self.y_all = np.concatenate((y, y_val))
        self.val_set_start_index = x.shape[0]
        #assert self.x_all.shape[0] == self.val_set_start_index + len(x_val) 

        self.val_roc_aucs = []

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_all = self.model.predict(self.x_all, batch_size=2048) # prediction probablities
        
        y_pred_train = y_pred_all[0:self.val_set_start_index]
        y_true_train = self.y_all[0:self.val_set_start_index]

        y_pred_val = y_pred_all[self.val_set_start_index:]
        y_true_val = self.y_all[self.val_set_start_index:]

        roc_train = roc_auc_score(y_true_train, y_pred_train, average='macro')
        roc_val = roc_auc_score(y_true_val, y_pred_val, average='macro')

        self.val_roc_aucs.append(roc_val)
        print('roc-auc: {:.4f} - roc-auc_val: {:.4f}'.format(roc_train, roc_val))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

class CustomAveraging(tf.keras.callbacks.Callback):

    def __init__(self, save_epoch):
        self.save_epoch = save_epoch - 1
        self.saved_weights = None

    def on_epoch_end(self, epoch, logs={}):
        if epoch == self.save_epoch:
            self.saved_weights = self.model.get_weights()
            print('Model weights saved.')

        if epoch == (self.save_epoch + 1):
            model_weights = self.model.get_weights()
            new_weights = [(model_weights[i] + self.saved_weights[i]) / 2 for i in range(len(model_weights))]
            self.model.set_weights(new_weights)
            print('Model weights averaged.')



def train_model_cv(data, hparams, build_model_fn, tokenizer, models_data_path, embedding_matrix=None, callbacks_=[], random_state=42, use_tensorboard=False, model_to_json=True):

    X = data['X']
    y = data['y']
    
    cv_indices = data['cv_indices']
    
    fold = 1

    min_val_losses = []
    
    train_losses_of_min_val_losses = []
    epochs_of_min_val_losses = []

    val_thresh_f1s = []
    val_best_thresholds = []

    experiment_id = str(time.time())
    experiment_path = models_data_path + 'experiments/' + experiment_id + '/'
    tensorboard_path =  models_data_path + 'tb_logs/' + experiment_id + '/'

    print()
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
        print(experiment_path, 'created')

    if use_tensorboard and not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
        print(tensorboard_path, 'created')
    
    for train_indices, val_indices in cv_indices:
        print('\n\nFold', fold)
        
        X_train = X[train_indices]
        y_train = y[train_indices]
        
        X_val = X[val_indices]
        y_val = y[val_indices]

        model = build_model_fn(hparams, embedding_matrix)
        
        if use_tensorboard:
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path + "Fold_" + str(fold) + "/")
            if fold > 1:
                callbacks_[-1] = tensorboard_callback
            else:
                callbacks_.append(tensorboard_callback)
        
        history = model.fit(X_train, y_train, 
                            epochs=hparams['epochs'], batch_size=hparams['batch_size'], 
                            validation_data=(X_val, y_val), 
                            callbacks=callbacks_, shuffle=True)

        max_f1, best_thresh = get_best_threshold(y_val, model.predict(X_val, batch_size=2048))
        val_thresh_f1s.append(max_f1)
        val_best_thresholds.append(best_thresh)

        history = history.history
            
        val_losses = np.array(history['val_loss'])
        val_losses_argmin = np.argmin(val_losses)
        min_val_losses.append(val_losses[val_losses_argmin])
        epochs_of_min_val_losses.append(val_losses_argmin+1)
        train_losses_of_min_val_losses.append(history['loss'][val_losses_argmin])

        fold += 1

    min_val_losses = np.array(min_val_losses)
    epochs_of_min_val_losses = np.array(epochs_of_min_val_losses)
    train_losses_of_min_val_losses = np.array(train_losses_of_min_val_losses)

    avg_min_val_loss = min_val_losses.mean()
    stddev_min_val_loss = min_val_losses.std()

    avg_loss_epoch = epochs_of_min_val_losses.mean()
    stddev_loss_epoch = epochs_of_min_val_losses.std()

    avg_max_val_thresh_f1 = np.array(val_thresh_f1s).mean()
    std_max_val_thresh_f1 = np.array(val_thresh_f1s).std()

    avg_val_best_thresh = np.array(val_best_thresholds).mean()
    std_val_best_thresh = np.array(val_best_thresholds).std()
    hparams['best_thresh'] = round(avg_val_best_thresh, 4)
    
    # average epoch is rounded to int and replaced in hparams
    avg_loss_epoch_rounded = int(round(avg_loss_epoch, 0))
    if hparams['avg_epochs_metric'] == 'val_loss':
        hparams['epochs'] = avg_loss_epoch_rounded

    avg_train_loss = train_losses_of_min_val_losses.mean()
    stddev_train_loss = train_losses_of_min_val_losses.std()

    text = []

    text.append('Average max val F1 is {:.5f} ± {:.5f} for thresh {:.4f} ± {:.4f}\n'. format(avg_max_val_thresh_f1, std_max_val_thresh_f1, avg_val_best_thresh, std_val_best_thresh))
    text.append('Average min val loss is {:.5f} with std of {:.5f}\n'.format(avg_min_val_loss, stddev_min_val_loss))
    text.append('Average train loss of min val loss is {:.5f} with std of {:.5f}\n'.format(avg_train_loss, stddev_train_loss))
    text.append('Average epoch of min val loss is {:.2f} with std of {:.2f} - rounded to {}\n'.format(avg_loss_epoch, stddev_loss_epoch, avg_loss_epoch_rounded))
    
    print()
    for t in text:
        print(t)

    # saving results to txt file
    experiment_file_name = experiment_path + 'results.txt'
    with open(experiment_file_name, 'w') as f:
        f.writelines(text)
    print('Results saved to', experiment_file_name)

    # saving model architecture to json file
    if model_to_json:
        model_json = model.to_json()
        model_json_file_name = experiment_path + 'model.json'
        with open(model_json_file_name, 'w') as json_file:
            json_file.write(model_json)
        print('Model architecture saved to', model_json_file_name)

    # saving hyperparameters
    if 'optimizer' in hparams.keys():
        optimizer = hparams['optimizer']
        if not isinstance(optimizer, str):
            hparams['optimizer'] = tf.keras.utils.serialize_keras_object(optimizer)
            for k, v in hparams['optimizer']['config'].items():
                if isinstance(v, np.float32):
                    hparams['optimizer']['config'][k] = float(v)
        hparams_json = json.dumps(hparams)
        hparams_json_file_name = experiment_path + 'hparams.json'
        with open(hparams_json_file_name, 'w') as json_file:
            json_file.write(hparams_json)
        print('Model hparams saved to', hparams_json_file_name)

    # saving model summary
    summary_file_name = experiment_path + 'model_summary.txt'
    def save_summary(summary_line):
        summary_line += '\n'
        with open(summary_file_name, 'a') as f:
            f.write(summary_line)
    model.summary(print_fn=save_summary)
    print('Model summary saved to', summary_file_name)

    # saving/copying build_model.py file
    build_model_file_name = 'build_model.py'
    if os.path.isfile(build_model_file_name):
        with open(build_model_file_name, 'r') as f:
            file_lines = f.readlines()
        build_model_file_path = experiment_path + build_model_file_name
        with open(build_model_file_path, 'w') as f:
            f.writelines(file_lines)
            print(build_model_file_name, 'file saved to', build_model_file_path)

    # saving experiment id, and average validation metrics to csv file
    results_csv_file_name = 'tcc_val_results.txt'
    with open(results_csv_file_name, 'a', newline='') as f:
        field_names = ['experiment_id',
                       'avg_max_val_thresh_f1', 
                       'avg_min_val_loss'
                      ]
        writer = csv.DictWriter(f, fieldnames=field_names, delimiter=';')
        writer.writerow({'experiment_id': str(experiment_id), 
                         'avg_max_val_thresh_f1': str(avg_max_val_thresh_f1), 
                         'avg_min_val_loss': str(avg_min_val_loss)})
        print('Validation metrics appended to', results_csv_file_name)

    return([experiment_id, 
            avg_max_val_thresh_f1, 
            avg_min_val_loss])

def load_hparams_and_model(experiment_id, models_data_path):
    print('Loading model hyperparameters from experiment', experiment_id, '...')
    experiment_path = models_data_path + 'experiments/' + experiment_id + '/'
    
    hparams_file_path = experiment_path + 'hparams.json'
    with open(hparams_file_path, 'r') as f:
        data = f.read()
        hparams = json.loads(data)
        
    model_file_path = experiment_path + 'model.json'
    with open(model_file_path, 'r') as f:
        model_json = f.read()
    model = tf.keras.models.model_from_json(model_json)
        
    return hparams, model

def train_model_from_experiment(data, hparams, model, callbacks_=[], random_state=42):
    X = data['X']
    y = data['y']
    
    print('\nContinuing training with following hparams:')
    print(hparams)
    
    # optimizer setup
    optimizer = hparams['optimizer']
    optimizer = tf.keras.optimizers.get(optimizer)
    
    print('Compiling model...')
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    print('Fitting model...')
    model.fit(X, y, epochs=hparams['epochs'], batch_size=hparams['batch_size'], callbacks=callbacks_, shuffle=True)
    
    return model