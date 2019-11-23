import tensorflow as tf
from tensorflow import keras
from utils import F1
import numpy as np
import random

'''
This script contains a function for building and compiling a model.
'''
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # values shape == (batch_size, max_length, hidden size)

        # hidden (query?) shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score (with broadcasting)
        hidden_with_time_axis = tf.expand_dims(query, 1)
       
        W1_out = self.W1(values) # (batch_size, max_length, units)
        W2_out = self.W2(hidden_with_time_axis) # (batch_size, 1, units)
        
        # the shape of the tensor (W_out_sum) before applying self.V is (batch_size, max_length, units)
        W_out_sum = W1_out + W2_out # (..., 1, units) broadcasted to (..., max_length, units)
        
        W_out_sum_tanh = tf.nn.tanh(W_out_sum)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        score = self.V(W_out_sum_tanh)

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1) # axis 1 je duljina sekvence - trebamo attn težinu za svaku riječ

        context_vector = attention_weights * values # (batch_size, max_length, hidden_size)
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

def build_model(hparams, embedding_matrix=None):
    print('Building model...')
    np.random.seed(hparams['random_state'])
    random.seed(hparams['random_state'])
    tf.random.set_seed(hparams['random_state'])


    input_ = keras.layers.Input(shape=(hparams['max_length'],))

    if embedding_matrix is not None:
        x = keras.layers.Embedding(input_dim=hparams['max_words'], output_dim=hparams['emb_out_dim'], 
                                   embeddings_initializer=keras.initializers.Constant(embedding_matrix), trainable=False)(input_)
    else:
        x = keras.layers.Embedding(input_dim=hparams['max_words'], output_dim=hparams['emb_out_dim'])(input_)

    x = keras.layers.SpatialDropout1D(rate=hparams['dropout_rate'])(x)

    x, x_h_state_1_1, _, x_h_state_1_2, _ = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, return_state=True))(x) # x.shape = None, 256, len
    avg_1 = keras.layers.GlobalAveragePooling1D()(x)
    max_1 = keras.layers.GlobalMaxPooling1D()(x)
    x_ctx_1_1, _ = BahdanauAttention(256)(x_h_state_1_1, x[:,:,0:256])
    x_ctx_1_2, _ = BahdanauAttention(256)(x_h_state_1_2, x[:,:,256:])

    x_seq_2_1, x_state_2_1 = keras.layers.GRU(128, return_sequences=True, return_state=True)(x)
    avg_2_1 = keras.layers.GlobalAveragePooling1D()(x_seq_2_1)
    max_2_1 = keras.layers.GlobalMaxPooling1D()(x_seq_2_1)
    x_ctx_2_1, _ = BahdanauAttention(128)(x_state_2_1, x_seq_2_1)
    
    x_seq_2_2, x_state_2_2 = keras.layers.GRU(128, return_sequences=True, return_state=True, go_backwards=True)(x)
    avg_2_2 = keras.layers.GlobalAveragePooling1D()(x_seq_2_2)
    max_2_2 = keras.layers.GlobalMaxPooling1D()(x_seq_2_2)
    x_ctx_2_2, _ = BahdanauAttention(128)(x_state_2_2, x_seq_2_2)
    
    c_out = keras.layers.concatenate([avg_1, 
                                      max_1,
                                      x_ctx_1_1,
                                      x_ctx_1_2,

                                      avg_2_1,
                                      max_2_1,
                                      x_ctx_2_1,
                                      
                                      avg_2_2,
                                      max_2_2,
                                      x_ctx_2_2,
                                    ])

    output = keras.layers.Dense(1, activation='sigmoid')(c_out)

    model = keras.Model(inputs=[input_], outputs=[output])

    model.compile(optimizer=hparams['optimizer'], loss='binary_crossentropy', metrics=[])
    print('Parameters:', model.count_params())
#    print(model.summary())
    return model

#build_model({'max_length':28, 'max_words': 200000, 'emb_out_dim':600, 'n_classes':2, 'optimizer':'nadam', 
#             'random_state': 42, 'select_top_k': 20000, 'activation': 'relu', 'dropout_rate': 0.2, 'batch_size':512}, None)