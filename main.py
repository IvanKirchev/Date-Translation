import tensorflow as tf
from utils import load_dataset, preprocessing, softmax
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
import numpy as np

def one_step_attention(a, s_prev):
    '''
    a: (m, Tx, 2*n_a)
    s_prev: (m, n_s)
    '''

    s_prev = REPEATOR(s_prev) # s_prev shape: (m, Tx, n_s)
    concat = CONCATENATOR([a, s_prev]) # concat shape: (m, Tx, n_s + 2*n_a)

    e = DENSOR1(concat) # e shape: (m, Tx, 10)
    energies = DENSOR2(e) # energies shape: (m, Tx, 1)

    alphas = ACTIVATOR(energies) # alphas shape: (m, Tx, 1)
    context = DOTOR([alphas, a]) # context shape: (m, 1, 64)

    return context

def modelf(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """

    X = tf.keras.layers.Input(shape=(Tx, human_vocab_size), name='input_x')

    s0 = tf.keras.layers.Input(shape=(n_s, ), name='input_cell_state')
    c0 = tf.keras.layers.Input(shape=(n_s, ), name='input_cell_context')

    # post attention hidden state
    s = s0
    # post attention hidden context
    c = c0

    a = Bidirectional(LSTM(units=n_a, return_sequences=True))(X)

    # Initialize empty list of outputs
    outputs = []
    
    for y in range(Ty):
        context = one_step_attention(a, s)

        _, s, c = POST_ATTENTION_LSTM_CELL(inputs = context, initial_state = [s, c])

        output = OUTPUT_LAYER(s)
        outputs.append(output)

    model = tf.keras.Model(inputs = [X, s0, c0], outputs = outputs)
    return model

if __name__ == "__main__":

    Tx = 30
    Ty = 10
    m = 10

    REPEATOR = tf.keras.layers.RepeatVector(Tx)
    CONCATENATOR = tf.keras.layers.Concatenate(axis=-1)
    DENSOR1 = tf.keras.layers.Dense(units=10, activation='tanh')
    DENSOR2 = tf.keras.layers.Dense(units=1, activation='relu')
    ACTIVATOR = tf.keras.layers.Activation(softmax, name = 'attention_weights')
    DOTOR = tf.keras.layers.Dot(axes=1)

    dataset, human_vocab, machine_vocab, inv_machine = load_dataset(m)
    X, Y, Xoh, Yoh = preprocessing(dataset, human_vocab, machine_vocab, 30, 10)

    n_a = 32 # number of units for the pre-attention, bi-directional LSTM's hidden state 'a'
    n_s = 64 # number of units for the post-attention LSTM's hidden state "s"

    # Please note, this is the post attention LSTM cell.  
    POST_ATTENTION_LSTM_CELL = LSTM(n_s, return_state = True, name = 'post_attention_lstm') 
    OUTPUT_LAYER = Dense(len(machine_vocab), activation=softmax)

    model = modelf(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))

    optm = tf.keras.optimizers.Adam(learning_rate=0.005)
    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics='accuracy')
    print(model.summary())
    s0 = np.zeros((m, n_s))
    c0 = np.zeros((m, n_s))
    outputs = list(Yoh.swapaxes(0,1))

    model.fit([Xoh, s0, c0], outputs, epochs=10, batch_size=128)
